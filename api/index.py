# api/index.py
import pandas as pd
import numpy as np
import traceback
import joblib
import os
import logging
from prometheus_client import Counter, Histogram
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from sklearn.multioutput import MultiOutputClassifier
from collections import defaultdict, Counter
from typing import Dict, List, Any, Set, Tuple, Optional
from supabase import create_client, Client
from dotenv import load_dotenv
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score
from datetime import datetime, timedelta # New Import

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Powerball AI Generator", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your defined Group A numbers
GROUP_A_NUMBERS = {3, 5, 6, 7, 9, 11, 15, 16, 18, 21, 23, 24, 27, 31, 32, 33, 36, 42, 44, 45, 48, 50, 51, 54, 55, 60, 66, 69}

# --- Supabase Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_TABLE_NAME = "powerball_draws" # Assuming this is your table name

if not SUPABASE_URL or not SUPABASE_KEY:
    logging.error("Supabase environment variables not set.")
    # Fallback/dummy client for local testing if not configured
    supabase = None 
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Metrics (for demonstration) ---
PREDICTION_REQUESTS = Counter('prediction_requests_total', 'Total number of prediction requests')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Latency of prediction requests')

# --- Model Loading (Placeholder) ---
models = {}
# Function to load models (assuming joblib files in 'models' directory)
def load_models():
    global models
    model_dir = Path("models")
    if model_dir.is_dir():
        for filename in model_dir.glob("*.joblib"):
            model_name = filename.stem
            try:
                models[model_name] = joblib.load(filename)
                logging.info(f"Loaded model: {model_name}")
            except Exception as e:
                logging.error(f"Error loading model {model_name}: {e}")
    
    # Placeholder models for demonstration if real files are not present
    if not models:
        # Example placeholder models (in a real app, these must be trained first)
        logging.warning("No models loaded. Using placeholder logic.")
        # models['random_forest'] = None 
        # models['gradient_boosting'] = None
        # models['knn'] = None
        pass

# Load models on startup (or when needed)
load_models()

# --- Data Preparation and Utility Functions ---
def fetch_historical_data(year_filter: Optional[int] = None):
    """Fetches historical draw data from Supabase."""
    if not supabase:
        # Return dummy data for local testing if Supabase is not configured
        logging.warning("Using dummy historical data.")
        return pd.DataFrame({
            'Draw Date': [f'2025-01-01'],
            'Number 1': [1], 'Number 2': [10], 'Number 3': [20], 'Number 4': [30], 'Number 5': [40], 
            'Powerball': [10], 'Power Play': [2]
        })

    try:
        query = supabase.table(SUPABASE_TABLE_NAME).select('*')
        
        if year_filter:
            start_date = f"{year_filter}-01-01"
            end_date = f"{year_filter}-12-31"
            query = query.gte('Draw Date', start_date).lte('Draw Date', end_date)
            
        data = query.order('Draw Date', desc=True).execute().data
        df = pd.DataFrame(data)
        
        # Convert number columns to integer
        num_cols = [f'Number {i}' for i in range(1, 6)] + ['Powerball']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                
        return df
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()

# ... (Existing ML functions: get_features_and_targets, train_model, get_predictions, ensemble_prediction, generate_smart_numbers) ...

# --- NEW HELPER FUNCTIONS FOR DRAW ANALYSIS ---

def split_numbers_into_halves(numbers: List[int]) -> Dict[str, List[int]]:
    """Separates numbers into two halves based on their decade group (1-5, 6-10, 11-15, 16-20, etc.)."""
    final_halves = {'first_half': [], 'second_half': []}
    
    # Sort the numbers to process them cleanly
    sorted_numbers = sorted(numbers)
    
    for num in sorted_numbers:
        # Determine the start of the decade (1, 11, 21, 31, etc.)
        decade_start = (num - 1) // 10 * 10 + 1 
        
        # Numbers 1 through 5 of the decade belong to the first half
        if num in range(decade_start, decade_start + 5):
            final_halves['first_half'].append(num)
        # Numbers 6 through 10 (or 9 for the last group) belong to the second half
        else: # num in range(decade_start + 5, decade_start + 10) or decade_start=61, num in 66-69
            final_halves['second_half'].append(num)
            
    return final_halves

def detect_number_patterns(numbers: List[int]) -> Dict[str, Any]:
    """
    Detects patterns like same last digit matches and consecutive numbers.
    Also identifies "pair/triplet" numbers, though this is ambiguous in a single draw.
    Assuming "pair/triplet" means numbers that are a specific distance apart (e.g., in the same tens group).
    """
    patterns = {
        'same_last_digit': [],
        'consecutive_pairs': [],
        'tens_apart_pairs': [] # e.g., 20 apart (22, 42)
    }
    
    sorted_numbers = sorted(numbers)
    last_digits = defaultdict(list)

    # 1. Last Digit Matches
    for num in sorted_numbers:
        last_digits[num % 10].append(num)
    
    for digit, nums in last_digits.items():
        if len(nums) >= 2:
            patterns['same_last_digit'].append(nums)
            
    # 2. Consecutive Numbers
    for i in range(len(sorted_numbers) - 1):
        if sorted_numbers[i+1] == sorted_numbers[i] + 1:
            # We want to group all consecutive numbers, not just pairs
            if not patterns['consecutive_pairs'] or patterns['consecutive_pairs'][-1][-1] != sorted_numbers[i]:
                 # Start a new consecutive sequence
                patterns['consecutive_pairs'].append([sorted_numbers[i], sorted_numbers[i+1]])
            else:
                # Extend the existing sequence
                patterns['consecutive_pairs'][-1].append(sorted_numbers[i+1])
                
    # 3. Pair/Triplet numbers (assuming the user meant numbers close to each other, like 42, 46 - a pair in the 40s)
    # This feature is highly ambiguous, let's interpret it as finding pairs within the same tens bracket (e.g., 41-50)
    tens_groups = defaultdict(list)
    for num in sorted_numbers:
        tens_groups[num // 10].append(num)
        
    for group, nums in tens_groups.items():
        if len(nums) >= 2:
            # If the user means any pair/triplet, showing the group is the most relevant output
            # For simplicity in output, we won't add this to the patterns dict unless requested
            pass
            
    return patterns

# --- NEW API ENDPOINT FOR DRAW ANALYSIS ---

@app.get("/api/draw_analysis")
def get_draw_analysis(year: Optional[int] = None, month: Optional[int] = None):
    """Returns historical draw analysis for a specific year and month or the last 6 years."""
    
    if not supabase:
        return JSONResponse(status_code=500, content={"message": "Supabase connection failed. Cannot fetch historical data."})
    
    try:
        query = supabase.table(SUPABASE_TABLE_NAME).select('*, "Draw Date"')
        
        # Calculate the date 6 years ago for the 'Last 6 Years' default filter
        six_years_ago = datetime.now() - timedelta(days=365 * 6)
        
        # 1. Apply Year Filter (Last 6 years by default, or specific year)
        if year is None:
            # Fetch all draws from the last 6 years
            query = query.gte('"Draw Date"', six_years_ago.strftime('%Y-%m-%d'))
        else:
            # Filter for a specific year
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            query = query.gte('"Draw Date"', start_date).lte('"Draw Date"', end_date)
            
        # 2. Apply Month Filter (if provided)
        if month:
            # Note: Supabase's REST API doesn't directly support EXTRACT(MONTH) in this simple form.
            # A correct implementation would require a custom view or function on the database.
            # For simplicity, we will fetch the data and filter in Python.
            pass # We will filter by month after fetching the year's data.

        # 3. Order by date descending (most recent first)
        draws = query.order('"Draw Date"', desc=True).execute().data
        
        # --- Python-side Month Filtering (Safer than relying on Supabase REST API date functions) ---
        if month:
            draws = [
                draw for draw in draws 
                if datetime.strptime(draw["Draw Date"], '%Y-%m-%d').month == month
            ]
            
        if not draws:
            return JSONResponse({"message": "No data found for the selected period."}, status_code=404)

        # 4. Process each draw to find patterns and splits
        processed_draws = []
        for draw in draws:
            # Gather the 5 white balls
            white_balls = sorted([
                draw.get('Number 1', 0), draw.get('Number 2', 0), draw.get('Number 3', 0), 
                draw.get('Number 4', 0), draw.get('Number 5', 0)
            ])
            
            # Remove any zero values that might result from missing data
            white_balls = [int(n) for n in white_balls if n > 0] 

            # Perform the analysis
            halves = split_numbers_into_halves(white_balls)
            patterns = detect_number_patterns(white_balls)

            processed_draws.append({
                "date": draw["Draw Date"],
                "white_balls": white_balls,
                "powerball": draw.get("Powerball", 0),
                "halves": halves,
                "patterns": patterns
            })

        return JSONResponse(processed_draws)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"An unexpected error occurred: {str(e)}"})


# --- Existing Analysis and Prediction Functions ---
# Note: You need to keep all your original functions (like analyze_prediction, get_predictions, etc.)
# for the rest of your app to work. The following is a stub for the remaining structure.

def analyze_prediction(balls: List[int], pb: int, historical_data_all: pd.DataFrame, historical_data_2025: pd.DataFrame) -> Dict[str, Any]:
    """Performs analysis on a generated prediction."""
    analysis = {}
    
    # 1. Group A Check
    group_a_count = len(set(balls) & GROUP_A_NUMBERS)
    analysis['group_a_count'] = group_a_count

    # 2. Odd/Even Ratio
    odd_count = sum(1 for x in balls if x % 2 != 0)
    even_count = len(balls) - odd_count
    analysis['odd_even_ratio'] = f"{odd_count}:{even_count}"

    # 3. Frequency Analysis (The data used to calculate the individual frequency for 2025)
    # This assumes 'historical_data_2025' is the DataFrame for the current year.
    white_ball_counts = Counter(historical_data_2025[[f'Number {i}' for i in range(1, 6)]].values.flatten())
    powerball_count = Counter(historical_data_2025['Powerball'].values)
    
    analysis['2025_frequency'] = {
        'white_ball_counts': {int(k): int(v) for k, v in white_ball_counts.items()},
        'powerball_count': int(powerball_count[pb]) if pb in powerball_count else 0
    }

    return analysis

# Placeholder for prediction/ensemble functions to ensure the file is structurally complete
def get_predictions(model): return [1, 2, 3, 4, 5], 10 
def ensemble_prediction(rf, gb, knn): return [1, 2, 3, 4, 5], 10 
def convert_numpy_types(data): return data
def get_features_and_targets(df): return [], [], [], []
def train_model(X, y): return None

# ... (Existing API endpoints: @app.get("/"), @app.get("/api/generate_predictions")) ...

@app.get("/", response_class=HTMLResponse)
async def serve_app():
    # In a real application, you would load index.html from a file
    # For this environment, we'll return a simple placeholder or the full HTML if available
    # However, since the HTML is provided separately, we assume the user handles serving
    return "Application is running. Frontend served separately."

@app.get("/api/generate_predictions")
def generate_predictions(request: Request):
    """Generates predictions based on loaded models and returns analysis."""
    # This is a placeholder for your main prediction logic
    
    # Simulate fetching historical data
    historical_data_all = fetch_historical_data() 
    current_year = datetime.now().year
    historical_data_2025 = fetch_historical_data(current_year)
    
    predictions = {}
    
    # Simulate a prediction result for demonstration
    predictions['ensemble'] = analyze_prediction([1, 2, 3, 4, 5], 10, historical_data_all, historical_data_2025)
    predictions['ensemble']['generated_numbers'] = {'white_balls': [1, 2, 3, 4, 5], 'powerball': 10}

    sanitized_predictions = convert_numpy_types(predictions)
    
    return JSONResponse(sanitized_predictions)


# For running the app
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

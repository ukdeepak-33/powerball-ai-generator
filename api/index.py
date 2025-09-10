# api/index.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from supabase import create_client, Client
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import joblib
from typing import List, Optional, Dict, Any
from pathlib import Path
from collections import Counter
from datetime import datetime

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
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://yksxzbbcoitehdmsxqex.supabase.co")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlrc3h6YmJjb2l0ZWhkbXN4cWV4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk3NzMwNjUsImV4cCI6MjA2NTM0OTA2NX0.AzUD7wjR7VbvtUH27NDqJ3AlvFW0nCWpiN9ADG8T_t4")

SUPABASE_TABLE_NAME = 'powerball_draws'

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Load the trained model
try:
    MODEL = joblib.load('trained_model.joblib')
    print("✅ Trained ML model loaded successfully!")
except FileNotFoundError:
    print("⚠ No pre-trained model found. Using random generation as fallback")
    MODEL = None

def predict_numbers(historical_data):
    """Use ML model to predict numbers based on historical data"""
    if MODEL is None:
        # Fallback to random if no model
        white_balls = sorted(np.random.choice(range(1, 70), size=5, replace=False))
        powerball = np.random.randint(1, 27)
        return white_balls, powerball
    
    # Prepare features from historical data
    df = pd.DataFrame(historical_data)
    number_columns = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
    
    # Use the last 5 draws to make prediction
    if len(df) >= 5:
        recent_draws = df.iloc[-5:]
        
        # Create feature vector exactly like during training
        feature = np.zeros(69)
        for _, draw in recent_draws.iterrows():
            numbers = [draw['Number 1'], draw['Number 2'], draw['Number 3'],
                      draw['Number 4'], draw['Number 5']]
            for num in numbers:
                if isinstance(num, (int, float)) and 1 <= num <= 69:
                    feature[int(num)-1] += 1
        
        try:
            # Get prediction probabilities from model
            probabilities = MODEL.predict_proba([feature])
            
            # Get the predicted classes directly
            predictions = MODEL.predict([feature])
            
            # Flatten all predictions and get the most frequent numbers
            all_predicted_numbers = []
            for pred in predictions:
                # Get indices where prediction is 1 (number is present)
                predicted_indices = np.where(pred == 1)[0]
                predicted_numbers = [idx + 1 for idx in predicted_indices]
                all_predicted_numbers.extend(predicted_numbers)
            
            # If we got predictions, use the most frequent ones
            if all_predicted_numbers:
                number_counts = Counter(all_predicted_numbers)
                most_common = number_counts.most_common(10)  # Top 10 most frequent
                
                # Select 5 unique numbers
                selected_numbers = []
                for num, count in most_common:
                    if num not in selected_numbers and 1 <= num <= 69:
                        selected_numbers.append(num)
                    if len(selected_numbers) >= 5:
                        break
                
                # If we didn't get 5 numbers, fill with random
                while len(selected_numbers) < 5:
                    random_num = np.random.randint(1, 70)
                    if random_num not in selected_numbers:
                        selected_numbers.append(random_num)
                
                white_balls = sorted(selected_numbers)
            else:
                # Fallback to random
                white_balls = sorted(np.random.choice(range(1, 70), size=5, replace=False))
            
        except Exception as e:
            print(f"❌ ML prediction failed: {e}, using random fallback")
            white_balls = sorted(np.random.choice(range(1, 70), size=5, replace=False))
        
    else:
        # Not enough data, use random
        white_balls = sorted(np.random.choice(range(1, 70), size=5, replace=False))
    
    powerball = np.random.randint(1, 27)  # Powerball is separate
    
    # Convert numpy types to Python native types for JSON serialization
    white_balls = [int(num) for num in white_balls]
    powerball = int(powerball)
    
    return white_balls, powerball

def fetch_historical_draws(limit: int = 1000) -> List[dict]:
    """Fetches historical draws from Supabase"""
    try:
        response = supabase.table(SUPABASE_TABLE_NAME) \
                          .select('*') \
                          .order('"Draw Date"', desc=True) \
                          .limit(limit) \
                          .execute()
        return response.data
    except Exception as e:
        print(f"Error fetching data from Supabase: {e}")
        return []

def fetch_2025_draws() -> List[dict]:
    """Fetches only 2025 Powerball draws from Supabase"""
    try:
        response = supabase.table(SUPABASE_TABLE_NAME) \
                          .select('*') \
                          .gte('"Draw Date"', '2025-01-01') \
                          .lte('"Draw Date"', '2025-12-31') \
                          .order('"Draw Date"', desc=True) \
                          .execute()
        return response.data
    except Exception as e:
        print(f"Error fetching 2025 data: {e}")
        return []

def analyze_2025_frequency(white_balls, historical_data) -> Dict[str, Any]:
    """Analyze generated numbers against 2025 data only"""
    analysis = {
        '2025_draws_count': 0,
        'number_frequencies_2025': {},
        'analysis_available': False,
        'message': 'No 2025 data available for analysis'
    }
    
    if not historical_data or len(historical_data) == 0:
        return analysis
    
    df = pd.DataFrame(historical_data)
    number_columns = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
    
    analysis['2025_draws_count'] = len(df)
    analysis['analysis_available'] = True
    analysis['message'] = f'Analyzed against {len(df)} 2025 draws'
    
    # Get all numbers drawn in 2025
    all_2025_numbers = []
    for _, draw in df.iterrows():
        all_2025_numbers.extend([draw[col] for col in number_columns])
    
    number_counts_2025 = Counter(all_2025_numbers)
    total_draws_2025 = len(df)
    
    # Analyze each generated number
    for num in white_balls:
        count_2025 = number_counts_2025.get(num, 0)
        analysis['number_frequencies_2025'][num] = {
            'count': count_2025,
            'percentage': f"{(count_2025 / total_draws_2025 * 100):.1f}%" if total_draws_2025 > 0 else "0.0%",
            'status': 'Hot' if count_2025 >= 3 else 'Warm' if count_2025 >= 1 else 'Cold'
        }
    
    return analysis

def check_historical_matches(white_balls, powerball, historical_data) -> Dict[str, Any]:
    """Check how many numbers match historical draws"""
    analysis = {
        'exact_matches': 0,
        'partial_matches': [],
        'max_matches_found': 0,
        'most_recent_match': None,
        'match_analysis': []
    }
    
    if not historical_data:
        return analysis
    
    df = pd.DataFrame(historical_data)
    number_columns = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
    
    generated_set = set(white_balls)
    
    for _, draw in df.iterrows():
        draw_numbers = [draw[col] for col in number_columns]
        draw_set = set(draw_numbers)
        
        common_numbers = generated_set & draw_set
        match_count = len(common_numbers)
        
        if match_count > analysis['max_matches_found']:
            analysis['max_matches_found'] = match_count
        
        if match_count >= 3:  # Only track significant matches
            match_info = {
                'draw_date': draw['Draw Date'],
                'match_count': match_count,
                'common_numbers': list(common_numbers),
                'powerball_match': powerball == draw['Powerball'],
                'exact_match': (match_count == 5) and (powerball == draw['Powerball'])
            }
            
            analysis['partial_matches'].append(match_info)
            
            if match_info['exact_match']:
                analysis['exact_matches'] += 1
            
            # Track most recent significant match
            if analysis['most_recent_match'] is None or draw['Draw Date'] > analysis['most_recent_match']['draw_date']:
                analysis['most_recent_match'] = match_info
    
    # Sort partial matches by most recent first
    analysis['partial_matches'].sort(key=lambda x: x['draw_date'], reverse=True)
    
    # Keep only top 5 most recent significant matches
    analysis['partial_matches'] = analysis['partial_matches'][:5]
    
    return analysis

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the HTML homepage"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Powerball AI Generator</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #2c3e50; text-align: center; }
            .btn { background: #e74c3c; color: white; padding: 15px 30px; border: none; border-radius: 5px; 
                  font-size: 18px; cursor: pointer; display: block; margin: 20px auto; }
            .btn:hover { background: #c0392b; }
        </style>
    </head>
    <body>
        <h1>🎰 Powerball AI Generator</h1>
        <p style="text-align: center;">API is running successfully! 🚀</p>
        <div style="text-align: center;">
            <a href="/generate" style="text-decoration: none;">
                <button class="btn">Generate Numbers</button>
            </a>
            <a href="/analyze" style="text-decoration: none;">
                <button class="btn" style="background: #3498db;">Analyze Trends</button>
            </a>
            <a href="/docs" style="text-decoration: none;">
                <button class="btn" style="background: #27ae60;">API Documentation</button>
            </a>
        </div>
        <p style="text-align: center; margin-top: 30px;">
            Your AI-powered Powerball number generator is ready to use!
        </p>
    </body>
    </html>
    """)

@app.get("/generate")
async def generate_numbers():
    """Generate Powerball numbers with analysis"""
    try:
        # Fetch historical data for prediction and analysis
        historical_data = fetch_historical_draws(limit=1000)
        
        if not historical_data:
            raise HTTPException(status_code=404, detail="No historical data found")
        
        # Fetch 2025 data for frequency display
        data_2025 = fetch_2025_draws()
        
        # Generate numbers using ML model (based on historical data)
        white_balls, powerball = predict_numbers(historical_data)
        
        # Ensure no exact historical matches
        historical_check = check_historical_matches(white_balls, powerball, historical_data)
        
        # If exact match found, generate new numbers (safety check)
        max_attempts = 10
        attempt = 0
        while historical_check['exact_matches'] > 0 and attempt < max_attempts:
            print(f"⚠ Exact match found, generating new numbers (attempt {attempt + 1})")
            white_balls, powerball = predict_numbers(historical_data)
            historical_check = check_historical_matches(white_balls, powerball, historical_data)
            attempt += 1
        
        # Analyze against 2025 data for frequency display
        analysis_2025 = analyze_2025_frequency(white_balls, data_2025)
        
        # Basic analysis
        group_a_count = sum(1 for num in white_balls if num in GROUP_A_NUMBERS)
        odd_count = sum(1 for num in white_balls if num % 2 == 1)
        
        response_data = {
            "numbers": f"{', '.join(map(str, white_balls))} ⚡ Powerball: {powerball}",
            "basic_analysis": {
                "group_a_numbers": group_a_count,
                "odd_even_ratio": f"{odd_count} odd, {5 - odd_count} even",
                "total_numbers": len(white_balls)
            },
            "2025_frequency": analysis_2025,
            "historical_safety_check": {
                "exact_matches_found": historical_check['exact_matches'],
                "max_partial_matches": historical_check['max_matches_found'],
                "recent_significant_match": historical_check['most_recent_match']
            },
            "message": "AI-generated numbers with 2025 frequency analysis"
        }
        
        return JSONResponse(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating numbers: {str(e)}")

@app.get("/check-numbers")
async def check_numbers(numbers: str, powerball: int):
    """Check any numbers against historical data"""
    try:
        # Parse the input numbers
        white_balls = [int(num.strip()) for num in numbers.split(',')]
        
        if len(white_balls) != 5:
            raise HTTPException(status_code=400, detail="Please provide exactly 5 white ball numbers")
        
        if not (1 <= powerball <= 26):
            raise HTTPException(status_code=400, detail="Powerball must be between 1 and 26")
        
        # Fetch historical data
        historical_data = fetch_historical_draws(limit=1000)
        
        if not historical_data:
            raise HTTPException(status_code=404, detail="No historical data found")
        
        # Check matches
        match_analysis = check_historical_matches(white_balls, powerball, historical_data)
        
        # Fetch 2025 data for frequency
        data_2025 = fetch_2025_draws()
        frequency_analysis = analyze_2025_frequency(white_balls, data_2025)
        
        # Basic analysis
        group_a_count = sum(1 for num in white_balls if num in GROUP_A_NUMBERS)
        odd_count = sum(1 for num in white_balls if num % 2 == 1)
        
        response_data = {
            "checked_numbers": f"{', '.join(map(str, white_balls))} ⚡ Powerball: {powerball}",
            "basic_analysis": {
                "group_a_numbers": group_a_count,
                "odd_even_ratio": f"{odd_count} odd, {5 - odd_count} even"
            },
            "2025_frequency": frequency_analysis,
            "historical_matches": match_analysis,
            "message": "Number analysis complete"
        }
        
        return JSONResponse(response_data)
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid number format. Please use format: 1,2,3,4,5")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing numbers: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Service is running normally"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

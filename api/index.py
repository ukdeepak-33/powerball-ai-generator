# api/index.py - Updated with Individual 2025 Frequency Fix
import pandas as pd
import numpy as np
import traceback
import joblib
import os
import logging
from datetime import datetime, timedelta
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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# --- Helper Functions ---

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
        logger.error(f"Error fetching 2025 data: {e}")
        return []

def fetch_historical_draws(limit: int = 2000) -> List[dict]:
    """Fetches historical draws from Supabase"""
    try:
        response = supabase.table(SUPABASE_TABLE_NAME) \
                         .select('*') \
                         .order('"Draw Date"', desc=True) \
                         .limit(limit) \
                         .execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching data from Supabase: {e}")
        return []

def fetch_year_draws(year: int) -> List[dict]:
    """Fetches draws for a specific year"""
    try:
        response = supabase.table(SUPABASE_TABLE_NAME) \
                         .select('*') \
                         .gte('"Draw Date"', f'{year}-01-01') \
                         .lte('"Draw Date"', f'{year}-12-31') \
                         .order('"Draw Date"', desc=True) \
                         .execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching {year} data: {e}")
        return []

def get_current_year_number_frequencies(year: int = 2025) -> Dict[int, int]:
    """Get frequency count for all numbers 1-69 in the specified year"""
    try:
        year_data = fetch_year_draws(year)
        if not year_data:
            return {i: 0 for i in range(1, 70)}
        
        df = pd.DataFrame(year_data)
        number_columns = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
        
        # Count frequency of each number 1-69
        frequency_counts = {i: 0 for i in range(1, 70)}
        
        for _, draw in df.iterrows():
            for col in number_columns:
                number = int(draw[col])
                if 1 <= number <= 69:
                    frequency_counts[number] += 1
        
        return frequency_counts
        
    except Exception as e:
        logger.error(f"Error getting {year} frequencies: {e}")
        return {i: 0 for i in range(1, 70)}

def get_2025_frequencies(white_balls, powerball, historical_data):
    """Get INDIVIDUAL frequency counts for numbers in 2025 only - FIXED VERSION"""
    if not historical_data:
        return {
            'white_ball_counts': {int(num): 0 for num in white_balls},
            'powerball_count': 0,
            'total_2025_draws': 0,
            'individual_frequencies': {int(num): 0 for num in white_balls},
            'individual_percentages': {int(num): 0.0 for num in white_balls}
        }

    df = pd.DataFrame(historical_data)
    number_columns = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']

    # Calculate individual frequencies for each white ball number
    individual_frequencies = {}
    individual_percentages = {}
    total_draws = len(df)
    
    # Count how many times each specific number appeared in 2025
    for num in white_balls:
        python_num = int(num)
        count = 0
        
        # Check each draw in 2025 data
        for _, draw in df.iterrows():
            draw_numbers = [draw[col] for col in number_columns]
            if python_num in draw_numbers:
                count += 1
        
        individual_frequencies[python_num] = count
        individual_percentages[python_num] = round((count / total_draws) * 100, 1) if total_draws > 0 else 0.0

    # Calculate powerball frequency for 2025
    powerball_counts = Counter(df['Powerball'])
    python_powerball = int(powerball)
    powerball_count = powerball_counts.get(python_powerball, 0)
    powerball_percentage = round((powerball_count / total_draws) * 100, 1) if total_draws > 0 else 0.0

    return {
        'white_ball_counts': individual_frequencies,  # Use individual frequencies
        'powerball_count': powerball_count,
        'total_2025_draws': total_draws,
        'individual_frequencies': individual_frequencies,
        'individual_percentages': individual_percentages,
        'powerball_percentage': powerball_percentage,
        'year': '2025'
    }

def get_enhanced_frequency_analysis(white_balls, powerball, year: int = 2025):
    """Get enhanced frequency analysis showing individual frequencies for the year"""
    try:
        # Get all number frequencies for the year
        all_frequencies = get_current_year_number_frequencies(year)
        
        # Get specific data for the year
        historical_data_year = fetch_year_draws(year)
        
        # Calculate statistics
        total_draws = len(historical_data_year) if historical_data_year else 0
        
        # Individual frequencies for selected numbers
        selected_frequencies = {}
        for num in white_balls:
            selected_frequencies[int(num)] = all_frequencies.get(int(num), 0)
        
        # Powerball frequency
        powerball_frequency = 0
        if historical_data_year:
            df = pd.DataFrame(historical_data_year)
            powerball_counts = Counter(df['Powerball'])
            powerball_frequency = powerball_counts.get(int(powerball), 0)
        
        # Calculate percentages
        selected_percentages = {}
        if total_draws > 0:
            for num, freq in selected_frequencies.items():
                selected_percentages[num] = round((freq / total_draws) * 100, 1)
        
        # Find hot and cold numbers
        sorted_frequencies = sorted(all_frequencies.items(), key=lambda x: x[1], reverse=True)
        hot_numbers = [num for num, freq in sorted_frequencies[:10]]  # Top 10
        cold_numbers = [num for num, freq in sorted_frequencies[-10:]]  # Bottom 10
        
        return {
            'individual_frequencies': selected_frequencies,
            'individual_percentages': selected_percentages,
            'powerball_frequency': powerball_frequency,
            'powerball_percentage': round((powerball_frequency / total_draws) * 100, 1) if total_draws > 0 else 0,
            'total_draws': total_draws,
            'year': year,
            'hot_numbers': hot_numbers,
            'cold_numbers': cold_numbers,
            'average_frequency': sum(all_frequencies.values()) / 69 / total_draws * 100 if total_draws > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error in enhanced frequency analysis: {e}")
        return {
            'individual_frequencies': {int(num): 0 for num in white_balls},
            'individual_percentages': {int(num): 0 for num in white_balls},
            'powerball_frequency': 0,
            'powerball_percentage': 0,
            'total_draws': 0,
            'year': year,
            'error': str(e)
        }

def detect_number_patterns(white_balls: List[int]) -> Dict[str, Any]:
    """Detect various patterns in the generated numbers"""
    patterns = {
        'grouped_patterns': [],
        'tens_apart': [],
        'same_last_digit': [],
        'consecutive_pairs': [],
        'repeating_digit_pairs': []
    }

    if not white_balls or len(white_balls) < 2:
        return patterns

    sorted_balls = sorted(white_balls)

    decade_groups = defaultdict(list)
    for num in sorted_balls:
        decade = (num - 1) // 10
        decade_groups[decade].append(num)

    for decade, numbers in decade_groups.items():
        if len(numbers) >= 2:
            patterns['grouped_patterns'].append({
                'decade_range': f"{decade*10+1}-{(decade+1)*10}",
                'numbers': numbers
            })

    for i in range(len(sorted_balls)):
        for j in range(i + 1, len(sorted_balls)):
            num1, num2 = sorted_balls[i], sorted_balls[j]
            if abs(num1 - num2) % 10 == 0 and abs(num1 - num2) >= 10:
                patterns['tens_apart'].append([num1, num2])
            if num1 % 10 == num2 % 10:
                patterns['same_last_digit'].append([num1, num2])

    for i in range(len(sorted_balls) - 1):
        if sorted_balls[i + 1] - sorted_balls[i] == 1:
            patterns['consecutive_pairs'].append([sorted_balls[i], sorted_balls[i + 1]])

    repeating_numbers = [num for num in sorted_balls if num < 70 and num % 11 == 0 and num > 0]
    if len(repeating_numbers) >= 2:
        for i in range(len(repeating_numbers)):
            for j in range(i + 1, len(repeating_numbers)):
                patterns['repeating_digit_pairs'].append([
                    repeating_numbers[i],
                    repeating_numbers[j]
                ])

    return patterns

def analyze_pattern_history(patterns: Dict[str, Any], historical_data: List[dict]) -> Dict[str, Any]:
    """Analyze historical occurrence of detected patterns"""
    pattern_history = {
        'grouped_patterns': [],
        'tens_apart': [],
        'same_last_digit': [],
        'consecutive_pairs': [],
        'repeating_digit_pairs': []
    }

    if not historical_data:
        return pattern_history

    df = pd.DataFrame(historical_data)
    number_columns = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']

    for pattern_type, pattern_list in patterns.items():
        if not pattern_list:
            continue

        for pattern in pattern_list:
            history_info = {
                'pattern': pattern,
                'pattern_type': pattern_type,
                'current_year_count': 0,
                'total_count': 0,
                'years_count': defaultdict(int)
            }

            for _, draw in df.iterrows():
                draw_numbers = [draw[col] for col in number_columns]
                draw_date = draw.get('Draw Date', '')
                draw_year = draw_date[:4] if draw_date and isinstance(draw_date, str) else 'Unknown'

                try:
                    is_match = False
                    if pattern_type == 'grouped_patterns':
                        if all(num in draw_numbers for num in pattern.get('numbers', [])):
                            is_match = True
                    elif isinstance(pattern, list) and all(num in draw_numbers for num in pattern):
                        is_match = True

                    if is_match:
                        history_info['total_count'] += 1
                        history_info['years_count'][draw_year] += 1
                        if draw_year == '2025':
                            history_info['current_year_count'] += 1

                except Exception as e:
                    logger.error(f"Error analyzing pattern {pattern_type}: {pattern}, error: {e}")
                    continue

            pattern_history[pattern_type].append(history_info)

    return pattern_history

def format_pattern_analysis(pattern_history: Dict[str, Any]) -> str:
    """Format pattern analysis for display"""
    analysis_lines = []

    for pattern_type, patterns in pattern_history.items():
        if not patterns:
            if pattern_type == 'consecutive_pairs':
                analysis_lines.append("• Consecutive Pairs: None found")
            elif pattern_type == 'repeating_digit_pairs':
                analysis_lines.append("• Repeating Digit Pairs: None found")
            continue

        for pattern_info in patterns:
            pattern = pattern_info['pattern']
            pattern_type = pattern_info['pattern_type']
            current_count = pattern_info['current_year_count']
            total_count = pattern_info['total_count']
            years_count = pattern_info['years_count']

            if pattern_type == 'grouped_patterns':
                pattern_str = f"Grouped ({pattern['decade_range']}): {', '.join(map(str, pattern['numbers']))}"
            elif pattern_type == 'repeating_digit_pairs':
                pattern_str = f"Repeating Digit Pair: {', '.join(map(str, pattern))}"
            else:
                readable_type = pattern_type.replace('_', ' ').title()
                pattern_str = f"{readable_type}: {', '.join(map(str, pattern))}"

            years_info = []
            for year, count in years_count.items():
                if year != 'Unknown' and year != '2025':
                    years_info.append(f"{year}:{count}")
            years_info.sort(reverse=True)

            current_year_status = "Yes" if current_count > 0 else "No"
            current_year_info = f"2025: {current_year_status}"
            if current_count > 0:
                current_year_info += f" ({current_count} times)"

            if total_count > 0:
                years_summary = f" | Total: {total_count} times"
                if years_info:
                    years_summary += f" ({', '.join(years_info)})"
                analysis_lines.append(f"• {pattern_str} → {current_year_info}{years_summary}")
            else:
                analysis_lines.append(f"• {pattern_str} → Never occurred historically")

    if not analysis_lines:
        return "• No significant patterns detected"

    return "\n".join(analysis_lines)

def convert_numpy_types(data: Any) -> Any:
    """Recursively converts numpy data types to standard Python types."""
    if isinstance(data, dict):
        return {convert_numpy_types(key): convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(element) for element in data]
    elif isinstance(data, (np.integer, np.floating)):
        return int(data) if isinstance(data, np.integer) else float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

def generate_smart_numbers(historical_data):
    """Smart fallback number generation"""
    all_numbers = []
    for draw in historical_data:
        all_numbers.extend([draw['Number 1'], draw['Number 2'], draw['Number 3'],
                            draw['Number 4'], draw['Number 5']])

    number_counts = Counter(all_numbers)

    numbers, counts = zip(*number_counts.items())
    total = sum(counts)
    weights = [count/total for count in counts]

    selected_numbers = []
    while len(selected_numbers) < 5:
        num = np.random.choice(numbers, p=weights)
        if num not in selected_numbers:
            selected_numbers.append(num)

    powerball = np.random.randint(1, 27)

    return sorted(selected_numbers), powerball

# --- Model Training and Prediction ---

def create_prediction_features():
    """Creates a single feature matrix (X) for prediction."""
    feature_dict = {f'num_{i}': 0 for i in range(1, 70)}
    X = pd.DataFrame([feature_dict])
    return X

# Keep a log of which predictions were closer to actual draws
performance_log = {
    'random_forest_wins': 0,
    'gradient_boosting_wins': 0,
    'ties': 0
}

def ensemble_prediction(rf_model, gb_model, knn_model):
    """Get predictions from all three models and combine them using a simple voting ensemble."""
    logger.info("🧠 Using ensemble model for prediction...")

    try:
        # Get probabilities from Random Forest
        rf_probabilities_list = rf_model.predict_proba(create_prediction_features())
        rf_probs = np.array([prob[0, 1] if len(prob[0]) > 1 else prob[0, 0] for prob in rf_probabilities_list])

        # Get probabilities from Gradient Boosting
        gb_probabilities_list = gb_model.predict_proba(create_prediction_features())
        gb_probs = np.array([prob[0, 1] if len(prob[0]) > 1 else prob[0, 0] for prob in gb_probabilities_list])

        # Get probabilities from KNN
        knn_probabilities_list = knn_model.predict_proba(create_prediction_features())
        knn_probs = np.array([prob[0, 1] if len(prob[0]) > 1 else prob[0, 0] for prob in knn_probabilities_list])

        # Average the probabilities from all three models
        combined_probs = (rf_probs + gb_probs + knn_probs) / 3

        # Normalize probabilities to sum to 1
        total_prob = combined_probs.sum()
        if total_prob == 0:
            normalized_probs = np.full(69, 1/69)
        else:
            normalized_probs = combined_probs / total_prob

        # Select 5 numbers based on the combined probability distribution
        numbers = list(range(1, 70))
        final_white_balls = np.random.choice(numbers, size=5, p=normalized_probs, replace=False)

        # Predict the Powerball randomly
        final_powerball = np.random.randint(1, 27)

        return sorted(final_white_balls), final_powerball

    except Exception as e:
        logger.error(f"Ensemble prediction failed: {e}")
        historical_data = fetch_historical_draws(limit=2000)
        return generate_smart_numbers(historical_data)

def get_predictions(model):
    """Get predictions for white balls and powerball from a single model."""
    if model is None:
        # Fallback to random generation if no model is loaded
        historical_data = fetch_historical_draws(limit=2000)
        return generate_smart_numbers(historical_data)

    try:
        # Create a single feature row for prediction
        features = create_prediction_features()

        # Get probabilities from the model
        probabilities_list = model.predict_proba(features)

        # Reshape probabilities to a single array
        high_freq_probs = np.array([prob[0, 1] if len(prob[0]) > 1 else prob[0, 0] for prob in probabilities_list])

        # Normalize probabilities to sum to 1
        total_prob = high_freq_probs.sum()
        if total_prob == 0:
            # Fallback to uniform distribution if all probabilities are zero
            normalized_probs = np.full(69, 1/69)
        else:
            normalized_probs = high_freq_probs / total_prob

        # Select 5 numbers based on the probability distribution
        numbers = list(range(1, 70))
        selected_numbers = np.random.choice(numbers, size=5, p=normalized_probs, replace=False)

        # Predict the Powerball randomly (no model for this)
        powerball = np.random.randint(1, 27)

        return sorted(selected_numbers), powerball

    except Exception as e:
        logger.error(f"❌ Prediction failed for a model: {e}, using fallback")
        historical_data = fetch_historical_draws(limit=2000)
        return generate_smart_numbers(historical_data)

def split_numbers_into_halves(numbers: List[int]) -> Dict[str, List[int]]:
    """Separates numbers into two halves based on their decade group"""
    final_halves = {'first_half': [], 'second_half': []}
    
    sorted_numbers = sorted(numbers)
    
    for num in sorted_numbers:
        decade_start = (num - 1) // 10 * 10 + 1 
        
        if num in range(decade_start, decade_start + 5):
            final_halves['first_half'].append(num)
        else:
            final_halves['second_half'].append(num)
            
    return final_halves

# UPDATED analyze_prediction function with individual frequency fix
def analyze_prediction(white_balls, powerball, historical_data_all, historical_data_2025):
    """Helper function to perform analysis with INDIVIDUAL 2025 frequencies"""
    
    # Get enhanced frequency analysis for 2025
    frequency_2025 = get_enhanced_frequency_analysis(white_balls, powerball, 2025)
    
    patterns = detect_number_patterns(white_balls)
    pattern_analysis = analyze_pattern_history(patterns, historical_data_all)
    formatted_patterns = format_pattern_analysis(pattern_analysis)
    
    group_a_count = sum(1 for num in white_balls if num in GROUP_A_NUMBERS)
    odd_count = sum(1 for num in white_balls if num % 2 == 1)
    even_count = 5 - odd_count
    odd_even_ratio = f"{odd_count}-{even_count}"
    
    # Create detailed frequency message with INDIVIDUAL frequencies
    frequency_details = []
    for num in white_balls:
        freq = frequency_2025['individual_frequencies'].get(int(num), 0)
        pct = frequency_2025['individual_percentages'].get(int(num), 0)
        frequency_details.append(f"#{num}: {freq} times ({pct}%)")
    
    frequency_message = f"2025 Individual Frequencies: {' | '.join(frequency_details)}"
    
    powerball_freq = frequency_2025['powerball_frequency']
    powerball_pct = frequency_2025['powerball_percentage']
    powerball_message = f"Powerball #{powerball}: {powerball_freq} times ({powerball_pct}%)"
    
    analysis_message = f"{frequency_message}\n{powerball_message}\n\nPattern Analysis:\n{formatted_patterns}"
    
    return {
        "generated_numbers": {
            "white_balls": white_balls,
            "powerball": powerball,
            "lucky_group_a_numbers": list(GROUP_A_NUMBERS.intersection(set(white_balls)))
        },
        "analysis": {
            "group_a_count": group_a_count,
            "odd_even_ratio": odd_even_ratio,
            "message": analysis_message,
            "2025_frequency": frequency_2025,
            "frequency_breakdown": {
                "individual_white_balls": frequency_2025['individual_frequencies'],
                "individual_percentages": frequency_2025['individual_percentages'],
                "powerball_frequency": powerball_freq,
                "powerball_percentage": powerball_pct,
                "total_2025_draws": frequency_2025['total_draws']
            }
        }
    }

# API Endpoints

@app.get("/")
def read_root():
    """Returns the main HTML page"""
    try:
        file_path = Path(__file__).parent.parent / "templates" / "index.html"
        
        with open(file_path, "r") as f:
            return HTMLResponse(content=f.read())
            
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"message": "index.html not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error loading index.html: {str(e)}"})

@app.get("/health")
def health_check():
    """Health check endpoint with model status"""
    model_status = {}
    for name, model in models.items():
        model_status[name] = "loaded" if model is not None else "not_loaded"
    
    return {
        "status": "healthy", 
        "message": "Powerball AI Generator is running",
        "models": model_status,
        "frequency_fix": "Individual 2025 frequencies enabled"
    }

@app.get("/advanced_analytics")
def get_advanced_analytics():
    """Provide deeper statistical insights"""
    return {
        "frequency_analysis": {
            "hot_numbers": "Coming soon",
            "cold_numbers": "Coming soon", 
            "overdue_numbers": "Coming soon"
        },
        "pattern_analysis": {
            "seasonal_trends": "Coming soon",
            "number_correlations": "Coming soon",
            "draw_intervals": "Coming soon"
        },
        "prediction_confidence": "Coming soon"
    }

@app.get("/historical_analysis")
def get_historical_analysis(request: Request):
    """Returns a JSON object with historical analysis of Powerball draws."""
    try:
        historical_draws = fetch_historical_draws(limit=2000)
        if not historical_draws:
            raise HTTPException(status_code=404, detail="No historical data found.")

        df = pd.DataFrame(historical_draws)
        white_ball_columns = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
        
        # Helper for consecutive check
        def has_consecutive(row):
            sorted_nums = sorted([row[col] for col in white_ball_columns])
            for i in range(len(sorted_nums) - 1):
                if sorted_nums[i+1] - sorted_nums[i] == 1:
                    return True
            return False

        df['group_a_count'] = df[white_ball_columns].apply(
            lambda x: sum(1 for num in x if num in GROUP_A_NUMBERS), axis=1)
        df['odd_count'] = df[white_ball_columns].apply(
            lambda x: sum(1 for num in x if num % 2 == 1), axis=1)
        df['has_consecutive'] = df.apply(has_consecutive, axis=1)

        avg_group_a = df['group_a_count'].mean()
        consecutive_frequency = df['has_consecutive'].mean()
        avg_odd_count = df['odd_count'].mean()

        return JSONResponse({
            "historical_analysis": {
                "total_draws_analyzed": len(df),
                "average_group_a_numbers": round(avg_group_a, 2),
                "consecutive_draw_frequency": round(consecutive_frequency * 100, 2),
                "average_odd_numbers": round(avg_odd_count, 2)
            }
        })

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"message": e.detail})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"An unexpected error occurred: {str(e)}"})

# NEW ENDPOINTS for frequency analysis

@app.get("/frequency_analysis/{year}")
def get_frequency_analysis(year: int):
    """Get detailed frequency analysis for a specific year"""
    try:
        all_frequencies = get_current_year_number_frequencies(year)
        year_data = fetch_year_draws(year)
        total_draws = len(year_data)
        
        # Calculate percentages and statistics
        frequency_stats = {}
        for num in range(1, 70):
            freq = all_frequencies[num]
            percentage = round((freq / total_draws) * 100, 1) if total_draws > 0 else 0
            frequency_stats[num] = {
                'frequency': freq,
                'percentage': percentage,
                'expected_frequency': round(total_draws / 69, 1) if total_draws > 0 else 0,
                'deviation': round(freq - (total_draws / 69), 1) if total_draws > 0 else 0
            }
        
        # Get powerball frequencies
        powerball_frequencies = {}
        if year_data:
            df = pd.DataFrame(year_data)
            pb_counts = Counter(df['Powerball'])
            for pb in range(1, 27):
                freq = pb_counts.get(pb, 0)
                powerball_frequencies[pb] = {
                    'frequency': freq,
                    'percentage': round((freq / total_draws) * 100, 1) if total_draws > 0 else 0
                }
        
        # Sort by frequency
        sorted_white_balls = sorted(frequency_stats.items(), key=lambda x: x[1]['frequency'], reverse=True)
        sorted_powerballs = sorted(powerball_frequencies.items(), key=lambda x: x[1]['frequency'], reverse=True) if powerball_frequencies else []
        
        return JSONResponse({
            'year': year,
            'total_draws': total_draws,
            'white_ball_frequencies': frequency_stats,
            'powerball_frequencies': powerball_frequencies,
            'top_10_white_balls': dict(sorted_white_balls[:10]),
            'bottom_10_white_balls': dict(sorted_white_balls[-10:]),
            'top_5_powerballs': dict(sorted_powerballs[:5]) if sorted_powerballs else {},
            'bottom_5_powerballs': dict(sorted_powerballs[-5:]) if sorted_powerballs else {},
            'statistics': {
                'average_white_ball_frequency': round(sum(f['frequency'] for f in frequency_stats.values()) / 69, 1),
                'average_powerball_frequency': round(sum(f['frequency'] for f in powerball_frequencies.values()) / 26, 1) if powerball_frequencies else 0,
                'most_frequent_white_ball': sorted_white_balls[0][0] if sorted_white_balls else None,
                'least_frequent_white_ball': sorted_white_balls[-1][0] if sorted_white_balls else None,
                'most_frequent_powerball': sorted_powerballs[0][0] if sorted_powerballs else None,
                'least_frequent_powerball': sorted_powerballs[-1][0] if sorted_powerballs else None
            }
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get frequency analysis: {str(e)}"}
        )


# Global variable to hold models
models = {}

def load_models():
    """Load all machine learning models from .joblib files"""
    global models
    model_files = {
        'random_forest': 'enhanced_model_random_forest.joblib',
        'gradient_boosting': 'enhanced_model_gradient_boosting.joblib',
        'knn': 'enhanced_model_knn.joblib',
    }
    for name, file_path in model_files.items():
        if os.path.exists(file_path):
            try:
                models[name] = joblib.load(file_path)
                logger.info(f"✅ Successfully loaded {name} model.")
            except Exception as e:
                logger.error(f"❌ Error loading {name} model: {e}")
                models[name] = None
        else:
            logger.info(f"⚠️ Warning: {file_path} not found. Skipping {name} model.")
            models[name] = None

# Load models at startup
load_models()

@app.get("/generate")
@app.get("/generate_all")
def generate_numbers(request: Request):
    """Generates a set of Powerball numbers from each model and the ensemble with INDIVIDUAL 2025 frequencies"""
    try:
        historical_data_2025 = fetch_2025_draws()
        historical_data_all = fetch_historical_draws(limit=2000)
        
        predictions = {}
        
        # Generate predictions from each individual model
        if models.get('random_forest'):
            rf_balls, rf_pb = get_predictions(models['random_forest'])
            predictions['random_forest'] = analyze_prediction(rf_balls, rf_pb, historical_data_all, historical_data_2025)
            
        if models.get('gradient_boosting'):
            gb_balls, gb_pb = get_predictions(models['gradient_boosting'])
            predictions['gradient_boosting'] = analyze_prediction(gb_balls, gb_pb, historical_data_all, historical_data_2025)

        if models.get('knn'):
            knn_balls, knn_pb = get_predictions(models['knn'])
            predictions['knn'] = analyze_prediction(knn_balls, knn_pb, historical_data_all, historical_data_2025)
            
        # Generate ensemble prediction
        if all(model is not None for model in models.values()):
            ensemble_balls, ensemble_pb = ensemble_prediction(
                models['random_forest'],
                models['gradient_boosting'],
                models['knn']
            )
            predictions['ensemble'] = analyze_prediction(ensemble_balls, ensemble_pb, historical_data_all, historical_data_2025)
        else:
            # Fallback to smart generation if not all models are loaded
            fallback_balls, fallback_pb = generate_smart_numbers(historical_data_all)
            predictions['fallback'] = analyze_prediction(fallback_balls, fallback_pb, historical_data_all, historical_data_2025)

        # Add metadata about the frequency fix
        for model_name in predictions:
            predictions[model_name]['model_info'] = {
                'frequency_calculation': 'Individual 2025 frequencies',
                'data_source': '2025 draws only',
                'fix_applied': True
            }
        
        # Convert numpy types to native Python types before returning
        sanitized_predictions = convert_numpy_types(predictions)
        
        return JSONResponse(sanitized_predictions)
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"An unexpected error occurred: {str(e)}"})

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
            pass # We will filter by month after fetching the year's data.

        # 3. Order by date descending (most recent first)
        draws = query.order('"Draw Date"', desc=True).execute().data
        
        # --- Python-side Month Filtering ---
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

# REPLACE the two duplicate /current_year_stats endpoints in your api/index.py
# Delete BOTH existing @app.get("/current_year_stats") functions
# and replace with this SINGLE one:

@app.get("/current_year_stats")
def get_current_year_stats():
    """Get detailed frequency stats for 2025 with grouped display"""
    try:
        logger.info("Fetching 2025 draws...")
        year_data = fetch_year_draws(2025)
        
        if not year_data or len(year_data) == 0:
            logger.warning("No 2025 data found")
            return JSONResponse({
                "error": None,
                "message": "No 2025 data available yet",
                "year": 2025,
                "total_draws": 0,
                "number_details": {},
                "frequency_groups": {},
                "powerball_details": {},
                "last_updated": None
            })
        
        df = pd.DataFrame(year_data)
        total_draws = len(df)
        logger.info(f"Processing {total_draws} draws from 2025")
        
        number_columns = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
        
        # Calculate frequencies directly
        frequency_counts = {}
        for i in range(1, 70):
            frequency_counts[i] = 0
        
        # Count each number's occurrences
        for idx, draw in df.iterrows():
            for col in number_columns:
                try:
                    number = int(draw[col])
                    if 1 <= number <= 69:
                        frequency_counts[number] += 1
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(f"Error processing number in column {col} at row {idx}: {e}")
                    continue
        
        logger.info(f"Calculated frequencies for all numbers")
        
        # Group numbers by frequency
        frequency_groups = {}
        for num, freq in frequency_counts.items():
            if freq not in frequency_groups:
                frequency_groups[freq] = []
            frequency_groups[freq].append(num)
        
        # Sort frequency groups (highest frequency first)
        sorted_frequency_groups = dict(sorted(frequency_groups.items(), key=lambda x: x[0], reverse=True))
        
        # Calculate percentages for each number
        number_details = {}
        for num, freq in frequency_counts.items():
            percentage = round((freq / total_draws) * 100, 1) if total_draws > 0 else 0
            number_details[num] = {
                'frequency': int(freq),
                'percentage': float(percentage),
                'is_group_a': bool(num in GROUP_A_NUMBERS)
            }
        
        logger.info(f"Created number_details with {len(number_details)} entries")
        
        # Get powerball frequencies
        pb_counts = Counter(df['Powerball'])
        powerball_details = {}
        for pb in range(1, 27):
            freq = pb_counts.get(pb, 0)
            powerball_details[pb] = {
                'frequency': int(freq),
                'percentage': round((freq / total_draws) * 100, 1) if total_draws > 0 else 0
            }
        
        result = {
            'year': 2025,
            'total_draws': int(total_draws),
            'number_details': number_details,
            'frequency_groups': sorted_frequency_groups,
            'powerball_details': powerball_details,
            'last_updated': str(df.iloc[0]['Draw Date']) if not df.empty else None
        }
        
        logger.info(f"✅ Successfully calculated stats for {total_draws} draws")
        logger.info(f"✅ Number details count: {len(number_details)}")
        logger.info(f"✅ Sample number detail: {number_details.get(1, 'N/A')}")
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"❌ Error in get_current_year_stats: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Failed to get current year stats: {str(e)}",
                "year": 2025,
                "total_draws": 0,
                "number_details": {},
                "frequency_groups": {},
                "powerball_details": {},
                "last_updated": None
            }
        )

@app.get("/group_a_analysis")
def get_group_a_analysis(start_year: int = 2017, end_year: Optional[int] = None):
    """Analyze Group A numbers by year from 2017 to current"""
    try:
        if end_year is None:
            end_year = datetime.now().year
        
        analysis_by_year = {}
        
        for year in range(start_year, end_year + 1):
            year_data = fetch_year_draws(year)
            if not year_data:
                continue
            
            df = pd.DataFrame(year_data)
            number_columns = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
            
            # Count Group A occurrences per draw
            group_a_distribution = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
            
            for _, draw in df.iterrows():
                draw_numbers = [draw[col] for col in number_columns]
                group_a_count = sum(1 for num in draw_numbers if num in GROUP_A_NUMBERS)
                group_a_distribution[group_a_count] += 1
            
            total_draws = len(df)
            
            # Calculate percentages
            group_a_percentages = {}
            for count, occurrences in group_a_distribution.items():
                percentage = round((occurrences / total_draws) * 100, 1) if total_draws > 0 else 0
                group_a_percentages[count] = percentage
            
            analysis_by_year[year] = {
                'total_draws': total_draws,
                'distribution': group_a_distribution,
                'percentages': group_a_percentages,
                'most_common': max(group_a_distribution.items(), key=lambda x: x[1])[0]
            }
        
        # Calculate overall statistics
        overall_distribution = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
        total_draws_all_years = 0
        
        for year_stats in analysis_by_year.values():
            for count, occurrences in year_stats['distribution'].items():
                overall_distribution[count] += occurrences
            total_draws_all_years += year_stats['total_draws']
        
        overall_percentages = {}
        for count, occurrences in overall_distribution.items():
            percentage = round((occurrences / total_draws_all_years) * 100, 1) if total_draws_all_years > 0 else 0
            overall_percentages[count] = percentage
        
        return JSONResponse({
            'start_year': start_year,
            'end_year': end_year,
            'analysis_by_year': analysis_by_year,
            'overall_statistics': {
                'total_draws': total_draws_all_years,
                'distribution': overall_distribution,
                'percentages': overall_percentages
            },
            'group_a_numbers': list(GROUP_A_NUMBERS)
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get Group A analysis: {str(e)}"}
        )
        
# ADD this new endpoint to your api/index.py
# This replaces or enhances the existing /group_a_analysis endpoint

@app.get("/group_a_detailed_analysis")
def get_group_a_detailed_analysis(start_year: int = 2017, end_year: Optional[int] = None):
    """Analyze Group A numbers with actual combinations shown"""
    try:
        if end_year is None:
            end_year = datetime.now().year
        
        analysis_by_year = {}
        all_combinations = {5: [], 4: [], 3: [], 2: [], 1: [], 0: []}
        
        for year in range(start_year, end_year + 1):
            year_data = fetch_year_draws(year)
            if not year_data:
                continue
            
            df = pd.DataFrame(year_data)
            number_columns = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
            
            # Store combinations for each count level
            year_combinations = {5: [], 4: [], 3: [], 2: [], 1: [], 0: []}
            
            for _, draw in df.iterrows():
                draw_numbers = sorted([draw[col] for col in number_columns])
                draw_date = draw.get('Draw Date', 'Unknown')
                
                # Identify Group A numbers in this draw
                group_a_in_draw = sorted([num for num in draw_numbers if num in GROUP_A_NUMBERS])
                non_group_a_in_draw = sorted([num for num in draw_numbers if num not in GROUP_A_NUMBERS])
                
                group_a_count = len(group_a_in_draw)
                
                # Store the combination
                combination_info = {
                    'date': draw_date,
                    'group_a_numbers': group_a_in_draw,
                    'non_group_a_numbers': non_group_a_in_draw,
                    'all_numbers': draw_numbers,
                    'powerball': draw.get('Powerball', 0)
                }
                
                year_combinations[group_a_count].append(combination_info)
                all_combinations[group_a_count].append({**combination_info, 'year': year})
            
            # Calculate statistics for the year
            total_draws = len(df)
            distribution = {i: len(year_combinations[i]) for i in range(6)}
            percentages = {i: round((distribution[i] / total_draws) * 100, 1) if total_draws > 0 else 0 
                          for i in range(6)}
            
            # Find most common Group A combinations for each level
            most_common_combos = {}
            for count in [5, 4, 3, 2]:
                if year_combinations[count]:
                    # Group by Group A numbers combination
                    combo_counts = defaultdict(int)
                    combo_details = defaultdict(list)
                    
                    for combo in year_combinations[count]:
                        key = tuple(combo['group_a_numbers'])
                        combo_counts[key] += 1
                        combo_details[key].append({
                            'date': combo['date'],
                            'non_group_a': combo['non_group_a_numbers'],
                            'all_numbers': combo['all_numbers']
                        })
                    
                    # Sort by frequency
                    sorted_combos = sorted(combo_counts.items(), key=lambda x: x[1], reverse=True)
                    
                    most_common_combos[count] = [
                        {
                            'group_a_combination': list(combo),
                            'frequency': freq,
                            'occurrences': combo_details[combo][:5]  # Show up to 5 examples
                        }
                        for combo, freq in sorted_combos[:10]  # Top 10 combinations
                    ]
            
            analysis_by_year[year] = {
                'total_draws': total_draws,
                'distribution': distribution,
                'percentages': percentages,
                'most_common_combinations': most_common_combos,
                'all_combinations': year_combinations
            }
        
        # Overall statistics
        overall_distribution = {i: len(all_combinations[i]) for i in range(6)}
        total_draws_all_years = sum(overall_distribution.values())
        overall_percentages = {i: round((overall_distribution[i] / total_draws_all_years) * 100, 1) 
                               if total_draws_all_years > 0 else 0 for i in range(6)}
        
        # Find most common combinations across all years
        overall_most_common = {}
        for count in [5, 4, 3, 2]:
            if all_combinations[count]:
                combo_counts = defaultdict(int)
                combo_years = defaultdict(set)
                combo_details = defaultdict(list)
                
                for combo in all_combinations[count]:
                    key = tuple(combo['group_a_numbers'])
                    combo_counts[key] += 1
                    combo_years[key].add(combo['year'])
                    combo_details[key].append({
                        'date': combo['date'],
                        'year': combo['year'],
                        'non_group_a': combo['non_group_a_numbers']
                    })
                
                sorted_combos = sorted(combo_counts.items(), key=lambda x: x[1], reverse=True)
                
                overall_most_common[count] = [
                    {
                        'group_a_combination': list(combo),
                        'total_frequency': freq,
                        'appeared_in_years': sorted(list(combo_years[combo]), reverse=True),
                        'year_count': len(combo_years[combo]),
                        'recent_occurrences': sorted(combo_details[combo], 
                                                    key=lambda x: x['date'], 
                                                    reverse=True)[:5]
                    }
                    for combo, freq in sorted_combos[:15]  # Top 15 overall
                ]
        
        return JSONResponse({
            'start_year': start_year,
            'end_year': end_year,
            'analysis_by_year': analysis_by_year,
            'overall_statistics': {
                'total_draws': total_draws_all_years,
                'distribution': overall_distribution,
                'percentages': overall_percentages,
                'most_common_combinations': overall_most_common
            },
            'group_a_numbers': sorted(list(GROUP_A_NUMBERS))
        })
        
    except Exception as e:
        logger.error(f"Error in detailed Group A analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get detailed Group A analysis: {str(e)}"}
        )
# For running the app
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

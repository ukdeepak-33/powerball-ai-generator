# api/index.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from supabase import create_client, Client
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import joblib
from typing import List, Optional
from pathlib import Path
from collections import Counter

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
            
            print(f"🔍 Probabilities shape: {[p.shape for p in probabilities] if hasattr(probabilities, '__iter__') else 'Unknown'}")
            
            # Since the model output is unexpected, let's use a simpler approach
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
                from collections import Counter
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

def prepare_features(draws_df: pd.DataFrame) -> pd.DataFrame:
    """Engineers features from raw draw data"""
    # Use your actual column names from Supabase
    white_ball_columns = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
    
    # Ensure we have the required columns
    if not all(col in draws_df.columns for col in white_ball_columns):
        available_cols = list(draws_df.columns)
        print(f"Available columns: {available_cols}")
        print(f"Required columns: {white_ball_columns}")
        raise ValueError(f"DataFrame missing required white ball columns. Available: {available_cols}")
    
    # Create a copy to avoid modifying the original
    df = draws_df.copy()
    
    # FIXED: Replace deprecated applymap with map
    # Feature: Count of Group A numbers
    df['group_a_count'] = df[white_ball_columns].map(lambda x: x in GROUP_A_NUMBERS).sum(axis=1)
    
    # Feature: Odd/Even count
    df['odd_count'] = df[white_ball_columns].map(lambda x: x % 2 == 1).sum(axis=1)
    
    # Feature: Sum of white balls
    df['sum_white'] = df[white_ball_columns].sum(axis=1)
    
    # Feature: Check for consecutive numbers
    def has_consecutive(row):
        sorted_nums = sorted([row[col] for col in white_ball_columns])
        for i in range(len(sorted_nums)-1):
            if sorted_nums[i+1] - sorted_nums[i] == 1:
                return 1
        return 0
    
    df['has_consecutive'] = df.apply(has_consecutive, axis=1)
    
    return df

def get_2025_frequencies(white_balls, powerball, historical_data):
    """Get frequency counts for numbers in 2025 only"""
    if not historical_data:
        return {
            'white_ball_counts': {num: 0 for num in white_balls},
            'powerball_count': 0,
            'total_2025_draws': 0
        }
    
    df = pd.DataFrame(historical_data)
    number_columns = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
    
    # Count white ball frequencies in 2025
    white_ball_counts = {}
    all_white_balls = []
    for _, draw in df.iterrows():
        all_white_balls.extend([draw[col] for col in number_columns])
    
    white_ball_counter = Counter(all_white_balls)
    for num in white_balls:
        white_ball_counts[num] = white_ball_counter.get(num, 0)
    
    # Count powerball frequency in 2025
    powerball_counts = Counter(df['Powerball'])
    powerball_count = powerball_counts.get(powerball, 0)
    
    return {
        'white_ball_counts': white_ball_counts,
        'powerball_count': powerball_count,
        'total_2025_draws': len(df)
    }

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the HTML homepage"""
    index_path = Path("templates/index.html")
    if index_path.exists():
        with open(index_path, 'r') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    else:
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
        # Fetch historical data
        print("📊 Fetching historical data...")
        historical_data = fetch_historical_draws(limit=500)
        if not historical_data:
            print("❌ No historical data found")
            raise HTTPException(status_code=404, detail="No historical data found")
        
        print(f"✅ Found {len(historical_data)} historical draws")
        # Fetch 2025 data for frequency analysis
        print("📅 Fetching 2025 data...")
        data_2025 = fetch_2025_draws()
        print(f"✅ Found {len(data_2025)} draws in 2025")
        
        df = pd.DataFrame(historical_data)
        print(f"📋 DataFrame columns: {list(df.columns)}")
        print(f"📋 First row: {dict(df.iloc[0]) if len(df) > 0 else 'No data'}")
        
        # Prepare features
        print("🔧 Preparing features...")
        engineered_data = prepare_features(df)
        print("✅ Features prepared successfully")
        
        # Generate numbers using ML model
        print("🤖 Generating numbers with ML model...")
        white_balls, powerball = predict_numbers(historical_data)
        print(f"✅ Generated numbers: {white_balls}, Powerball: {powerball}")
        
        # Analyze the generated numbers
        group_a_count = sum(1 for num in white_balls if num in GROUP_A_NUMBERS)
        odd_count = sum(1 for num in white_balls if num % 2 == 1)

        # Get 2025 frequencies
        data_2025 = fetch_2025_draws()
        freq_2025 = get_2025_frequencies(white_balls, powerball, data_2025)

        # Detect patterns in generated numbers
        print("🔍 Detecting patterns...")
        patterns = detect_number_patterns(white_balls)
        print(f"✅ Patterns detected: {patterns}")
        
        # Analyze pattern history
        pattern_history = analyze_pattern_history(patterns, historical_data)
        pattern_analysis = format_pattern_analysis(pattern_history)
        print(f"📊 Pattern analysis complete")
        
        return JSONResponse({
            "generated_numbers": {
                "white_balls": [int(num) for num in white_balls],
                "powerball": int(powerball)
            },
            "analysis": {
                "group_a_count": int(group_a_count),
                "odd_even_ratio": f"{int(odd_count)} odd, {5 - int(odd_count)} even",
                "total_numbers_generated": len(white_balls),
                "message": "AI-generated numbers based on historical patterns",
                
                # NEW: Add 2025 frequency data
                "2025_frequency": {
                    "white_balls": freq_2025['white_ball_counts'],
                    "powerball": freq_2025['powerball_count'],
                    "total_draws_2025": freq_2025['total_2025_draws']
                },
                # NEW: Pattern analysis
                "pattern_analysis": pattern_analysis
            }
        })
    except Exception as e:
        print(f"❌ Error in generate_numbers: {str(e)}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/analyze")
async def analyze_trends():
    """Analyze historical trends"""
    try:
        historical_data = fetch_historical_draws(limit=1000)
        if not historical_data:
            raise HTTPException(status_code=404, detail="No historical data available for analysis")
        
        df = pd.DataFrame(historical_data)
        engineered_data = prepare_features(df)
        
        # Calculate various statistics
        avg_group_a = engineered_data['group_a_count'].mean()
        consecutive_frequency = engineered_data['has_consecutive'].mean()
        avg_odd_count = engineered_data['odd_count'].mean()
        
        return JSONResponse({
            "historical_analysis": {
                "total_draws_analyzed": len(engineered_data),
                "average_group_a_numbers": round(avg_group_a, 2),
                "consecutive_number_frequency": f"{consecutive_frequency * 100:.1f}%",
                "average_odd_numbers": round(avg_odd_count, 2),
                "data_timeframe": {
                    "oldest_draw": df['Draw Date'].min() if 'Draw Date' in df.columns else "Unknown",
                    "newest_draw": df['Draw Date'].max() if 'Draw Date' in df.columns else "Unknown"
                }
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Service is running normally"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

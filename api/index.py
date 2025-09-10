# api/index.py#
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
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "YOUR_ACTUAL_SUPABASE_ANON_KEY_GOES_HERE")

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
        
        # Get prediction probabilities from model
        probabilities = MODEL.predict_proba([feature])
        
        # Get the most likely numbers
        predicted_numbers = []
        for i, proba in enumerate(probabilities):
            # Get probabilities for this number position
            number_probs = proba[0][:, 1]  # Probability that number is present
            
            # Add some randomness to avoid always same numbers
            adjusted_probs = number_probs * np.random.uniform(0.8, 1.2, size=number_probs.shape)
            
            # Get top candidates
            top_candidates = list(np.argsort(adjusted_probs)[-10:])  # Top 10 likely numbers
            predicted_numbers.extend(top_candidates)
        
        # Get unique numbers and select top 5
        unique_numbers = list(set(predicted_numbers))
        white_balls = sorted([x + 1 for x in unique_numbers[-5:]])  # Convert back to 1-69
        
    else:
        # Not enough data, use random
        white_balls = sorted(np.random.choice(range(1, 70), size=5, replace=False))
    
    powerball = np.random.randint(1, 27)  # Powerball is separate
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

def prepare_features(draws_df: pd.DataFrame) -> pd.DataFrame:
    """Engineers features from raw draw data"""
    # Use your actual column names from Supabase
    white_ball_columns = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
    
    # Ensure we have the required columns
    if not all(col in draws_df.columns for col in white_ball_columns):
        raise ValueError("DataFrame missing required white ball columns")
    
    # Create a copy to avoid modifying the original
    df = draws_df.copy()
    
    # Feature: Count of Group A numbers
    df['group_a_count'] = df[white_ball_columns].applymap(lambda x: x in GROUP_A_NUMBERS).sum(axis=1)
    
    # Feature: Odd/Even count
    df['odd_count'] = df[white_ball_columns].applymap(lambda x: x % 2 == 1).sum(axis=1)
    
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
        historical_data = fetch_historical_draws(limit=500)
        if not historical_data:
            raise HTTPException(status_code=404, detail="No historical data found")
        
        df = pd.DataFrame(historical_data)
        
        # Prepare features
        engineered_data = prepare_features(df)
        
        # Generate numbers using ML model
        white_balls, powerball = predict_numbers(historical_data)
        
        # Analyze the generated numbers
        group_a_count = sum(1 for num in white_balls if num in GROUP_A_NUMBERS)
        odd_count = sum(1 for num in white_balls if num % 2 == 1)
        
        return JSONResponse({
            "generated_numbers": {
                "white_balls": white_balls,
                "powerball": powerball
            },
            "analysis": {
                "group_a_count": group_a_count,
                "odd_even_ratio": f"{odd_count} odd, {5 - odd_count} even",
                "total_numbers_generated": len(white_balls),
                "message": "AI-generated numbers based on historical patterns"
            }
        })
        
    except Exception as e:
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
    uvicorn.run(app, host="0.0.0.0", port=port)

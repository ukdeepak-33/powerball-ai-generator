# api/index.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import joblib
from typing import List, Optional

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

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_historical_draws(limit: int = 1000) -> List[dict]:
    """Fetches historical draws from Supabase"""
    try:
        response = supabase.table('draws') \
                          .select('*') \
                          .order('draw_date', desc=True) \
                          .limit(limit) \
                          .execute()
        return response.data
    except Exception as e:
        print(f"Error fetching data from Supabase: {e}")
        return []

def prepare_features(draws_df: pd.DataFrame) -> pd.DataFrame:
    """Engineers features from raw draw data"""
    white_ball_columns = ['num1', 'num2', 'num3', 'num4', 'num5']
    
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

@app.get("/")
async def root():
    return {"message": "Powerball AI Generator API is running!", "status": "healthy"}

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
        
        # For now, generate random numbers (Replace with your ML model later)
        white_balls = sorted(np.random.choice(range(1, 70), size=5, replace=False))
        powerball = np.random.randint(1, 27)
        
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
                "message": "Numbers generated successfully. Integrate ML model for smarter predictions."
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
                    "oldest_draw": df['draw_date'].min(),
                    "newest_draw": df['draw_date'].max()
                }
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

# Health check endpoint for Render
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Service is running normally"}

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
from typing import List, Dict, Any
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

# Supabase Configuration
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

# Core Functions
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
        print(f"Error fetching data: {e}")
        return []

def fetch_2025_draws() -> List[dict]:
    """Fetches only 2025 Powerball draws"""
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

def predict_numbers(historical_data):
    """Generate numbers using ML or random fallback"""
    if MODEL is None:
        white_balls = sorted(np.random.choice(range(1, 70), size=5, replace=False))
        powerball = np.random.randint(1, 27)
        return white_balls, powerball
    
    df = pd.DataFrame(historical_data)
    number_columns = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
    
    if len(df) >= 5:
        recent_draws = df.iloc[-5:]
        feature = np.zeros(69)
        
        for _, draw in recent_draws.iterrows():
            numbers = [draw[col] for col in number_columns]
            for num in numbers:
                if isinstance(num, (int, float)) and 1 <= num <= 69:
                    feature[int(num)-1] += 1
        
        try:
            predictions = MODEL.predict([feature])
            all_predicted_numbers = []
            
            for pred in predictions:
                predicted_indices = np.where(pred == 1)[0]
                predicted_numbers = [idx + 1 for idx in predicted_indices]
                all_predicted_numbers.extend(predicted_numbers)
            
            if all_predicted_numbers:
                number_counts = Counter(all_predicted_numbers)
                most_common = number_counts.most_common(10)
                
                selected_numbers = []
                for num, count in most_common:
                    if num not in selected_numbers and 1 <= num <= 69:
                        selected_numbers.append(num)
                    if len(selected_numbers) >= 5:
                        break
                
                while len(selected_numbers) < 5:
                    random_num = np.random.randint(1, 70)
                    if random_num not in selected_numbers:
                        selected_numbers.append(random_num)
                
                white_balls = sorted(selected_numbers)
            else:
                white_balls = sorted(np.random.choice(range(1, 70), size=5, replace=False))
            
        except Exception as e:
            print(f"ML prediction failed: {e}")
            white_balls = sorted(np.random.choice(range(1, 70), size=5, replace=False))
    else:
        white_balls = sorted(np.random.choice(range(1, 70), size=5, replace=False))
    
    powerball = np.random.randint(1, 27)
    return [int(num) for num in white_balls], int(powerball)

def analyze_2025_frequency(white_balls, historical_data) -> Dict[str, Any]:
    """Analyze numbers against 2025 data"""
    analysis = {
        '2025_draws_count': 0,
        'number_frequencies_2025': {},
        'analysis_available': False,
        'message': 'No 2025 data available'
    }
    
    if not historical_data:
        return analysis
    
    df = pd.DataFrame(historical_data)
    number_columns = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
    
    analysis['2025_draws_count'] = len(df)
    analysis['analysis_available'] = True
    analysis['message'] = f'Analyzed against {len(df)} 2025 draws'
    
    all_2025_numbers = []
    for _, draw in df.iterrows():
        all_2025_numbers.extend([draw[col] for col in number_columns])
    
    number_counts_2025 = Counter(all_2025_numbers)
    total_draws_2025 = len(df)
    
    for num in white_balls:
        count_2025 = number_counts_2025.get(num, 0)
        analysis['number_frequencies_2025'][num] = {
            'count': count_2025,
            'percentage': f"{(count_2025 / total_draws_2025 * 100):.1f}%" if total_draws_2025 > 0 else "0.0%",
            'status': 'Hot' if count_2025 >= 3 else 'Warm' if count_2025 >= 1 else 'Cold'
        }
    
    return analysis

# Simple HTML Interface
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Powerball AI Generator</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .btn { background: #e74c3c; color: white; padding: 15px 30px; border: none; border-radius: 8px; font-size: 18px; 
                  cursor: pointer; display: block; margin: 20px auto; }
            .btn:hover { background: #c0392b; }
            .numbers-display { font-size: 28px; font-weight: bold; text-align: center; margin: 30px 0; padding: 20px;
                             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 12px; }
            .powerball { color: #ffeb3b; font-weight: bold; font-size: 32px; }
            .analysis { margin: 20px 0; padding: 20px; background: #ecf0f1; border-radius: 10px; }
            .number-badge { display: inline-block; padding: 5px 12px; margin: 5px; border-radius: 20px; font-weight: bold; font-size: 14px; }
            .hot { background: #e74c3c; color: white; }
            .warm { background: #f39c12; color: white; }
            .cold { background: #7f8c8d; color: white; }
            .error { color: #e74c3c; text-align: center; padding: 10px; background: #ffeaea; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎰 Powerball AI Generator</h1>
            <form action="/generate" method="POST">
                <button type="submit" class="btn">Generate AI Numbers</button>
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/generate", response_class=HTMLResponse)
async def generate_numbers():
    """Simple number generation without complex historical matching"""
    try:
        historical_data = fetch_historical_draws(limit=500)
        
        if not historical_data:
            raise Exception("No historical data found")
        
        data_2025 = fetch_2025_draws()
        white_balls, powerball = predict_numbers(historical_data)
        analysis_2025 = analyze_2025_frequency(white_balls, data_2025)
        
        group_a_count = sum(1 for num in white_balls if num in GROUP_A_NUMBERS)
        odd_count = sum(1 for num in white_balls if num % 2 == 1)
        
        # Prepare frequency badges
        freq_badges = ""
        for num, info in analysis_2025['number_frequencies_2025'].items():
            freq_badges += f'<span class="number-badge {info["status"].lower()}">{num}: {info["count"]}</span>'
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Powerball AI Generator</title>
            <style>/* Same CSS as above */</style>
        </head>
        <body>
            <div class="container">
                <h1>🎰 Powerball AI Generator</h1>
                <form action="/generate" method="POST">
                    <button type="submit" class="btn">Generate New Numbers</button>
                </form>
                
                <div class="numbers-display">
                    {', '.join(map(str, white_balls))} <span class="powerball">⚡ Powerball: {powerball}</span>
                </div>
                
                <div class="analysis">
                    <h3>📊 Basic Analysis</h3>
                    <p>Group A Numbers: {group_a_count}</p>
                    <p>Odd/Even Ratio: {odd_count} odd, {5 - odd_count} even</p>
                </div>
                
                <div class="analysis">
                    <h3>📅 2025 Frequency Analysis</h3>
                    <p>2025 Draws Analyzed: {analysis_2025['2025_draws_count']}</p>
                    {freq_badges}
                </div>
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        error_html = f"""
        <div class="container">
            <h1>🎰 Powerball AI Generator</h1>
            <div class="error">
                <h3>Error Generating Numbers</h3>
                <p>{str(e)}</p>
            </div>
            <form action="/generate" method="POST">
                <button type="submit" class="btn">Try Again</button>
            </form>
        </div>
        """
        return HTMLResponse(content=error_html)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Service is running normally"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

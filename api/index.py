# api/index.py
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
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

# ======== CORE FUNCTIONS (ADDED BACK) ========

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

# ======== UI ENDPOINTS ========

def generate_numbers_internal():
    """Internal function to generate numbers"""
    # Fetch historical data for prediction and analysis
    historical_data = fetch_historical_draws(limit=1000)
    
    if not historical_data:
        raise Exception("No historical data found")
    
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
    
    return {
        "white_balls": white_balls,
        "powerball": powerball,
        "basic_analysis": {
            "group_a_numbers": group_a_count,
            "odd_even_ratio": f"{odd_count} odd, {5 - odd_count} even",
            "total_numbers": len(white_balls)
        },
        "2025_frequency": analysis_2025,
        "historical_safety_check": historical_check
    }

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Powerball AI Generator</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 900px; 
                margin: 0 auto; 
                padding: 20px; 
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            }
            h1 { 
                color: #2c3e50; 
                text-align: center; 
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #7f8c8d;
                margin-bottom: 30px;
            }
            .btn { 
                background: #e74c3c; 
                color: white; 
                padding: 15px 30px; 
                border: none; 
                border-radius: 8px; 
                font-size: 18px; 
                cursor: pointer; 
                display: block; 
                margin: 20px auto;
                transition: all 0.3s ease;
            }
            .btn:hover { 
                background: #c0392b; 
                transform: translateY(-2px);
                box-shadow: 0 6px 15px rgba(231, 76, 60, 0.3);
            }
            .numbers-display {
                font-size: 28px;
                font-weight: bold;
                text-align: center;
                margin: 30px 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 12px;
                box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            }
            .powerball {
                color: #ffeb3b;
                font-weight: bold;
                font-size: 32px;
            }
            .analysis-section {
                margin: 20px 0;
                padding: 20px;
                background: #ecf0f1;
                border-radius: 10px;
                border-left: 5px solid #3498db;
            }
            .analysis-toggle {
                background: #34495e;
                color: white;
                padding: 12px 20px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                margin: 10px 0;
                width: 100%;
                text-align: left;
                font-weight: bold;
            }
            .analysis-content {
                display: none;
                padding: 15px;
                background: white;
                border-radius: 8px;
                margin-top: 10px;
                border: 1px solid #bdc3c7;
            }
            .number-badge {
                display: inline-block;
                padding: 5px 12px;
                margin: 5px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 14px;
            }
            .hot { background: #e74c3c; color: white; }
            .warm { background: #f39c12; color: white; }
            .cold { background: #7f8c8d; color: white; }
            .match-good { background: #27ae60; color: white; }
            .match-warning { background: #f39c12; color: white; }
            .match-bad { background: #e74c3c; color: white; }
            .loading {
                display: none;
                text-align: center;
                color: #7f8c8d;
                font-size: 18px;
                margin: 20px 0;
            }
            .result-container {
                display: none;
            }
            .error {
                color: #e74c3c;
                text-align: center;
                padding: 10px;
                background: #ffeaea;
                border-radius: 5px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎰 Powerball AI Generator</h1>
            <p class="subtitle">AI-powered number generation with historical analysis</p>
            
            <form action="/generate-ui" method="POST">
                <button type="submit" class="btn">Generate AI Numbers</button>
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/generate-ui", response_class=HTMLResponse)
async def generate_numbers_ui(request: Request):
    """Generate numbers and display in UI"""
    try:
        # Generate numbers using internal function
        result = generate_numbers_internal()
        
        # Render the HTML with the results
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Powerball AI Generator</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    max-width: 900px; 
                    margin: 0 auto; 
                    padding: 20px; 
                    background-color: #f5f5f5;
                }}
                .container {{
                    background: white;
                    padding: 30px;
                    border-radius: 15px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                }}
                h1 {{ 
                    color: #2c3e50; 
                    text-align: center; 
                    margin-bottom: 10px;
                }}
                .subtitle {{
                    text-align: center;
                    color: #7f8c8d;
                    margin-bottom: 30px;
                }}
                .btn {{ 
                    background: #e74c3c; 
                    color: white; 
                    padding: 15px 30px; 
                    border: none; 
                    border-radius: 8px; 
                    font-size: 18px; 
                    cursor: pointer; 
                    display: block; 
                    margin: 20px auto;
                    transition: all 0.3s ease;
                }}
                .btn:hover {{ 
                    background: #c0392b; 
                    transform: translateY(-2px);
                    box-shadow: 0 6px 15px rgba(231, 76, 60, 0.3);
                }}
                .numbers-display {{
                    font-size: 28px;
                    font-weight: bold;
                    text-align: center;
                    margin: 30px 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 12px;
                    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
                }}
                .powerball {{
                    color: #ffeb3b;
                    font-weight: bold;
                    font-size: 32px;
                }}
                .analysis-toggle {{
                    background: #34495e;
                    color: white;
                    padding: 12px 20px;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                    margin: 10px 0;
                    width: 100%;
                    text-align: left;
                    font-weight: bold;
                }}
                .analysis-content {{
                    display: none;
                    padding: 15px;
                    background: white;
                    border-radius: 8px;
                    margin-top: 10px;
                    border: 1px solid #bdc3c7;
                }}
                .number-badge {{
                    display: inline-block;
                    padding: 5px 12px;
                    margin: 5px;
                    border-radius: 20px;
                    font-weight: bold;
                    font-size: 14px;
                }}
                .hot {{ background: #e74c3c; color: white; }}
                .warm {{ background: #f39c12; color: white; }}
                .cold {{ background: #7f8c8d; color: white; }}
                .match-good {{ background: #27ae60; color: white; }}
                .match-warning {{ background: #f39c12; color: white; }}
                .match-bad {{ background: #e74c3c; color: white; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎰 Powerball AI Generator</h1>
                <p class="subtitle">AI-powered number generation with historical analysis</p>
                
                <form action="/generate-ui" method="POST">
                    <button type="submit" class="btn">Generate New Numbers</button>
                </form>
                
                <div class="numbers-display">
                    {', '.join(map(str, result['white_balls']))} <span class="powerball">⚡ Powerball: {result['powerball']}</span>
                </div>
                
                <button class="analysis-toggle" onclick="toggleAnalysis('basic-analysis')">
                    📊 Basic Analysis
                </button>
                <div id="basic-analysis" class="analysis-content">
                    <p><strong>Group A Numbers:</strong> {result['basic_analysis']['group_a_numbers']}</p>
                    <p><strong>Odd/Even Ratio:</strong> {result['basic_analysis']['odd_even_ratio']}</p>
                    <p><strong>Total Numbers:</strong> {result['basic_analysis']['total_numbers']}</p>
                </div>
                
                <button class="analysis-toggle" onclick="toggleAnalysis('2025-analysis')">
                    📅 2025 Frequency Analysis
                </button>
                <div id="2025-analysis" class="analysis-content">
                    <p><strong>2025 Draws Analyzed:</strong> {result['2025_frequency']['2025_draws_count']}</p>
                    {"".join([f'<span class="number-badge {info["status"].lower()}">{num}: {info["count"]} ({info["percentage"]})</span>' 
                             for num, info in result['2025_frequency']['number_frequencies_2025'].items()])}
                </div>
                
                <button class="analysis-toggle" onclick="toggleAnalysis('historical-analysis')">
                    📈 Historical Match Check
                </button>
                <div id="historical-analysis" class="analysis-content">
                    <p><strong>Exact Matches Found:</strong> <span class="{'match-bad' if result['historical_safety_check']['exact_matches_found'] > 0 else 'match-good'} number-badge">{result['historical_safety_check']['exact_matches_found']}</span></p>
                    <p><strong>Maximum Partial Matches:</strong> <span class="{'match-warning' if result['historical_safety_check']['max_partial_matches'] >= 4 else 'match-good'} number-badge">{result['historical_safety_check']['max_partial_matches']}</span></p>
                    {f'<p><strong>Most Recent Significant Match:</strong></p><p>Date: {result["historical_safety_check"]["recent_significant_match"]["draw_date"]}, Matches: {result["historical_safety_check"]["recent_significant_match"]["match_count"]}</p><p>Common Numbers: {", ".join(map(str, result["historical_safety_check"]["recent_significant_match"]["common_numbers"]))}</p><p>Powerball Match: {"Yes" if result["historical_safety_check"]["recent_significant_match"]["powerball_match"] else "No"}</p>' 
                     if result['historical_safety_check']['recent_significant_match'] else ''}
                </div>
            </div>

            <script>
                function toggleAnalysis(id) {{
                    const content = document.getElementById(id);
                    content.style.display = content.style.display === 'none' ? 'block' : 'none';
                }}
            </script>
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
            <form action="/generate-ui" method="POST">
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

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

# ... [Keep all your existing functions: predict_numbers, fetch_historical_draws, 
# fetch_2025_draws, analyze_2025_frequency, check_historical_matches] 
# They remain exactly the same as in the previous version ...

def generate_numbers_internal():
    """Internal function to generate numbers (same logic as before)"""
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
            
            <div id="loading" class="loading">
                <p>🤖 AI is analyzing historical patterns...</p>
            </div>
            
            <div id="result-container" class="result-container">
                <div id="numbers-display" class="numbers-display"></div>
                
                <button class="analysis-toggle" onclick="toggleAnalysis('basic-analysis')">
                    📊 Basic Analysis
                </button>
                <div id="basic-analysis" class="analysis-content"></div>
                
                <button class="analysis-toggle" onclick="toggleAnalysis('2025-analysis')">
                    📅 2025 Frequency Analysis
                </button>
                <div id="2025-analysis" class="analysis-content"></div>
                
                <button class="analysis-toggle" onclick="toggleAnalysis('historical-analysis')">
                    📈 Historical Match Check
                </button>
                <div id="historical-analysis" class="analysis-content"></div>
            </div>
        </div>

        <script>
            // Check if we have result data from server-side rendering
            const resultData = JSON.parse(document.getElementById('result-data')?.textContent || 'null');
            
            if (resultData) {
                displayResults(resultData);
            }
            
            function displayResults(data) {
                // Display numbers
                document.getElementById('numbers-display').innerHTML = 
                    `${data.white_balls.join(', ')} <span class="powerball">⚡ Powerball: ${data.powerball}</span>`;
                
                // Display basic analysis
                document.getElementById('basic-analysis').innerHTML = `
                    <p><strong>Group A Numbers:</strong> ${data.basic_analysis.group_a_numbers}</p>
                    <p><strong>Odd/Even Ratio:</strong> ${data.basic_analysis.odd_even_ratio}</p>
                    <p><strong>Total Numbers:</strong> ${data.basic_analysis.total_numbers}</p>
                `;
                
                // Display 2025 frequency analysis
                let freqHtml = `<p><strong>2025 Draws Analyzed:</strong> ${data['2025_frequency']['2025_draws_count']}</p>`;
                for (const [num, info] of Object.entries(data['2025_frequency']['number_frequencies_2025'])) {
                    freqHtml += `<span class="number-badge ${info.status.toLowerCase()}">${num}: ${info.count} (${info.percentage})</span>`;
                }
                document.getElementById('2025-analysis').innerHTML = freqHtml;
                
                // Display historical analysis
                let histHtml = `
                    <p><strong>Exact Matches Found:</strong> <span class="${data.historical_safety_check.exact_matches_found > 0 ? 'match-bad' : 'match-good'} number-badge">${data.historical_safety_check.exact_matches_found}</span></p>
                    <p><strong>Maximum Partial Matches:</strong> <span class="${data.historical_safety_check.max_partial_matches >= 4 ? 'match-warning' : 'match-good'} number-badge">${data.historical_safety_check.max_partial_matches}</span></p>
                `;
                
                if (data.historical_safety_check.recent_significant_match) {
                    const match = data.historical_safety_check.recent_significant_match;
                    histHtml += `
                        <p><strong>Most Recent Significant Match:</strong></p>
                        <p>Date: ${match.draw_date}, Matches: ${match.match_count}</p>
                        <p>Common Numbers: ${match.common_numbers.join(', ')}</p>
                        <p>Powerball Match: ${match.powerball_match ? 'Yes' : 'No'}</p>
                    `;
                }
                
                document.getElementById('historical-analysis').innerHTML = histHtml;
                
                // Show results
                document.getElementById('result-container').style.display = 'block';
            }
            
            function toggleAnalysis(id) {
                const content = document.getElementById(id);
                content.style.display = content.style.display === 'none' ? 'block' : 'none';
            }
        </script>
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
                /* Same CSS as above */
                body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }}
                .container {{ background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; margin-bottom: 10px; }}
                /* ... include all the same CSS styles ... */
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎰 Powerball AI Generator</h1>
                <p class="subtitle">AI-powered number generation with historical analysis</p>
                
                <form action="/generate-ui" method="POST">
                    <button type="submit" class="btn">Generate New Numbers</button>
                </form>
                
                <div id="result-container" class="result-container" style="display: block;">
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

# Keep the API endpoint for other clients
@app.get("/generate-api")
async def generate_numbers_api():
    """API endpoint for generating numbers (for other clients)"""
    try:
        result = generate_numbers_internal()
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating numbers: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Service is running normally"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

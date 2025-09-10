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

# Core Functions (keep the same as before)

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
            .number-badge { display: inline-block; padding: 8px 15px; margin: 8px; border-radius: 20px; font-weight: bold; font-size: 14px; }
            .hot { background: #e74c3c; color: white; }
            .warm { background: #f39c12; color: white; }
            .cold { background: #7f8c8d; color: white; }
            .error { color: #e74c3c; text-align: center; padding: 10px; background: #ffeaea; border-radius: 5px; margin: 10px 0; }
            .loading { display: none; text-align: center; color: #7f8c8d; font-size: 18px; margin: 20px 0; }
            .result-container { display: none; }
            .frequency-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 15px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎰 Powerball AI Generator</h1>
            <button class="btn" onclick="generateNumbers()">Generate AI Numbers</button>
            
            <div id="loading" class="loading">
                <p>🤖 AI is analyzing historical patterns...</p>
            </div>
            
            <div id="result-container" class="result-container">
                <div id="numbers-display" class="numbers-display"></div>
                
                <div class="analysis">
                    <h3>📊 Basic Analysis</h3>
                    <div id="basic-analysis"></div>
                </div>
                
                <div class="analysis">
                    <h3>📅 2025 Frequency Analysis</h3>
                    <p id="draws-count"></p>
                    <div id="frequency-analysis" class="frequency-grid"></div>
                </div>
            </div>
        </div>

        <script>
            async function generateNumbers() {
                const loadingDiv = document.getElementById('loading');
                const resultDiv = document.getElementById('result-container');
                const numbersDisplay = document.getElementById('numbers-display');
                const basicAnalysis = document.getElementById('basic-analysis');
                const drawsCount = document.getElementById('draws-count');
                const frequencyAnalysis = document.getElementById('frequency-analysis');
                
                // Show loading, hide previous results
                loadingDiv.style.display = 'block';
                resultDiv.style.display = 'none';
                
                try {
                    const response = await fetch('/generate-api');
                    const data = await response.json();
                    
                    // Display numbers
                    numbersDisplay.innerHTML = `${data.white_balls.join(', ')} <span class="powerball">⚡ Powerball: ${data.powerball}</span>`;
                    
                    // Display basic analysis
                    basicAnalysis.innerHTML = `
                        <p><strong>Group A Numbers:</strong> ${data.basic_analysis.group_a_numbers}</p>
                        <p><strong>Odd/Even Ratio:</strong> ${data.basic_analysis.odd_even_ratio}</p>
                    `;
                    
                    // Display 2025 frequency analysis
                    drawsCount.innerHTML = `<strong>2025 Draws Analyzed:</strong> ${data['2025_frequency']['2025_draws_count']}`;
                    
                    let freqHtml = '';
                    for (const [num, info] of Object.entries(data['2025_frequency']['number_frequencies_2025'])) {
                        freqHtml += `
                            <div class="number-badge ${info.status.toLowerCase()}">
                                <div style="font-size: 18px; font-weight: bold;">${num}</div>
                                <div style="font-size: 12px;">${info.count} times (${info.percentage})</div>
                                <div style="font-size: 11px;">${info.status}</div>
                            </div>
                        `;
                    }
                    frequencyAnalysis.innerHTML = freqHtml;
                    
                    // Show results, hide loading
                    resultDiv.style.display = 'block';
                    loadingDiv.style.display = 'none';
                    
                } catch (error) {
                    loadingDiv.style.display = 'none';
                    alert('Error generating numbers. Please try again.');
                    console.error('Error:', error);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/generate-api")
async def generate_numbers_api():
    """API endpoint for generating numbers"""
    try:
        historical_data = fetch_historical_draws(limit=500)
        
        if not historical_data:
            raise HTTPException(status_code=404, detail="No historical data found")
        
        data_2025 = fetch_2025_draws()
        white_balls, powerball = predict_numbers(historical_data)
        analysis_2025 = analyze_2025_frequency(white_balls, data_2025)
        
        group_a_count = sum(1 for num in white_balls if num in GROUP_A_NUMBERS)
        odd_count = sum(1 for num in white_balls if num % 2 == 1)
        
        return {
            "white_balls": white_balls,
            "powerball": powerball,
            "basic_analysis": {
                "group_a_numbers": group_a_count,
                "odd_even_ratio": f"{odd_count} odd, {5 - odd_count} even"
            },
            "2025_frequency": analysis_2025
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating numbers: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Service is running normally"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

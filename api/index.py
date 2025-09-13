# api/index.py
import pandas as pd
import numpy as np
import traceback
import joblib
import os
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score

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
        print(f"Error fetching 2025 data: {e}")
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
        print(f"Error fetching data from Supabase: {e}")
        return []

def get_2025_frequencies(white_balls, powerball, historical_data):
    """Get frequency counts for numbers in 2025 only"""
    if not historical_data:
        return {
            'white_ball_counts': {int(num): 0 for num in white_balls},
            'powerball_count': 0,
            'total_2025_draws': 0
        }
    
    df = pd.DataFrame(historical_data)
    number_columns = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
    
    white_ball_counts = {}
    all_white_balls = []
    for _, draw in df.iterrows():
        all_white_balls.extend([draw[col] for col in number_columns])
    
    white_ball_counter = Counter(all_white_balls)
    for num in white_balls:
        python_num = int(num)
        white_ball_counts[python_num] = white_ball_counter.get(python_num, 0)
    
    powerball_counts = Counter(df['Powerball'])
    python_powerball = int(powerball)
    powerball_count = powerball_counts.get(python_powerball, 0)
    
    return {
        'white_ball_counts': white_ball_counts,
        'powerball_count': powerball_count,
        'total_2025_draws': len(df)
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
                    print(f"Error analyzing pattern {pattern_type}: {pattern}, error: {e}")
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
    """
    Creates a single feature matrix (X) for prediction.
    This is a row vector of zeros and ones to represent the numbers.
    """
    feature_dict = {f'num_{i}': 0 for i in range(1, 70)}
    X = pd.DataFrame([feature_dict])
    return X

# New function to get predictions from a single model
def predict_with_model(model):
    """Get predictions for white balls and powerball from a single model."""
    if model is None:
        historical_data = fetch_historical_draws(limit=2000)
        return generate_smart_numbers(historical_data)
        
    try:
        features = create_prediction_features()
        probabilities_list = model.predict_proba(features)
        high_freq_probs = np.array([prob[0, 1] for prob in probabilities_list])
        
        total_prob = high_freq_probs.sum()
        if total_prob == 0:
            normalized_probs = np.full(69, 1/69)
        else:
            normalized_probs = high_freq_probs / total_prob
        
        numbers = list(range(1, 70))
        selected_numbers = np.random.choice(numbers, size=5, p=normalized_probs, replace=False)
        powerball = np.random.randint(1, 27)
        
        return sorted(selected_numbers), powerball
        
    except Exception as e:
        print(f"❌ Prediction failed for a model: {e}, using fallback")
        return generate_smart_numbers(fetch_historical_draws(limit=2000))

def ensemble_prediction(rf_model, gb_model, knn_model):
    """
    Get predictions from all three models and combine them using a simple voting ensemble.
    """
    print("🧠 Using ensemble model for prediction...")
    
    # Get probabilities from Random Forest
    rf_probabilities_list = rf_model.predict_proba(create_prediction_features())
    rf_probs = np.array([prob[0, 1] for prob in rf_probabilities_list])
    
    # Get probabilities from Gradient Boosting
    gb_probabilities_list = gb_model.predict_proba(create_prediction_features())
    gb_probs = np.array([prob[0, 1] for prob in gb_probabilities_list])
    
    # Get probabilities from KNN
    knn_probabilities_list = knn_model.predict_proba(create_prediction_features())
    knn_probs = np.array([prob[0, 1] for prob in knn_probabilities_list])
    
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


# --- Main Application Logic ---

print("🚀 Starting application...")
RF_MODEL = None
GB_MODEL = None
KNN_MODEL = None

try:
    historical_data_for_training = fetch_historical_draws(limit=2000)
    if historical_data_for_training:
        print(f"✅ Fetched {len(historical_data_for_training)} records for model loading.")
        
        df = pd.DataFrame(historical_data_for_training)
        white_balls_list = df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.tolist()
        mlb = MultiLabelBinarizer(classes=range(1, 70))
        y_white_balls = mlb.fit_transform(white_balls_list)
        
        features = []
        for draw in historical_data_for_training:
            feature_dict = {f'num_{i}': 1 for i in [draw['Number 1'], draw['Number 2'], draw['Number 3'], draw['Number 4'], draw['Number 5']]}
            features.append(feature_dict)
        X = pd.DataFrame(features).fillna(0).astype(int)
        
        min_samples = min(len(X), len(y_white_balls))
        X = X.iloc[:min_samples]
        y_white_balls = y_white_balls[:min_samples]

        all_possible_features = [f'num_{i}' for i in range(1, 70)]
        X_reindexed = pd.DataFrame(0, index=X.index, columns=all_possible_features)
        X_reindexed.update(X)
        X = X_reindexed
        
        try:
            RF_MODEL = joblib.load('enhanced_model_random_forest.joblib')
            print("✅ Trained Random Forest model loaded.")
        except FileNotFoundError:
            print("⚠ Random Forest model not found. Training it now...")
            rf_instance = RandomForestClassifier(n_estimators=100, random_state=42)
            RF_MODEL = MultiOutputClassifier(rf_instance, n_jobs=-1)
            RF_MODEL.fit(X, y_white_balls)
            joblib.dump(RF_MODEL, 'enhanced_model_random_forest.joblib')
        
        try:
            GB_MODEL = joblib.load('enhanced_model_gradient_boosting.joblib')
            print("✅ Trained Gradient Boosting model loaded.")
        except FileNotFoundError:
            print("⚠ Gradient Boosting model not found. Training it now...")
            gb_instance = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            GB_MODEL = MultiOutputClassifier(gb_instance, n_jobs=-1)
            GB_MODEL.fit(X, y_white_balls)
            joblib.dump(GB_MODEL, 'enhanced_model_gradient_boosting.joblib')

        try:
            KNN_MODEL = joblib.load('enhanced_model_knn.joblib')
            print("✅ Trained KNN model loaded.")
        except FileNotFoundError:
            print("⚠ KNN model not found. Training it now...")
            knn_instance = KNeighborsClassifier(n_neighbors=5, weights='distance')
            KNN_MODEL = MultiOutputClassifier(knn_instance, n_jobs=-1)
            KNN_MODEL.fit(X, y_white_balls)
            joblib.dump(KNN_MODEL, 'enhanced_model_knn.joblib')

    else:
        print("❌ No historical data found for model training.")

except Exception as e:
    print(f"❌ Initial model loading or training failed: {e}")
    print(f"🔍 Traceback: {traceback.format_exc()}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
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
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; text-align: center; }
                h1 { color: #2c3e50; }
                .btn-container { display: flex; justify-content: center; gap: 10px; margin-top: 20px; }
                .btn { padding: 10px 20px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; color: white; }
                #ensemble-btn { background-color: #2c3e50; }
                #rf-btn { background-color: #e74c3c; }
                #gb-btn { background-color: #3498db; }
                #knn-btn { background-color: #27ae60; }
                .result-container { margin-top: 30px; }
            </style>
        </head>
        <body>
            <h1>Powerball AI Generator</h1>
            <div class="btn-container">
                <button id="ensemble-btn" class="btn">Generate (Ensemble)</button>
                <button id="rf-btn" class="btn">Generate (RF)</button>
                <button id="gb-btn" class="btn">Generate (GB)</button>
                <button id="knn-btn" class="btn">Generate (KNN)</button>
            </div>
            <div class="result-container">
                <h2 id="model-name"></h2>
                <div id="numbers-display"></div>
                <div id="analysis-display"></div>
            </div>
            <script>
                document.addEventListener('DOMContentLoaded', () => {
                    const buttons = {
                        'ensemble-btn': '/generate',
                        'rf-btn': '/generate/rf',
                        'gb-btn': '/generate/gb',
                        'knn-btn': '/generate/knn'
                    };

                    const numbersDisplay = document.getElementById('numbers-display');
                    const analysisDisplay = document.getElementById('analysis-display');
                    const modelNameDisplay = document.getElementById('model-name');

                    function displayNumbers(data) {
                        const { generated_numbers, analysis } = data;
                        
                        numbersDisplay.innerHTML = `
                            <h3>White Balls:</h3>
                            <p style="font-size: 24px; font-weight: bold;">${generated_numbers.white_balls.join(', ')}</p>
                            <h3>Powerball:</h3>
                            <p style="font-size: 24px; font-weight: bold;">${generated_numbers.powerball}</p>
                        `;

                        analysisDisplay.innerHTML = `
                            <h4>Analysis:</h4>
                            <pre>${JSON.stringify(analysis, null, 2)}</pre>
                        `;
                    }

                    function fetchNumbers(url, modelName) {
                        modelNameDisplay.textContent = `Generated by ${modelName}`;
                        fetch(url)
                            .then(response => {
                                if (!response.ok) {
                                    throw new Error('Network response was not ok');
                                }
                                return response.json();
                            })
                            .then(data => displayNumbers(data))
                            .catch(error => {
                                console.error('Error fetching numbers:', error);
                                numbersDisplay.textContent = 'Failed to generate numbers.';
                                analysisDisplay.textContent = '';
                            });
                    }

                    for (const buttonId in buttons) {
                        document.getElementById(buttonId).addEventListener('click', () => {
                            const url = buttons[buttonId];
                            const modelName = document.getElementById(buttonId).textContent.replace('Generate (', '').replace(')', '');
                            fetchNumbers(url, modelName);
                        });
                    }
                });
            </script>
        </body>
        </html>
        """)

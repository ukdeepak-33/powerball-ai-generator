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
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
    
# --- Model Training and Comparison ---

# Updated create_features function
def create_features(df):
    """
    Creates a feature matrix (X) for model training or prediction.
    Ensures the feature matrix always contains all 69 possible numbers.
    """
    white_balls_df = df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']]
    melted_df = white_balls_df.melt(value_name='Number').drop(columns='variable')
    
    # One-hot encode the numbers
    X = pd.get_dummies(melted_df, columns=['Number'], prefix='', prefix_sep='').groupby(melted_df.index).sum()
    
    # Get all possible numbers from 1 to 69
    all_possible_features = [f'num_present_{i}' for i in range(1, 70)]
    
    # Reindex the DataFrame to ensure all 69 columns exist
    # Fill any missing columns with zeros.
    X_reindexed = pd.DataFrame(0, index=X.index, columns=all_possible_features)
    X_reindexed.update(X.add_prefix('num_present_'))
    
    return X_reindexed

def train_and_evaluate_model(model_instance, historical_data, model_name):
    """
    Trains a multi-output classifier, evaluates its performance, and saves it.
    This version includes a train-test split for a more rigorous evaluation.
    
    Args:
        model_instance: The base scikit-learn model to use.
        historical_data (list): The list of historical draw data.
        model_name (str): The name of the model for logging and file naming.
    
    Returns:
        dict: A dictionary containing the model and its evaluation score.
    """
    print(f"\n🤖 Training and evaluating {model_name}...")
    
    if isinstance(historical_data, list):
        df = pd.DataFrame(historical_data)
    else:
        df = historical_data

    df = df.dropna()
    
    # Create features (X) and target (y)
    X = create_features(df)
    white_balls_list = df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.tolist()
    
    mlb = MultiLabelBinarizer(classes=range(1, 70))
    y_white_balls = mlb.fit_transform(white_balls_list)
    
    # Align the number of samples
    min_samples = min(X.shape[0], y_white_balls.shape[0])
    X = X[:min_samples]
    y_white_balls = y_white_balls[:min_samples]
    
    # --- The key change: Splitting the data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_white_balls, test_size=0.2, random_state=42
    )
    
    print(f"✅ Data split: Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples.")
    
    # Initialize and fit the multi-output classifier on the TRAINING data
    model = MultiOutputClassifier(model_instance, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate on the TESTING data
    y_pred = model.predict(X_test)
    score = jaccard_score(y_test, y_pred, average='samples')
    
    # Save the trained model
    model_path = f"enhanced_model_{model_name.lower().replace(' ', '_')}.joblib"
    joblib.dump(model, model_path)
    
    print(f"✅ {model_name} trained and saved with Jaccard Score on Test Data: {score:.4f}")
    
    return {"model": model, "score": score}


def compare_models(historical_data):
    """
    Orchestrates the training and comparison of multiple models, including Ridge and MLP.
    """
    models_to_test = {
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
        "Ridge Classifier": RidgeClassifier(random_state=42),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42)
    }
    
    results = {}
    best_model_name = ""
    best_score = -1
    
    for name, model_instance in models_to_test.items():
        try:
            result = train_and_evaluate_model(model_instance, historical_data, name)
            results[name] = result["score"]
            if result["score"] > best_score:
                best_score = result["score"]
                best_model_name = name
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n📊 --- Final Model Comparison ---")
    for name, score in results.items():
        print(f"  {name}: Jaccard Score = {score:.4f}")
    print("-----------------------------------")
    
    if best_model_name:
        best_model_path = f"enhanced_model_{best_model_name.lower().replace(' ', '_')}.joblib"
        try:
            best_model = joblib.load(best_model_path)
            print(f"✅ Best model ({best_model_name}) loaded successfully.")
            return best_model
        except Exception as e:
            print(f"❌ Error loading best model: {e}")
    return None

# Updated predict_enhanced_numbers function
def predict_enhanced_numbers(historical_data, model):
    """Generate numbers using enhanced prediction"""
    if model is None:
        return generate_smart_numbers(historical_data)
    
    try:
        # Get the most recent draws to create features for prediction
        recent_draws = historical_data[-5:] if len(historical_data) >= 5 else historical_data
        recent_df = pd.DataFrame(recent_draws)
        
        # Ensure features are correctly formatted with all 69 columns
        features = create_features(recent_df)
        
        # Predict probabilities for each of the 69 numbers
        try:
            # model.predict_proba returns a list of 69 arrays, one for each number
            probabilities_list = model.predict_proba(features)
            
            # Extract the probability of each number being drawn (class 1)
            high_freq_probs = [prob[0, 1] for prob in probabilities_list]
            
        except:
            # Fallback to direct prediction if probabilities are not available
            predictions = model.predict(features)[0]
            high_freq_probs = predictions.astype(float)
        
        number_probs = [(i+1, high_freq_probs[i]) for i in range(len(high_freq_probs))]
        number_probs.sort(key=lambda x: x[1], reverse=True)
        
        selected_numbers = []
        for num, prob in number_probs:
            if len(selected_numbers) >= 5:
                break
            if num not in selected_numbers:
                selected_numbers.append(num)
        
        while len(selected_numbers) < 5:
            fallback_nums, _ = generate_smart_numbers(historical_data)
            for num in fallback_nums:
                if num not in selected_numbers and len(selected_numbers) < 5:
                    selected_numbers.append(num)
        
        powerball = np.random.randint(1, 27)
        
        return sorted(selected_numbers), powerball
        
    except Exception as e:
        print(f"❌ Enhanced prediction failed: {e}, using fallback")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return generate_smart_numbers(historical_data)


# --- Main Application Logic ---

# Initial model loading and training
print("🚀 Starting application...")
MODEL = None
try:
    historical_data_for_training = fetch_historical_draws(limit=2000)
    if historical_data_for_training:
        print(f"✅ Fetched {len(historical_data_for_training)} records for model loading/training.")
        try:
            MODEL = joblib.load('enhanced_model_random_forest.joblib')
            print("✅ Trained Random Forest model loaded successfully!")
        except FileNotFoundError:
            print("⚠ Random Forest model not found. Starting model training and comparison...")
            MODEL = compare_models(historical_data_for_training)
    else:
        print("❌ No historical data found for model training.")
except Exception as e:
    print(f"❌ Initial model loading failed: {e}")
    print(f"🔍 Traceback: {traceback.format_exc()}")
    print("⚠ Using random generation as a fallback.")
    MODEL = None

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
        print("📊 Fetching historical data...")
        historical_data = fetch_historical_draws(limit=2000)
        if not historical_data:
            print("❌ No historical data found")
            raise HTTPException(status_code=404, detail="No historical data found")
        
        print(f"✅ Found {len(historical_data)} historical draws")
        
        print("🤖 Generating numbers with ML model...")
        white_balls, powerball = predict_enhanced_numbers(historical_data, MODEL)
        print(f"✅ Generated numbers: {white_balls}, Powerball: {powerball}")
        
        print("📅 Fetching 2025 data...")
        data_2025 = fetch_2025_draws()
        print(f"✅ Found {len(data_2025)} draws in 2025")
        
        group_a_count = sum(1 for num in white_balls if num in GROUP_A_NUMBERS)
        odd_count = sum(1 for num in white_balls if num % 2 == 1)

        freq_2025 = get_2025_frequencies(white_balls, powerball, data_2025)

        print("🔍 Detecting patterns...")
        patterns = detect_number_patterns(white_balls)
        print(f"✅ Patterns detected: {patterns}")
        
        pattern_history = analyze_pattern_history(patterns, historical_data)
        pattern_analysis = format_pattern_analysis(pattern_history)
        print(f"📊 Pattern analysis complete")
        
        json_compatible_analysis = {
            "group_a_count": int(group_a_count),
            "odd_even_ratio": f"{int(odd_count)} odd, {5 - int(odd_count)} even",
            "total_numbers_generated": len(white_balls),
            "message": "AI-generated numbers based on historical patterns",
            "2025_frequency": {
                "white_balls": freq_2025['white_ball_counts'],
                "powerball": freq_2025['powerball_count'],
                "total_draws_2025": freq_2025['total_2025_draws']
            },
            "pattern_analysis": pattern_analysis
        }
        
        return JSONResponse(content={
            "generated_numbers": {
                "white_balls": [int(x) for x in white_balls],
                "powerball": int(powerball),
            },
            "analysis": json_compatible_analysis
        })
        
    except Exception as e:
        print(f"❌ Error in generate_numbers: {e}")
        return JSONResponse(content={"error": str(e), "traceback": traceback.format_exc()}, status_code=500)
        
@app.get("/analyze")
async def analyze_trends():
    """Analyze historical trends"""
    try:
        historical_data = fetch_historical_draws()
        if not historical_data:
            raise HTTPException(status_code=404, detail="No historical data available for analysis")
        
        df = pd.DataFrame(historical_data)
        
        white_ball_columns = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']
        
        def has_consecutive(row):
            sorted_nums = sorted([row[col] for col in white_ball_columns])
            for i in range(len(sorted_nums)-1):
                if sorted_nums[i+1] - sorted_nums[i] == 1:
                    return 1
            return 0
        
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

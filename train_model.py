# train_model.py
import pandas as pd
import numpy as np
from supabase import create_client
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Your defined Group A numbers
GROUP_A_NUMBERS = {3, 5, 6, 7, 9, 11, 15, 16, 18, 21, 23, 24, 27, 31, 32, 33, 36, 42, 44, 45, 48, 50, 51, 54, 55, 60, 66, 69}

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")

def fetch_training_data(limit: int = 2000):
    """Fetch data for training"""
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    response = supabase.table('powerball_draws') \
                      .select('*') \
                      .order('draw_date', desc=True) \
                      .limit(limit) \
                      .execute()
    return pd.DataFrame(response.data)

def prepare_features(draws_df):
    """Prepare features for training"""
    white_ball_columns = ['num1', 'num2', 'num3', 'num4', 'num5']
    
    df = draws_df.copy()
    df['group_a_count'] = df[white_ball_columns].applymap(lambda x: x in GROUP_A_NUMBERS).sum(axis=1)
    df['odd_count'] = df[white_ball_columns].applymap(lambda x: x % 2 == 1).sum(axis=1)
    df['sum_white'] = df[white_ball_columns].sum(axis=1)
    
    return df

def create_training_data(features_df, window_size=10):
    """Create training examples from historical data"""
    X = []
    y = []
    
    for i in range(window_size, len(features_df)):
        # Use past 'window_size' draws as context
        context = features_df[['group_a_count', 'odd_count', 'sum_white']].iloc[i-window_size:i]
        context_flat = context.values.flatten()
        
        # Target is the next draw's numbers
        target = features_df.iloc[i][['num1', 'num2', 'num3', 'num4', 'num5']].values
        
        X.append(context_flat)
        y.append(target)
    
    return np.array(X), np.array(y)

def train_and_save_model():
    """Main training function"""
    print("🚀 Starting model training...")
    
    # 1. Fetch data
    print("📊 Fetching training data from Supabase...")
    data = fetch_training_data(limit=2000)
    print(f"✅ Retrieved {len(data)} historical draws")
    
    # 2. Prepare features
    print("🔧 Engineering features...")
    features_df = prepare_features(data)
    
    # 3. Create training data
    print("🎯 Creating training examples...")
    X, y = create_training_data(features_df, window_size=10)
    print(f"✅ Created {len(X)} training examples")
    
    # 4. Transform target to multi-label format
    y_transformed = np.zeros((len(y), 69), dtype=int)
    for i, numbers in enumerate(y):
        for num in numbers:
            y_transformed[i, num-1] = 1  # Number 1 -> index 0, number 69 -> index 68
    
    # 5. Train model
    print("🤖 Training machine learning model...")
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    model = MultiOutputClassifier(base_model)
    model.fit(X, y_transformed)
    
    # 6. Save model
    print("💾 Saving trained model...")
    joblib.dump(model, 'trained_model.joblib')
    
    # 7. Evaluate
    train_score = model.score(X, y_transformed)
    print(f"🎯 Model training complete! Training accuracy: {train_score:.4f}")
    print("✅ Model saved as 'trained_model.joblib'")

if __name__ == "__main__":
    train_and_save_model()

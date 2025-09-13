# train_models.py
import pandas as pd
import numpy as np
import os
import joblib
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Load environment variables from .env file
load_dotenv()

# --- Supabase Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://yksxzbbcoitehdmsxqex.supabase.co")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlrc3h6YmJjb2l0ZWhkbXN4cWV4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk3NzMwNjUsImV4cCI6MjA2NTM0OTA2NX0.AzUD7wjR7VbvtUH27NDqJ3AlvFW0nCWpiN9ADG8T_t4")
SUPABASE_TABLE_NAME = 'powerball_draws'

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def fetch_historical_draws(limit: int = 2000) -> list[dict]:
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

def train_and_save_models():
    """Trains and saves the machine learning models to joblib files."""
    try:
        historical_data = fetch_historical_draws(limit=2000)
        if not historical_data:
            print("❌ No historical data found to train models.")
            return

        print(f"✅ Fetched {len(historical_data)} records for training.")
        
        df = pd.DataFrame(historical_data)
        white_balls_list = df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.tolist()
        mlb = MultiLabelBinarizer(classes=range(1, 70))
        y_white_balls = mlb.fit_transform(white_balls_list)
        
        features = []
        for draw in historical_data:
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
        
        print("Starting model training...")

        # Random Forest Classifier
        print("Training Random Forest model...")
        rf_instance = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model = MultiOutputClassifier(rf_instance, n_jobs=-1)
        rf_model.fit(X, y_white_balls)
        joblib.dump(rf_model, 'enhanced_model_random_forest.joblib')
        print("✅ Random Forest model saved to enhanced_model_random_forest.joblib")

        # Gradient Boosting Classifier
        print("Training Gradient Boosting model...")
        gb_instance = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gb_model = MultiOutputClassifier(gb_instance, n_jobs=-1)
        gb_model.fit(X, y_white_balls)
        joblib.dump(gb_model, 'enhanced_model_gradient_boosting.joblib')
        print("✅ Gradient Boosting model saved to enhanced_model_gradient_boosting.joblib")

        # K-Nearest Neighbors Classifier
        print("Training KNN model...")
        knn_instance = KNeighborsClassifier(n_neighbors=5, weights='distance')
        knn_model = MultiOutputClassifier(knn_instance, n_jobs=-1)
        knn_model.fit(X, y_white_balls)
        joblib.dump(knn_model, 'enhanced_model_knn.joblib')
        print("✅ KNN model saved to enhanced_model_knn.joblib")

    except Exception as e:
        print(f"❌ An error occurred during training: {e}")

if __name__ == "__main__":
    train_and_save_models()
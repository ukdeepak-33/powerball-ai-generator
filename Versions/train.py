# train.py - Simple training script
import pandas as pd
import numpy as np
from supabase import create_client
import joblib
from sklearn.ensemble import RandomForestClassifier

# Your credentials - REPLACE THESE!
SUPABASE_URL = "https://yksxzbbcoitehdmsxqex.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlrc3h6YmJjb2l0ZWhkbXN4cWV4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjU1MjU2MTQsImV4cCI6MjA0MTEwMTYxNH0.0u0z0O7kMSv6q7SdVOqJp2nqD6n7qkYX6q6Q2qJZJqA"

print("🚀 Starting Powerball model training...")

# Connect to Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Fetch data
print("📊 Downloading historical data...")
response = supabase.table('powerball_draws').select('*').execute()
data = response.data

if not data:
    print("❌ No data found!")
else:
    print(f"✅ Found {len(data)} historical draws")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Simple training - predict if a number will appear
    all_numbers = list(range(1, 70))
    X = []  # Features: just using draw sequence for now
    y = []  # Target: which numbers appeared
    
    for i, draw in enumerate(data):
        if i > 0:  # Use previous draw to predict current
            prev_draw = data[i-1]
            current_draw = draw
            
            # Simple feature: which numbers were in previous draw
            prev_numbers = [prev_draw['num1'], prev_draw['num2'], prev_draw['num3'], 
                           prev_draw['num4'], prev_draw['num5']]
            
            current_numbers = [current_draw['num1'], current_draw['num2'], current_draw['num3'],
                              current_draw['num4'], current_draw['num5']]
            
            # Create feature vector (69 elements, 1 for each possible number)
            feature = [1 if num in prev_numbers else 0 for num in all_numbers]
            
            # Create target vector (69 elements, 1 for each number that appears)
            target = [1 if num in current_numbers else 0 for num in all_numbers]
            
            X.append(feature)
            y.append(target)
    
    if X:
        X = np.array(X)
        y = np.array(y)
        
        print(f"🤖 Training on {len(X)} examples...")
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Save the model
        joblib.dump(model, 'powerball_model.joblib')
        print("💾 Model saved as 'powerball_model.joblib'")
        
        # Test prediction
        test_prediction = model.predict_proba([X[-1]])
        print("🎯 Model training complete!")
        
    else:
        print("❌ Not enough data for training")
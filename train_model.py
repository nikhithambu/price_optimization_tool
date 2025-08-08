import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

def generate_sample_data(n_samples=2000):
    """Generate sample data for training the price optimization model"""
    print(f"Generating {n_samples} sample data points...")
    np.random.seed(42)
    
    # Define categories and seasons
    categories = ['electronics', 'clothing', 'home_garden', 'sports', 'books']
    seasons = ['spring', 'summer', 'autumn', 'winter']
    
    data = []
    
    for i in range(n_samples):
        if i % 500 == 0:
            print(f"Generated {i} samples...")
            
        # Generate base features
        category = np.random.choice(categories)
        season = np.random.choice(seasons)
        cost_price = np.random.uniform(5, 500)
        rating = np.random.uniform(1, 5)
        inventory_count = np.random.randint(0, 1000)
        
        # Generate competitor price based on cost price
        competitor_price = cost_price * np.random.uniform(1.2, 4.0)
        
        # Create category-specific pricing logic
        category_multipliers = {
            'electronics': np.random.uniform(1.3, 2.8),
            'clothing': np.random.uniform(1.5, 3.5),
            'home_garden': np.random.uniform(1.2, 2.5),
            'sports': np.random.uniform(1.4, 3.0),
            'books': np.random.uniform(1.1, 2.0)
        }
        
        # Season-specific adjustments
        season_adjustments = {
            'spring': np.random.uniform(0.95, 1.05),
            'summer': np.random.uniform(1.0, 1.15),
            'autumn': np.random.uniform(0.9, 1.0),
            'winter': np.random.uniform(0.95, 1.1)
        }
        
        # Calculate optimal price based on multiple factors
        base_price = cost_price * category_multipliers[category]
        
        # Rating influence (higher rating = higher price potential)
        rating_factor = 0.8 + (rating - 1) * 0.15
        
        # Inventory influence (lower inventory = slightly higher price)
        inventory_factor = 1.0 + max(0, (500 - inventory_count) / 5000)
        
        # Competitor price influence (don't go too far above competitor)
        competitor_factor = min(competitor_price / base_price, 1.3) if base_price > 0 else 1.0
        
        # Season adjustment
        season_factor = season_adjustments[season]
        
        # Calculate final optimal price
        optimal_price = base_price * rating_factor * inventory_factor * competitor_factor * season_factor
        
        # Add some realistic noise
        optimal_price *= np.random.uniform(0.92, 1.08)
        
        # Ensure minimum profit margin (at least 10% above cost)
        optimal_price = max(optimal_price, cost_price * 1.1)
        
        # Cap maximum price to be reasonable
        optimal_price = min(optimal_price, cost_price * 5.0)
        
        data.append({
            'cost_price': round(cost_price, 2),
            'category': category,
            'season': season,
            'rating': round(rating, 1),
            'inventory_count': inventory_count,
            'competitor_price': round(competitor_price, 2),
            'optimal_price': round(optimal_price, 2)
        })
    
    print(f"Generated {len(data)} data points successfully!")
    return pd.DataFrame(data)

def encode_categorical_features(df):
    """Encode categorical features"""
    print("Encoding categorical features...")
    df_encoded = df.copy()
    
    # Category encoding
    category_mapping = {
        'electronics': 0,
        'clothing': 1,
        'home_garden': 2,
        'sports': 3,
        'books': 4
    }
    df_encoded['category_encoded'] = df_encoded['category'].map(category_mapping)
    
    # Season encoding
    season_mapping = {
        'spring': 0,
        'summer': 1,
        'autumn': 2,
        'winter': 3
    }
    df_encoded['season_encoded'] = df_encoded['season'].map(season_mapping)
    
    print("Categorical encoding completed!")
    return df_encoded

def train_model():
    """Train the price optimization model"""
    print("=" * 50)
    print("PRICE OPTIMIZATION MODEL TRAINING")
    print("=" * 50)
    
    # Generate sample data
    df = generate_sample_data(2000)
    
    # Show data statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Categories: {df['category'].unique()}")
    print(f"Seasons: {df['season'].unique()}")
    print(f"Price range: ${df['optimal_price'].min():.2f} - ${df['optimal_price'].max():.2f}")
    
    # Encode categorical features
    df_encoded = encode_categorical_features(df)
    
    # Define features and target
    feature_columns = ['cost_price', 'category_encoded', 'season_encoded', 'rating', 'inventory_count', 'competitor_price']
    X = df_encoded[feature_columns]
    y = df_encoded['optimal_price']
    
    print(f"\nFeatures: {feature_columns}")
    print(f"Target: optimal_price")
    
    # Split data
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    # Evaluate model
    print("\nEvaluating model performance...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\nModel Performance:")
    print(f"Training MSE: {train_mse:.2f}")
    print(f"Testing MSE: {test_mse:.2f}")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Testing R² Score: {test_r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    # Save model
    print("\nSaving model to 'model.pkl'...")
    try:
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("✓ Model saved successfully!")
        
        # Verify model can be loaded
        with open('model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        print("✓ Model loading verification successful!")
        
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return None
    
    # Test with sample predictions
    print("\nSample Predictions (first 5 test samples):")
    sample_indices = range(min(5, len(X_test)))
    for i in sample_indices:
        actual = y_test.iloc[i]
        predicted = y_pred_test[i]
        cost = X_test.iloc[i]['cost_price']
        print(f"Sample {i+1}: Cost=${cost:.2f}, Actual=${actual:.2f}, Predicted=${predicted:.2f}, Diff=${abs(actual-predicted):.2f}")
    
    print("\n" + "=" * 50)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("You can now run 'python app.py' to start the web application.")
    print("=" * 50)
    
    return model

if __name__ == "__main__":
    try:
        model = train_model()
        if model is not None:
            print("\n🎉 Success! Model is ready for use.")
        else:
            print("\n❌ Training failed. Please check the error messages above.")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
from datetime import datetime
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Global variable to store the model
model = None
feature_names = ['cost_price', 'category_encoded', 'season_encoded', 'rating', 'inventory_count', 'competitor_price']

def load_model():
    """Load the trained ML model"""
    global model
    try:
        if os.path.exists('model.pkl'):
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            logger.info("Model loaded successfully")
            return True
        else:
            logger.error("Model file not found. Please run train_model.py first.")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def validate_input(data):
    """Validate user input"""
    errors = []
    
    try:
        # Check if all required fields are present
        required_fields = ['cost_price', 'category', 'season', 'rating', 'inventory_count', 'competitor_price']
        for field in required_fields:
            if field not in data or not data[field] or data[field] == '':
                errors.append(f"{field.replace('_', ' ').title()} is required")
        
        if errors:
            return False, errors
        
        # Validate numeric fields
        cost_price = float(data['cost_price'])
        rating = float(data['rating'])
        inventory_count = int(data['inventory_count'])
        competitor_price = float(data['competitor_price'])
        
        # Check ranges
        if cost_price <= 0:
            errors.append("Cost price must be greater than 0")
        if rating < 1 or rating > 5:
            errors.append("Rating must be between 1 and 5")
        if inventory_count < 0:
            errors.append("Inventory count cannot be negative")
        if competitor_price <= 0:
            errors.append("Competitor price must be greater than 0")
            
        return len(errors) == 0, errors
        
    except ValueError as e:
        errors.append("Please enter valid numeric values")
        return False, errors
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
        return False, errors

def encode_categorical_features(category, season):
    """Encode categorical features to match training data"""
    # Category encoding (same as in train_model.py)
    category_mapping = {
        'electronics': 0,
        'clothing': 1,
        'home_garden': 2,
        'sports': 3,
        'books': 4
    }
    
    # Season encoding
    season_mapping = {
        'spring': 0,
        'summer': 1,
        'autumn': 2,
        'winter': 3
    }
    
    category_encoded = category_mapping.get(category.lower(), 0)
    season_encoded = season_mapping.get(season.lower(), 0)
    
    return category_encoded, season_encoded

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle price prediction"""
    try:
        logger.info("Received prediction request")
        
        # Check if model is loaded
        if model is None:
            logger.info("Model not loaded, attempting to load...")
            if not load_model():
                return jsonify({
                    'success': False,
                    'error': 'Model not available. Please ensure model.pkl exists and run train_model.py if needed.'
                }), 500
        
        # Get form data - handle both form data and JSON
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        logger.info(f"Received data: {data}")
        
        # Validate input
        is_valid, errors = validate_input(data)
        if not is_valid:
            logger.warning(f"Validation failed: {errors}")
            return jsonify({
                'success': False,
                'error': 'Validation failed',
                'details': errors
            }), 400
        
        # Prepare features for prediction
        cost_price = float(data['cost_price'])
        category_encoded, season_encoded = encode_categorical_features(data['category'], data['season'])
        rating = float(data['rating'])
        inventory_count = int(data['inventory_count'])
        competitor_price = float(data['competitor_price'])
        
        # Create feature array
        features = np.array([[cost_price, category_encoded, season_encoded, rating, inventory_count, competitor_price]])
        logger.info(f"Features for prediction: {features}")
        
        # Make prediction
        predicted_price = model.predict(features)[0]
        logger.info(f"Raw prediction: {predicted_price}")
        
        # Ensure predicted price is reasonable (at least cost price + small margin)
        min_price = cost_price * 1.1  # At least 10% markup
        predicted_price = max(predicted_price, min_price)
        
        # Calculate profit margin
        profit_margin = ((predicted_price - cost_price) / cost_price) * 100
        
        # Prepare response
        response = {
            'success': True,
            'predicted_price': round(predicted_price, 2),
            'profit_margin': round(profit_margin, 2),
            'cost_price': cost_price,
            'competitor_price': competitor_price,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"Prediction successful: ${predicted_price:.2f} for {data['category']} product")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model on startup
    print("Starting Price Optimization Tool...")
    print("Loading model...")
    if load_model():
        print("Model loaded successfully!")
    else:
        print("Warning: Model not loaded. Please run 'python train_model.py' first.")
    
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

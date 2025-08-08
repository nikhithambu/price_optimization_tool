# Price Optimization Web Application

A machine learning-powered web application built with Flask that predicts optimal product pricing based on various factors including category, cost price, season, rating, inventory count, and competitor pricing.

## Features

- **ML-Powered Predictions**: Uses Random Forest regression model for accurate price optimization
- **Responsive Web Interface**: Clean, modern UI with Bootstrap styling and animations
- **Input Validation**: Comprehensive form validation with error handling
- **Real-time Results**: Instant price predictions with profit margin calculations
- **Multiple Product Categories**: Support for Electronics, Clothing, Home & Garden, Sports, and Books
- **Seasonal Adjustments**: Pricing optimization based on seasonal demand patterns

## Project Structure

\`\`\`
price-optimization-tool/
├── app.py                 # Flask web application
├── train_model.py         # Model training script
├── model.pkl             # Trained ML model (generated)
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── templates/
    └── index.html       # Web interface template
\`\`\`

## Setup Instructions

### 1. Clone and Setup Environment

\`\`\`bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate
\`\`\`

### 2. Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Train the Model

\`\`\`bash
python train_model.py
\`\`\`

This will:
- Generate 2000 sample data points with realistic pricing patterns
- Train a Random Forest regression model
- Save the trained model as \`model.pkl\`
- Display model performance metrics

### 4. Run the Application

\`\`\`bash
python app.py
\`\`\`

The application will be available at: **http://localhost:5000**

## Usage

1. **Open the Web Interface**: Navigate to http://localhost:5000
2. **Fill in Product Details**:
   - Select product category (Electronics, Clothing, etc.)
   - Enter cost price
   - Choose season
   - Provide product rating (1-5)
   - Enter inventory count
   - Input competitor price
3. **Get Prediction**: Click "Calculate Optimal Price"
4. **View Results**: See the optimized price along with profit margin analysis

## API Endpoints

### Main Routes
- \`GET /\` - Main web interface
- \`POST /predict\` - Price prediction endpoint
- \`GET /health\` - Health check endpoint

### Prediction API Usage

\`\`\`bash
curl -X POST http://localhost:5000/predict \\
  -F "category=electronics" \\
  -F "cost_price=100" \\
  -F "season=summer" \\
  -F "rating=4.5" \\
  -F "inventory_count=50" \\
  -F "competitor_price=150"
\`\`\`

## Model Details

### Algorithm
- **Random Forest Regressor** with 100 estimators
- Optimized hyperparameters for price prediction accuracy

### Features Used
1. **Cost Price**: Base manufacturing/procurement cost
2. **Category**: Product category (encoded)
3. **Season**: Seasonal demand factor (encoded)
4. **Rating**: Product quality rating (1-5)
5. **Inventory Count**: Available stock quantity
6. **Competitor Price**: Market reference price

### Training Data
- 2000 synthetic data points with realistic pricing patterns
- Category-specific pricing multipliers
- Seasonal demand adjustments
- Rating and inventory influence factors

## Future Enhancements

### Planned Features
- [ ] **Model Retraining Interface**: Web UI for updating the model with new data
- [ ] **Price Comparison Charts**: Visual analytics and trend analysis
- [ ] **Database Integration**: Store predictions in SQLite/PostgreSQL
- [ ] **REST API**: Full RESTful API for programmatic access
- [ ] **User Authentication**: Login system and session tracking
- [ ] **Export Functionality**: PDF/CSV report generation
- [ ] **A/B Testing**: Compare different pricing strategies
- [ ] **Real-time Market Data**: Integration with competitor pricing APIs

### Technical Improvements
- [ ] **Docker Support**: Containerized deployment
- [ ] **Cloud Deployment**: AWS/GCP deployment guides
- [ ] **Model Versioning**: Track and manage different model versions
- [ ] **Performance Monitoring**: Model drift detection and alerts
- [ ] **Advanced ML**: Deep learning models for complex pricing scenarios

## Troubleshooting

### Common Issues

1. **Model Not Found Error**
   \`\`\`bash
   # Solution: Run the training script
   python train_model.py
   \`\`\`

2. **Port Already in Use**
   \`\`\`bash
   # Solution: Use a different port
   python app.py --port 5001
   \`\`\`

3. **Import Errors**
   \`\`\`bash
   # Solution: Ensure all dependencies are installed
   pip install -r requirements.txt
   \`\`\`

### Performance Tips
- The model loads once on startup for optimal performance
- Form validation happens client-side for immediate feedback
- Results are cached for identical requests

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.
\`\`\`

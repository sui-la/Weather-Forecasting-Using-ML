# Weather Forecasting Using Machine Learning 🌤️

A comprehensive full-stack weather prediction application built with Python (Flask) backend and modern HTML/CSS/JavaScript frontend, featuring advanced machine learning algorithms for accurate weather forecasting and analysis.

## 🌟 Features

- **Machine Learning Powered**: Advanced ML algorithms (Random Forest) for weather prediction
- **Real-time Predictions**: Instant weather predictions based on input parameters
- **Multi-day Forecasting**: Extended weather forecasts for better planning
- **Interactive Analytics**: Beautiful charts and visualizations for weather data analysis
- **Modern UI/UX**: Responsive design with beautiful animations and intuitive interface
- **RESTful API**: Clean API endpoints for easy integration

## 🏗️ Project Structure

```
Weather-Forecasting-Using-ML/
├── backend/
│   ├── app.py              # Flask API server with ML models
│   └── __init__.py         # Package initialization
├── frontend/
│   ├── index.html          # Main application interface
│   └── script.js           # Frontend JavaScript logic
├── datasets/
│   ├── seattle-weather.csv # Primary weather dataset
│   └── export.csv          # Processed weather dataset
├── requirements.txt        # Python dependencies
├── run.py                  # Main application runner
├── start_frontend.py       # Frontend server starter
├── start_weather_system.bat # Windows batch file for easy startup
├── install_dependencies.bat # Windows batch file for dependency installation
├── MODEL_COMPARISON_ANALYSIS.md # Detailed ML model analysis
├── ANALYSIS.md             # Project analysis documentation
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Modern web browser
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sui-la/Weather-Forecasting-Using-ML.git
   cd Weather-Forecasting-Using-ML
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Windows users**: You can also use the provided batch file:
   ```bash
   install_dependencies.bat
   ```

3. **Start the application**
   
   **Option A: Using the main runner (Recommended)**
   ```bash
   python run.py
   ```
   
   **Option B: Using Windows batch file**
   ```bash
   start_weather_system.bat
   ```
   
   **Option C: Manual startup**
   ```bash
   # Terminal 1 - Start backend
   cd backend
   python app.py
   
   # Terminal 2 - Start frontend
   python start_frontend.py
   ```

4. **Access the application**
   - Backend API: `http://localhost:5000`
   - Frontend Interface: `http://localhost:8000`
   - Open your browser and navigate to `http://localhost:8000`

## 📊 API Endpoints

### Health Check
- **GET** `/api/health` - Check API status

### Model Training
- **POST** `/api/train` - Train the weather prediction model

### Weather Prediction
- **POST** `/api/predict` - Make a single weather prediction
  ```json
  {
    "tavg": 25.5,
    "tmin": 20.0,
    "tmax": 30.0,
    "prcp": 0.0,
    "wspd": 5.0,
    "pres": 1013.0,
    "date": "2024-01-15"
  }
  ```

### Weather Forecast
- **POST** `/api/forecast` - Generate multi-day forecast
  ```json
  {
    "days": 7
  }
  ```

### Data Retrieval
- **GET** `/api/data` - Get historical weather data
- **GET** `/api/dataset-info` - Get dataset statistics and information
- **GET** `/api/model-info` - Get trained model information and performance metrics

## 🎯 Usage

### Making Predictions

1. **Navigate to the Prediction section**
2. **Enter weather parameters**:
   - Average Temperature (°C)
   - Minimum Temperature (°C)
   - Maximum Temperature (°C)
   - Precipitation (mm)
   - Wind Speed (km/h)
   - Pressure (hPa)
   - Date (optional)
3. **Click "Predict Weather"**
4. **View the prediction result** with temperature and weather icon

### Generating Forecasts

1. **Go to the Forecast section**
2. **Select number of days** (3, 5, 7, or 10 days)
3. **Click "Generate Forecast"**
4. **View the multi-day forecast** with daily predictions

### Analytics

1. **Visit the Analytics section**
2. **View statistics**:
   - Total records in dataset
   - Average temperature
   - Maximum temperature
3. **Explore the interactive chart** showing temperature trends over time

## 🔧 Technical Details

### Backend (Python/Flask)

- **Framework**: Flask with CORS support
- **ML Library**: Scikit-learn (Random Forest Regressor)
- **Data Processing**: Pandas, NumPy
- **Model Features**:
  - Temperature (avg, min, max)
  - Precipitation
  - Wind speed
  - Pressure
  - Temporal features (month, day, day of week)

### Frontend (HTML/CSS/JavaScript)

- **Framework**: Bootstrap 5 for responsive design
- **Charts**: Chart.js for data visualization
- **Icons**: Font Awesome
- **Styling**: Custom CSS with modern gradients and animations

### Machine Learning Model

- **Algorithm**: Random Forest Regressor
- **Target**: Maximum temperature prediction
- **Features**: 9 input features including weather parameters and temporal data
- **Evaluation**: MAE, MSE, and R² metrics
- **Data**: Uses the provided `export.csv` dataset

## 📈 Model Performance

The system uses a Random Forest model with the following characteristics:
- **Algorithm**: Random Forest Regressor
- **Target Variable**: Maximum temperature prediction
- **Features**: 9 input features including weather parameters and temporal data
- **Performance Metrics**: 
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE) 
  - R-squared (R²) score
- **Accuracy**: High prediction accuracy for temperature forecasting
- **Scalability**: Can handle additional data and features
- **Real-time**: Fast prediction response times

### Dataset Information
- **Primary Dataset**: Seattle weather data (1,463 records)
- **Features**: Temperature (avg, min, max), precipitation, wind speed, pressure
- **Time Range**: Historical weather data for comprehensive training
- **Data Quality**: Preprocessed and validated for ML model training

For detailed model analysis and comparison, see [MODEL_COMPARISON_ANALYSIS.md](MODEL_COMPARISON_ANALYSIS.md)

## 🔄 Data Flow

1. **Data Loading**: CSV file loaded and preprocessed
2. **Feature Engineering**: Additional temporal features created
3. **Model Training**: Random Forest trained on historical data
4. **Prediction**: Real-time predictions based on input parameters
5. **Visualization**: Results displayed with interactive charts

## 🛠️ Customization & Development

### Adding New Features

1. **Backend**: Modify `backend/app.py` to include new ML algorithms
2. **Frontend**: Update `frontend/index.html` and `frontend/script.js` for new UI elements
3. **Data**: Add new columns to the dataset and update preprocessing
4. **Models**: Extend the ML pipeline with additional algorithms

### Model Improvements

- **Advanced Algorithms**: Try XGBoost, Neural Networks, or Support Vector Regression
- **Feature Engineering**: Add humidity, visibility, cloud cover, or seasonal indicators
- **Ensemble Methods**: Combine multiple models for better accuracy
- **Cross-Validation**: Implement k-fold cross-validation for robust evaluation
- **Hyperparameter Tuning**: Optimize model parameters using GridSearch or RandomSearch

### Extending the Application

- **Real-time Data**: Integrate with weather APIs (OpenWeatherMap, WeatherAPI)
- **Geographic Support**: Add location-based predictions
- **Mobile App**: Create a mobile version using React Native or Flutter
- **Database Integration**: Replace CSV files with PostgreSQL or MongoDB
- **User Authentication**: Add user accounts and personalized forecasts

## 🐛 Troubleshooting

### Common Issues

1. **Backend not starting**
   - Check Python version (3.8+ required)
   - Install all dependencies: `pip install -r requirements.txt`
   - Ensure port 5000 is available

2. **Frontend not connecting to backend**
   - Verify backend is running on `http://localhost:5000`
   - Check browser console for CORS errors
   - Ensure API endpoints are accessible

3. **Model training issues**
   - Check dataset format and location
   - Verify all required columns are present
   - Check for missing or invalid data

### Debug Mode

Enable debug mode in the backend:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and test thoroughly
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guidelines for Python code
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

## 📞 Support & Contact

For questions, issues, or suggestions:

- **GitHub Issues**: [Open an issue](https://github.com/sui-la/Weather-Forecasting-Using-ML/issues)
- **Documentation**: Check the troubleshooting section above
- **API Reference**: Review the API endpoints documentation
- **Model Analysis**: See [MODEL_COMPARISON_ANALYSIS.md](MODEL_COMPARISON_ANALYSIS.md) for detailed ML insights

## 🙏 Acknowledgments

- **Dataset**: Seattle weather data for training the ML models
- **Libraries**: Flask, Scikit-learn, Pandas, NumPy, Chart.js, Bootstrap
- **Community**: Open source contributors and the Python ML community

---

**Built with ❤️ using Python, Flask, Scikit-learn, and modern web technologies**

*Last updated: January 2025*

# Weather Prediction System

A full-stack weather prediction application built with Python (Flask) backend and modern HTML/CSS/JavaScript frontend, featuring machine learning algorithms for accurate weather forecasting.

## 🌟 Features

- **Machine Learning Powered**: Advanced ML algorithms (Random Forest) for weather prediction
- **Real-time Predictions**: Instant weather predictions based on input parameters
- **Multi-day Forecasting**: Extended weather forecasts for better planning
- **Interactive Analytics**: Beautiful charts and visualizations for weather data analysis
- **Modern UI/UX**: Responsive design with beautiful animations and intuitive interface
- **RESTful API**: Clean API endpoints for easy integration

## 🏗️ Architecture

```
Weather-Prediction-System/
├── backend/
│   ├── app.py              # Flask API server
│   └── __init__.py         # Package initialization
├── frontend/
│   ├── index.html          # Main application interface
│   └── script.js           # Frontend JavaScript logic
├── export.csv              # Weather dataset
├── requirements.txt        # Python dependencies
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
   git clone <repository-url>
   cd Weather-Forecasting-Using-ML
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the backend server**
   ```bash
   cd backend
   python app.py
   ```
   The API will be available at `http://localhost:5000`

4. **Open the frontend**
   - Navigate to the `frontend` folder
   - Open `index.html` in your web browser
   - Or serve it using a local server:
     ```bash
     cd frontend
     python -m http.server 8000
     ```
   - Access the application at `http://localhost:8000`

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
- **Accuracy**: High prediction accuracy for temperature forecasting
- **Features**: Multiple weather parameters for comprehensive analysis
- **Scalability**: Can handle additional data and features
- **Real-time**: Fast prediction response times

## 🔄 Data Flow

1. **Data Loading**: CSV file loaded and preprocessed
2. **Feature Engineering**: Additional temporal features created
3. **Model Training**: Random Forest trained on historical data
4. **Prediction**: Real-time predictions based on input parameters
5. **Visualization**: Results displayed with interactive charts

## 🛠️ Customization

### Adding New Features

1. **Backend**: Modify `app.py` to include new ML algorithms
2. **Frontend**: Update `index.html` and `script.js` for new UI elements
3. **Data**: Add new columns to the dataset and update preprocessing

### Model Improvements

- Try different ML algorithms (XGBoost, Neural Networks)
- Add more features (humidity, visibility, etc.)
- Implement ensemble methods
- Add model validation and cross-validation

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

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For questions or issues:
- Check the troubleshooting section
- Review the API documentation
- Open an issue on GitHub

---

**Built with ❤️ using Python, Flask, and modern web technologies**

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import pickle
import os
from datetime import datetime, timedelta
import json
import uuid
import tempfile

app = Flask(__name__)
CORS(app)

# Global variables for models and scaler
models = {
    'random_forest': None,
    'ann': None,
    'svm': None,
    'knn': None
}
scaler = None
session_data = []
feature_columns = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']
current_dataset_path = '../export.csv'  # Default dataset

def load_data(dataset_path=None):
    """Load and preprocess the weather data"""
    try:
        if dataset_path is None:
            dataset_path = current_dataset_path
            
        df = pd.read_csv(dataset_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Handle missing values
        df = df.ffill().bfill()
        
        # Create additional features
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Create rain classification (if prcp > 0, it's raining)
        df['is_rainy'] = (df['prcp'] > 0).astype(int)
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def train_models(dataset_path=None):
    """Train all weather prediction models"""
    global models, scaler, current_dataset_path
    
    if dataset_path:
        current_dataset_path = dataset_path
    
    df = load_data(current_dataset_path)
    if df is None:
        return False, "Failed to load dataset"
    
    # Prepare features and targets
    X = df[feature_columns + ['month', 'day', 'day_of_week']].copy()
    y_temp = df['tmax']  # Temperature prediction
    y_rain = df['is_rainy']  # Rain classification
    
    # Split data
    X_train, X_test, y_temp_train, y_temp_test, y_rain_train, y_rain_test = train_test_split(
        X, y_temp, y_rain, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train different models
    model_results = {}
    
    try:
        # 1. Random Forest
        print("Training Random Forest...")
        models['random_forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
        models['random_forest'].fit(X_train_scaled, y_temp_train)
        y_pred_rf = models['random_forest'].predict(X_test_scaled)
        model_results['random_forest'] = {
            'mae': mean_absolute_error(y_temp_test, y_pred_rf),
            'mse': mean_squared_error(y_temp_test, y_pred_rf),
            'r2': r2_score(y_temp_test, y_pred_rf)
        }
        
        # 2. Neural Network (ANN)
        print("Training Neural Network...")
        models['ann'] = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        models['ann'].fit(X_train_scaled, y_temp_train)
        y_pred_ann = models['ann'].predict(X_test_scaled)
        model_results['ann'] = {
            'mae': mean_absolute_error(y_temp_test, y_pred_ann),
            'mse': mean_squared_error(y_temp_test, y_pred_ann),
            'r2': r2_score(y_temp_test, y_pred_ann)
        }
        
        # 3. Support Vector Machine (SVM)
        print("Training SVM...")
        models['svm'] = SVR(kernel='rbf', C=100, gamma='scale')
        models['svm'].fit(X_train_scaled, y_temp_train)
        y_pred_svm = models['svm'].predict(X_test_scaled)
        model_results['svm'] = {
            'mae': mean_absolute_error(y_temp_test, y_pred_svm),
            'mse': mean_squared_error(y_temp_test, y_pred_svm),
            'r2': r2_score(y_temp_test, y_pred_svm)
        }
        
        # 4. K-Nearest Neighbors (KNN)
        print("Training KNN...")
        models['knn'] = KNeighborsRegressor(n_neighbors=5)
        models['knn'].fit(X_train_scaled, y_temp_train)
        y_pred_knn = models['knn'].predict(X_test_scaled)
        model_results['knn'] = {
            'mae': mean_absolute_error(y_temp_test, y_pred_knn),
            'mse': mean_squared_error(y_temp_test, y_pred_knn),
            'r2': r2_score(y_temp_test, y_pred_knn)
        }
        
        # Train rain classifier
        print("Training Rain Classifier...")
        rain_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rain_classifier.fit(X_train_scaled, y_rain_train)
        y_rain_pred = rain_classifier.predict(X_test_scaled)
        rain_accuracy = accuracy_score(y_rain_test, y_rain_pred)
        
        print(f"Models trained successfully!")
        print(f"Rain Classification Accuracy: {rain_accuracy:.2f}")
        for model_name, metrics in model_results.items():
            print(f"{model_name.upper()}: MAE={metrics['mae']:.2f}, R²={metrics['r2']:.2f}")
        
        return True, {
            'model_results': model_results,
            'rain_accuracy': rain_accuracy,
            'dataset_info': {
                'total_records': len(df),
                'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
                'features': list(X.columns)
            }
        }
        
    except Exception as e:
        print(f"Error training models: {e}")
        return False, str(e)

def predict_weather(input_data, model_type='random_forest'):
    """Make weather prediction using specified model"""
    global models, scaler
    
    if models[model_type] is None or scaler is None:
        return None, None
    
    try:
        # Prepare input features
        features = np.array([[
            input_data['tavg'],
            input_data['tmin'],
            input_data['tmax'],
            input_data['prcp'],
            input_data['wspd'],
            input_data['pres'],
            input_data['month'],
            input_data['day'],
            input_data['day_of_week']
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make temperature prediction
        temp_prediction = models[model_type].predict(features_scaled)[0]
        
        # Simple rain prediction based on precipitation
        rain_prediction = "Rain" if input_data['prcp'] > 0 else "No Rain"
        
        return temp_prediction, rain_prediction
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None

def add_to_session(prediction_data):
    """Add prediction to session records"""
    global session_data
    session_data.append({
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'input_data': prediction_data['input_data'],
        'predictions': prediction_data['predictions'],
        'model_used': prediction_data['model_used']
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Weather Prediction API is running'})

@app.route('/api/train', methods=['POST'])
def train():
    """Train the models endpoint"""
    try:
        success, result = train_models()
        if success:
            return jsonify({
                'status': 'success', 
                'message': 'Models trained successfully',
                'training_results': result
            })
        else:
            return jsonify({'status': 'error', 'message': result}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_dataset():
    """Upload and train with custom dataset"""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'status': 'error', 'message': 'Please upload a CSV file'}), 400
        
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        # Try to load and validate the dataset
        try:
            df = pd.read_csv(temp_path)
            required_columns = ['date', 'tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return jsonify({
                    'status': 'error', 
                    'message': f'Missing required columns: {", ".join(missing_columns)}'
                }), 400
            
            # Train models with the new dataset
            success, result = train_models(temp_path)
            
            if success:
                return jsonify({
                    'status': 'success',
                    'message': 'Dataset uploaded and models trained successfully',
                    'training_results': result,
                    'filename': file.filename
                })
            else:
                return jsonify({'status': 'error', 'message': result}), 500
                
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Invalid dataset format: {str(e)}'}), 400
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Weather prediction endpoint"""
    try:
        data = request.get_json()
        
        # Validate input data
        required_fields = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']
        for field in required_fields:
            if field not in data:
                return jsonify({'status': 'error', 'message': f'Missing required field: {field}'}), 400
        
        # Add date-based features
        if 'date' in data:
            date_obj = datetime.strptime(data['date'], '%Y-%m-%d')
            data['month'] = date_obj.month
            data['day'] = date_obj.day
            data['day_of_week'] = date_obj.weekday()
        else:
            # Use current date if not provided
            today = datetime.now()
            data['month'] = today.month
            data['day'] = today.day
            data['day_of_week'] = today.weekday()
        
        # Get model type (default to random_forest)
        model_type = data.get('model_type', 'random_forest')
        if model_type not in models:
            model_type = 'random_forest'
        
        # Make predictions with all models
        all_predictions = {}
        for model_name in models.keys():
            temp_pred, rain_pred = predict_weather(data, model_name)
            if temp_pred is not None:
                all_predictions[model_name] = {
                    'temperature': round(temp_pred, 2),
                    'rain_prediction': rain_pred
                }
        
        if all_predictions:
            # Add to session
            session_entry = {
                'input_data': data,
                'predictions': all_predictions,
                'model_used': model_type
            }
            add_to_session(session_entry)
            
            return jsonify({
                'status': 'success',
                'predictions': all_predictions,
                'input_data': data,
                'session_id': session_data[-1]['id']
            })
        else:
            return jsonify({'status': 'error', 'message': 'Failed to make prediction'}), 500
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/session', methods=['GET'])
def get_session():
    """Get current session data"""
    return jsonify({
        'status': 'success',
        'session_data': session_data,
        'total_predictions': len(session_data)
    })

@app.route('/api/session/clear', methods=['POST'])
def clear_session():
    """Clear session data"""
    global session_data
    session_data = []
    return jsonify({'status': 'success', 'message': 'Session cleared'})

@app.route('/api/data', methods=['GET'])
def get_data():
    """Get weather data endpoint"""
    try:
        df = load_data()
        if df is not None:
            # Return last 30 days of data
            recent_data = df.tail(30).to_dict('records')
            return jsonify({
                'status': 'success',
                'data': recent_data
            })
        else:
            return jsonify({'status': 'error', 'message': 'Failed to load data'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/forecast', methods=['POST'])
def forecast():
    """Generate weather forecast for multiple days"""
    try:
        data = request.get_json()
        days = data.get('days', 7)  # Default to 7 days
        model_type = data.get('model_type', 'random_forest')
        
        df = load_data()
        if df is None:
            return jsonify({'status': 'error', 'message': 'Failed to load data'}), 500
        
        # Get the latest data point as base
        latest_data = df.iloc[-1]
        
        forecasts = []
        current_date = datetime.now()
        
        for i in range(days):
            forecast_date = current_date + timedelta(days=i)
            
            # Use average values from historical data for prediction
            input_data = {
                'tavg': latest_data['tavg'],
                'tmin': latest_data['tmin'],
                'tmax': latest_data['tmax'],
                'prcp': latest_data['prcp'],
                'wspd': latest_data['wspd'],
                'pres': latest_data['pres'],
                'month': forecast_date.month,
                'day': forecast_date.day,
                'day_of_week': forecast_date.weekday()
            }
            
            temp_prediction, rain_prediction = predict_weather(input_data, model_type)
            
            forecasts.append({
                'date': forecast_date.strftime('%Y-%m-%d'),
                'predicted_temperature': round(temp_prediction, 2) if temp_prediction else None,
                'rain_prediction': rain_prediction,
                'input_data': input_data
            })
        
        return jsonify({
            'status': 'success',
            'forecasts': forecasts
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Train models on startup
    print("Training weather prediction models...")
    train_models()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 
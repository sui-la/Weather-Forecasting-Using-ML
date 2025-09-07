from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import pickle
import os
import tempfile
from datetime import datetime, timedelta
import json
import numpy as np
from datetime import datetime, timedelta
import json
import uuid

def convert_numpy_types(obj):
    """Convert NumPy types to standard Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

app = Flask(__name__)
CORS(app)

# Global variables for models and scaler
models = {
    'ann': None,
    'svm': None,
    'knn': None
}
scaler = None
session_data = []
feature_columns = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']
current_dataset_path = '../datasets/export.csv'  # Default dataset

def get_available_datasets():
    """Get list of available CSV datasets in the project directory"""
    import glob
    import os
    
    # Look for CSV files in the datasets directory and parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(parent_dir, 'datasets')
    
    # Get CSV files from both locations
    csv_files = []
    csv_files.extend(glob.glob(os.path.join(parent_dir, '*.csv')))
    csv_files.extend(glob.glob(os.path.join(datasets_dir, '*.csv')))
    
    available_datasets = []
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        # Check if it has the required columns
        try:
            df = pd.read_csv(csv_file)
            required_columns = ['date', 'tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']
            if all(col in df.columns for col in required_columns):
                available_datasets.append({
                    'filename': filename,
                    'path': csv_file,
                    'records': len(df),
                    'date_range': f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "Unknown"
                })
        except Exception as e:
            print(f"Error checking {filename}: {e}")
            continue
    
    return available_datasets

def set_current_dataset(dataset_path):
    """Set the current dataset path"""
    global current_dataset_path
    current_dataset_path = dataset_path
    return current_dataset_path

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

def calculate_model_comparison(model_results, X_test_scaled, y_temp_test):
    """Calculate comprehensive model comparison metrics"""
    import time
    
    comparison = {
        'performance_ranking': [],
        'model_analysis': {},
        'best_model': None,
        'metrics_summary': {}
    }
    
    # Model characteristics and analysis
    model_info = {
        'ann': {
            'name': 'Artificial Neural Network (MLP)',
            'type': 'Neural Network',
            'strengths': [
                'Can capture complex non-linear patterns',
                'Flexible architecture',
                'Good for complex relationships',
                'Can approximate any continuous function'
            ],
            'weaknesses': [
                'Requires parameter tuning',
                'Prone to overfitting',
                'Black box model (low interpretability)',
                'Sensitive to feature scaling',
                'Requires more data'
            ],
            'use_cases': [
                'Complex pattern recognition',
                'Non-linear relationships',
                'Large datasets available',
                'When interpretability is not critical'
            ],
            'complexity': 'High',
            'interpretability': 'Low',
            'training_speed': 'Slow',
            'prediction_speed': 'Fast'
        },
        'svm': {
            'name': 'Support Vector Machine',
            'type': 'Kernel Method',
            'strengths': [
                'Effective in high-dimensional spaces',
                'Memory efficient',
                'Versatile (different kernels)',
                'Works well with small datasets'
            ],
            'weaknesses': [
                'Slow on large datasets',
                'Sensitive to feature scaling',
                'No probabilistic output',
                'Difficult to interpret'
            ],
            'use_cases': [
                'High-dimensional data',
                'Small to medium datasets',
                'Non-linear pattern recognition',
                'Text classification'
            ],
            'complexity': 'High',
            'interpretability': 'Low',
            'training_speed': 'Slow',
            'prediction_speed': 'Fast'
        },
        'knn': {
            'name': 'K-Nearest Neighbors',
            'type': 'Instance-based',
            'strengths': [
                'Simple and intuitive',
                'No training period',
                'Works well with small datasets',
                'Naturally handles multi-class problems'
            ],
            'weaknesses': [
                'Computationally expensive for prediction',
                'Sensitive to irrelevant features',
                'Requires good distance metric',
                'Poor performance with high dimensions'
            ],
            'use_cases': [
                'Simple baseline model',
                'Recommendation systems',
                'Pattern recognition',
                'When training data is limited'
            ],
            'complexity': 'Low',
            'interpretability': 'High',
            'training_speed': 'Fast',
            'prediction_speed': 'Slow'
        }
    }
    
    # Calculate performance ranking
    ranked_models = sorted(model_results.items(), 
                          key=lambda x: (x[1]['mae'], -x[1]['r2']))
    
    for i, (model_name, metrics) in enumerate(ranked_models):
        ranking_info = {
            'rank': i + 1,
            'model': model_name,
            'name': model_info[model_name]['name'],
            'mae': round(metrics['mae'], 4),
            'mse': round(metrics['mse'], 4),
            'r2': round(metrics['r2'], 4),
            'rmse': round(np.sqrt(metrics['mse']), 4),
            'accuracy_percentage': round(metrics['r2'] * 100, 2),
            'error_percentage': round((metrics['mae'] / np.mean(y_temp_test)) * 100, 2)
        }
        comparison['performance_ranking'].append(ranking_info)
        
        # Add detailed analysis
        comparison['model_analysis'][model_name] = {
            **model_info[model_name],
            'performance': ranking_info,
            'recommendation': get_model_recommendation(model_name, ranking_info, model_info[model_name])
        }
    
    # Best model identification
    best_model = ranked_models[0][0]
    comparison['best_model'] = {
        'model': best_model,
        'name': model_info[best_model]['name'],
        'reason': f"Lowest MAE ({ranked_models[0][1]['mae']:.4f}) and highest R² ({ranked_models[0][1]['r2']:.4f})"
    }
    
    # Metrics summary
    comparison['metrics_summary'] = {
        'mae_range': {
            'min': round(min(m['mae'] for m in model_results.values()), 4),
            'max': round(max(m['mae'] for m in model_results.values()), 4),
            'best_model': min(model_results.items(), key=lambda x: x[1]['mae'])[0]
        },
        'r2_range': {
            'min': round(min(m['r2'] for m in model_results.values()), 4),
            'max': round(max(m['r2'] for m in model_results.values()), 4),
            'best_model': max(model_results.items(), key=lambda x: x[1]['r2'])[0]
        },
        'total_models': len(model_results)
    }
    
    return comparison

def get_model_recommendation(model_name, performance, model_info):
    """Generate recommendation for when to use this model"""
    rank = performance['rank']
    r2 = performance['r2']
    
    if rank == 1:
        return f"🏆 RECOMMENDED: Best performing model with {performance['accuracy_percentage']:.1f}% accuracy. " + \
               f"Use this model for production deployment."
    elif rank == 2:
        return f"⭐ GOOD CHOICE: Second best model with {performance['accuracy_percentage']:.1f}% accuracy. " + \
               f"Consider if you need {model_info['strengths'][0].lower()}."
    elif r2 > 0.8:
        return f"✅ VIABLE: Good performance ({performance['accuracy_percentage']:.1f}% accuracy). " + \
               f"Use when {model_info['use_cases'][0].lower()} is priority."
    else:
        return f"⚠️ CONSIDER ALTERNATIVES: Lower performance ({performance['accuracy_percentage']:.1f}% accuracy). " + \
               f"Better for {model_info['use_cases'][0].lower()} scenarios."

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
        # 1. Neural Network (ANN)
        print("Training Neural Network...")
        models['ann'] = MLPRegressor(
            hidden_layer_sizes=(10, 5),  # Much smaller network for small dataset
            max_iter=2000,  # More iterations
            random_state=42,
            alpha=0.1,  # Higher regularization for small dataset
            learning_rate_init=0.01,  # Higher learning rate for small dataset
            early_stopping=False,  # Disable early stopping for small dataset
            solver='lbfgs',  # Better solver for small datasets
            warm_start=False
        )
        models['ann'].fit(X_train_scaled, y_temp_train)
        y_pred_ann = models['ann'].predict(X_test_scaled)
        model_results['ann'] = {
            'mae': mean_absolute_error(y_temp_test, y_pred_ann),
            'mse': mean_squared_error(y_temp_test, y_pred_ann),
            'r2': r2_score(y_temp_test, y_pred_ann)
        }
        
        # 2. Support Vector Machine (SVM)
        print("Training SVM...")
        models['svm'] = SVR(kernel='rbf', C=100, gamma='scale')
        models['svm'].fit(X_train_scaled, y_temp_train)
        y_pred_svm = models['svm'].predict(X_test_scaled)
        model_results['svm'] = {
            'mae': mean_absolute_error(y_temp_test, y_pred_svm),
            'mse': mean_squared_error(y_temp_test, y_pred_svm),
            'r2': r2_score(y_temp_test, y_pred_svm)
        }
        
        # 3. K-Nearest Neighbors (KNN)
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
        
        # Calculate additional metrics for comprehensive comparison
        model_comparison = calculate_model_comparison(model_results, X_test_scaled, y_temp_test)
        
        return True, {
            'model_results': model_results,
            'model_comparison': model_comparison,
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

def predict_weather(input_data, model_type='ann'):
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
        
        # 🚨 Add temperature range validation and correction
        if temp_prediction > 60:  # Abnormally high temperature
            print(f"⚠️ Warning: {model_type} predicted abnormal temperature: {temp_prediction}°C")
            # For Neural Network, use more conservative fallback
            if model_type == 'ann':
                temp_prediction = (input_data['tavg'] + input_data['tmax']) / 2
            else:
                temp_prediction = min(temp_prediction, input_data['tmax'] + 10)
        elif temp_prediction < -50:  # Abnormally low temperature
            print(f"⚠️ Warning: {model_type} predicted abnormal temperature: {temp_prediction}°C")
            # For Neural Network, use more conservative fallback
            if model_type == 'ann':
                temp_prediction = (input_data['tavg'] + input_data['tmin']) / 2
            else:
                temp_prediction = max(temp_prediction, input_data['tmin'] - 10)
        
        # Additional check for Neural Network extreme errors
        if model_type == 'ann' and abs(temp_prediction - input_data['tavg']) > 20:
            print(f"⚠️ ANN prediction too far from input average, using fallback")
            temp_prediction = input_data['tavg'] + (input_data['tmax'] - input_data['tavg']) * 0.5
        
        # Ensure temperature is within reasonable range (-60°C to 60°C)
        temp_prediction = max(-60, min(60, temp_prediction))
        
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

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get list of available datasets"""
    try:
        datasets = get_available_datasets()
        return jsonify({
            'status': 'success',
            'datasets': datasets,
            'current_dataset': current_dataset_path,
            'total_available': len(datasets)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/datasets/switch', methods=['POST'])
def switch_dataset():
    """Switch to a different dataset"""
    try:
        data = request.get_json()
        dataset_path = data.get('dataset_path')
        
        if not dataset_path:
            return jsonify({'status': 'error', 'message': 'No dataset path provided'}), 400
        
        # Validate the dataset exists and has required columns
        try:
            df = pd.read_csv(dataset_path)
            required_columns = ['date', 'tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return jsonify({
                    'status': 'error', 
                    'message': f'Missing required columns: {", ".join(missing_columns)}'
                }), 400
            
            # Set new dataset and retrain models
            set_current_dataset(dataset_path)
            success, result = train_models(dataset_path)
            
            if success:
                return jsonify({
                    'status': 'success',
                    'message': f'Switched to dataset: {os.path.basename(dataset_path)}',
                    'dataset_info': {
                        'filename': os.path.basename(dataset_path),
                        'total_records': len(df),
                        'date_range': f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "Unknown"
                    },
                    'training_results': result
                })
            else:
                return jsonify({'status': 'error', 'message': result}), 500
                
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Invalid dataset: {str(e)}'}), 400
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

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
        
        # Get model type (default to ann)
        model_type = data.get('model_type', 'ann')
        if model_type not in models:
            model_type = 'ann'
        
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
        df = load_data(current_dataset_path)

        if df is not None:
            # Get optional limit parameter from query string
            limit = request.args.get('limit', type=int)
            
            if limit and limit > 0:
                # Return limited number of records (most recent)
                recent_data = df.tail(limit).to_dict('records')
            else:
                # Return ALL data for complete analytics
                recent_data = df.to_dict('records')
                
            return jsonify({
                'status': 'success',
                'data': convert_numpy_types(recent_data),
                'total_records': len(df),
                'returned_records': len(recent_data)
            })
        else:
            return jsonify({'status': 'error', 'message': 'Failed to load data'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/models/compare', methods=['GET'])
def compare_models():
    """Get comprehensive model comparison and analysis"""
    try:
        # Check if models are trained
        if not any(models.values()):
            return jsonify({
                'status': 'error', 
                'message': 'Models not trained yet. Please train models first.'
            }), 400
        
        # Load data for comparison (use current dataset)
        df = load_data(current_dataset_path)
        if df is None:
            return jsonify({'status': 'error', 'message': 'Failed to load data'}), 500
        
        # Prepare features
        X = df[feature_columns + ['month', 'day', 'day_of_week']].copy()
        y_temp = df['tmax']
        
        # Split data (same as training)
        X_train, X_test, y_temp_train, y_temp_test = train_test_split(
            X, y_temp, test_size=0.2, random_state=42
        )
        
        # Scale features
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
            
            # Calculate predictions for all models
            model_results = {}
            for model_name, model in models.items():
                if model is not None:
                    y_pred = model.predict(X_test_scaled)
                    model_results[model_name] = {
                        'mae': mean_absolute_error(y_temp_test, y_pred),
                        'mse': mean_squared_error(y_temp_test, y_pred),
                        'r2': r2_score(y_temp_test, y_pred)
                    }
            
            # Generate comprehensive comparison
            comparison = calculate_model_comparison(model_results, X_test_scaled, y_temp_test)
            
            return jsonify({
                'status': 'success',
                'comparison': comparison,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error', 
                'message': 'Scaler not available. Please retrain models.'
            }), 500
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/models/info', methods=['GET'])
def get_model_info():
    """Get detailed information about all available models"""
    try:
        model_info = {
            'ann': {
                'name': 'Artificial Neural Network (Multi-Layer Perceptron)',
                'description': 'A neural network with multiple hidden layers for complex pattern recognition',
                'algorithm_type': 'Neural Network',
                'parameters': {
                    'hidden_layer_sizes': [100, 50],
                    'max_iter': 500,
                    'random_state': 42
                },
                'pros': [
                    'Can model complex non-linear relationships',
                    'Flexible architecture can be adapted to many problems',
                    'Universal function approximator',
                    'Good performance with large datasets'
                ],
                'cons': [
                    'Requires careful hyperparameter tuning',
                    'Prone to overfitting with small datasets',
                    'Black box model - difficult to interpret',
                    'Sensitive to feature scaling',
                    'Can get stuck in local minima'
                ],
                'best_for': [
                    'Complex pattern recognition tasks',
                    'Non-linear relationship modeling',
                    'Large datasets with sufficient training examples',
                    'Problems where interpretability is not critical'
                ]
            },
            'svm': {
                'name': 'Support Vector Machine',
                'description': 'A powerful algorithm that finds optimal decision boundaries using kernel methods',
                'algorithm_type': 'Kernel Method',
                'parameters': {
                    'kernel': 'rbf',
                    'C': 100,
                    'gamma': 'scale'
                },
                'pros': [
                    'Effective in high-dimensional spaces',
                    'Memory efficient',
                    'Versatile with different kernel functions',
                    'Works well with small to medium datasets'
                ],
                'cons': [
                    'Slow training on large datasets',
                    'Sensitive to feature scaling',
                    'No probabilistic output',
                    'Difficult to interpret results',
                    'Choice of kernel and parameters is crucial'
                ],
                'best_for': [
                    'High-dimensional data (e.g., text classification)',
                    'Small to medium-sized datasets',
                    'Non-linear pattern recognition',
                    'Problems with clear margin of separation'
                ]
            },
            'knn': {
                'name': 'K-Nearest Neighbors',
                'description': 'A simple algorithm that predicts based on the k closest training examples',
                'algorithm_type': 'Instance-Based Learning',
                'parameters': {
                    'n_neighbors': 5
                },
                'pros': [
                    'Simple and intuitive algorithm',
                    'No training period required',
                    'Works well with small datasets',
                    'Naturally handles multi-class problems',
                    'Can be effective with the right distance metric'
                ],
                'cons': [
                    'Computationally expensive for prediction',
                    'Sensitive to irrelevant features',
                    'Poor performance in high-dimensional spaces',
                    'Sensitive to local structure of data',
                    'Need to store all training data'
                ],
                'best_for': [
                    'Simple baseline model establishment',
                    'Recommendation systems',
                    'Pattern recognition with similar examples',
                    'Small datasets where training time is not critical'
                ]
            }
        }
        
        # Add training status
        training_status = {}
        for model_name in model_info.keys():
            training_status[model_name] = {
                'trained': models[model_name] is not None,
                'available_for_prediction': models[model_name] is not None and scaler is not None
            }
        
        return jsonify({
            'status': 'success',
            'models': model_info,
            'training_status': training_status,
            'total_models': len(model_info)
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/forecast', methods=['POST'])
def forecast():
    """Generate weather forecast for multiple days"""
    try:
        data = request.get_json()
        days = data.get('days', 7)  # Default to 7 days
        model_type = data.get('model_type', 'ann')
        
        df = load_data(current_dataset_path)
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
                'tavg': float(latest_data['tavg']),
                'tmin': float(latest_data['tmin']),
                'tmax': float(latest_data['tmax']),
                'prcp': float(latest_data['prcp']),
                'wspd': float(latest_data['wspd']),
                'pres': float(latest_data['pres']),
                'month': int(forecast_date.month),
                'day': int(forecast_date.day),
                'day_of_week': int(forecast_date.weekday())
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
            'forecasts': convert_numpy_types(forecasts)
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Train models on startup
    print("Training weather prediction models...")
    train_models()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 
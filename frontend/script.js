// API Base URL
const API_BASE_URL = 'http://localhost:5000/api';

// Global variables
let weatherChart = null;
let sessionData = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('Weather Prediction System initialized');
    
    // Set default date to today
    document.getElementById('date').value = new Date().toISOString().split('T')[0];
    
    // Load initial data
    loadWeatherData();
    loadSessionData();
    
    // Setup form submission
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    // Prediction form submission
    document.getElementById('predictionForm').addEventListener('submit', function(e) {
        e.preventDefault();
        makePrediction();
    });
    
    // Smooth scrolling for navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Upload and train with custom dataset
async function uploadAndTrain() {
    const fileInput = document.getElementById('datasetFile');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a CSV file to upload');
        return;
    }
    
    if (!file.name.endsWith('.csv')) {
        alert('Please select a CSV file');
        return;
    }
    
    // Show loading
    showLoading('uploadLoading');
    hideResult('uploadResult');
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            displayUploadResult(result);
        } else {
            showError('uploadResult', result.message);
        }
    } catch (error) {
        console.error('Error uploading dataset:', error);
        showError('uploadResult', 'Failed to upload dataset. Please check server connection.');
    } finally {
        hideLoading('uploadLoading');
    }
}

// Display upload and training result
function displayUploadResult(result) {
    const resultDiv = document.getElementById('uploadResult');
    
    let resultHTML = `
        <div class="alert alert-success" role="alert">
            <h5 class="alert-heading">
                <i class="fas fa-check-circle me-2"></i>
                Dataset Uploaded Successfully!
            </h5>
            <p><strong>File:</strong> ${result.filename}</p>
            <p><strong>Message:</strong> ${result.message}</p>
        </div>
        
        <div class="card mt-3">
            <div class="card-header">
                <h6 class="mb-0">Dataset Information</h6>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-4">
                        <p><strong>Total Records:</strong> ${result.training_results.dataset_info.total_records}</p>
                    </div>
                    <div class="col-md-4">
                        <p><strong>Date Range:</strong> ${result.training_results.dataset_info.date_range}</p>
                    </div>
                    <div class="col-md-4">
                        <p><strong>Rain Accuracy:</strong> ${(result.training_results.rain_accuracy * 100).toFixed(1)}%</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card mt-3">
            <div class="card-header">
                <h6 class="mb-0">Model Performance</h6>
            </div>
            <div class="card-body">
    `;
    
    // Display model performance
    Object.entries(result.training_results.model_results).forEach(([modelName, metrics]) => {
        resultHTML += `
            <div class="model-card">
                <div class="row align-items-center">
                    <div class="col-md-3">
                        <strong>${getModelDisplayName(modelName)}</strong>
                    </div>
                    <div class="col-md-3">
                        <span class="badge bg-info">MAE: ${metrics.mae.toFixed(2)}</span>
                    </div>
                    <div class="col-md-3">
                        <span class="badge bg-success">R²: ${metrics.r2.toFixed(2)}</span>
                    </div>
                    <div class="col-md-3">
                        <span class="badge bg-warning">MSE: ${metrics.mse.toFixed(2)}</span>
                    </div>
                </div>
            </div>
        `;
    });
    
    resultHTML += `
            </div>
        </div>
    `;
    
    resultDiv.innerHTML = resultHTML;
}

// Make weather prediction
async function makePrediction() {
    const formData = {
        tavg: parseFloat(document.getElementById('tavg').value),
        tmin: parseFloat(document.getElementById('tmin').value),
        tmax: parseFloat(document.getElementById('tmax').value),
        prcp: parseFloat(document.getElementById('prcp').value),
        wspd: parseFloat(document.getElementById('wspd').value),
        pres: parseFloat(document.getElementById('pres').value),
        model_type: document.querySelector('input[name="modelType"]:checked').value
    };
    
    const dateValue = document.getElementById('date').value;
    if (dateValue) {
        formData.date = dateValue;
    }
    
    // Show loading
    showLoading('predictionLoading');
    hideResult('predictionResult');
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            displayPredictionResult(result);
            loadSessionData(); // Refresh session data
        } else {
            showError('predictionResult', result.message);
        }
    } catch (error) {
        console.error('Error making prediction:', error);
        showError('predictionResult', 'Failed to connect to the server. Please check if the backend is running.');
    } finally {
        hideLoading('predictionLoading');
    }
}

// Display prediction result
function displayPredictionResult(result) {
    const resultDiv = document.getElementById('predictionResult');
    const selectedModel = document.querySelector('input[name="modelType"]:checked').value;
    
    let resultHTML = `
        <div class="prediction-result">
            <div class="weather-icon mb-3">
                ${getWeatherIcon(result.predictions[selectedModel].temperature)}
            </div>
            <h3 class="mb-2">Weather Prediction</h3>
            <div class="display-4 fw-bold ${getTemperatureClass(result.predictions[selectedModel].temperature)} mb-3">
                ${result.predictions[selectedModel].temperature}°C
            </div>
            <p class="mb-0">Predicted maximum temperature for the given conditions</p>
        </div>
        
        <div class="rain-prediction">
            <h5 class="mb-2">
                <i class="fas fa-cloud-rain me-2"></i>
                Rain Prediction
            </h5>
            <div class="h4 mb-0">
                ${result.predictions[selectedModel].rain_prediction}
            </div>
        </div>
        
        <div class="model-comparison">
            <h5 class="mb-3">All Model Predictions:</h5>
    `;
    
    // Display all model predictions
    Object.entries(result.predictions).forEach(([modelName, prediction]) => {
        const isSelected = modelName === selectedModel;
        resultHTML += `
            <div class="model-card ${isSelected ? 'border-success' : ''}">
                <div class="row align-items-center">
                    <div class="col-md-3">
                        <strong>${getModelDisplayName(modelName)}</strong>
                        ${isSelected ? '<span class="badge bg-success ms-2">Selected</span>' : ''}
                    </div>
                    <div class="col-md-3">
                        <span class="h5 ${getTemperatureClass(prediction.temperature)}">${prediction.temperature}°C</span>
                    </div>
                    <div class="col-md-3">
                        <span class="badge ${prediction.rain_prediction === 'Rain' ? 'bg-primary' : 'bg-warning'}">${prediction.rain_prediction}</span>
                    </div>
                    <div class="col-md-3">
                        <small class="text-muted">${getModelAccuracy(modelName)}</small>
                    </div>
                </div>
            </div>
        `;
    });
    
    resultHTML += `
        </div>
        
        <div class="card mt-3">
            <div class="card-header">
                <h6 class="mb-0">Input Parameters</h6>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <p><strong>Average Temp:</strong> ${result.input_data.tavg}°C</p>
                        <p><strong>Min Temp:</strong> ${result.input_data.tmin}°C</p>
                        <p><strong>Max Temp:</strong> ${result.input_data.tmax}°C</p>
                    </div>
                    <div class="col-md-6">
                        <p><strong>Precipitation:</strong> ${result.input_data.prcp} mm</p>
                        <p><strong>Wind Speed:</strong> ${result.input_data.wspd} km/h</p>
                        <p><strong>Pressure:</strong> ${result.input_data.pres} hPa</p>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    resultDiv.innerHTML = resultHTML;
}

// Generate weather forecast
async function generateForecast() {
    const days = parseInt(document.getElementById('forecastDays').value);
    const model = document.getElementById('forecastModel').value;
    
    // Show loading
    showLoading('forecastLoading');
    hideResult('forecastResult');
    
    try {
        const response = await fetch(`${API_BASE_URL}/forecast`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                days: days,
                model_type: model
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            displayForecastResult(result.forecasts, model);
        } else {
            showError('forecastResult', result.message);
        }
    } catch (error) {
        console.error('Error generating forecast:', error);
        showError('forecastResult', 'Failed to connect to the server. Please check if the backend is running.');
    } finally {
        hideLoading('forecastLoading');
    }
}

// Display forecast result
function displayForecastResult(forecasts, modelType) {
    const resultDiv = document.getElementById('forecastResult');
    
    let forecastHTML = `
        <h5 class="mb-3">Weather Forecast (${getModelDisplayName(modelType)})</h5>
    `;
    
    forecasts.forEach((forecast, index) => {
        const weatherIcon = getWeatherIcon(forecast.predicted_temperature);
        const temperatureClass = getTemperatureClass(forecast.predicted_temperature);
        
        forecastHTML += `
            <div class="forecast-item">
                <div class="row align-items-center">
                    <div class="col-md-2">
                        <div class="weather-icon">
                            ${weatherIcon}
                        </div>
                    </div>
                    <div class="col-md-4">
                        <h6 class="mb-1">${formatDate(forecast.date)}</h6>
                        <p class="mb-0 text-muted">Day ${index + 1} of forecast</p>
                    </div>
                    <div class="col-md-3 text-center">
                        <div class="h4 ${temperatureClass} mb-0">
                            ${forecast.predicted_temperature}°C
                        </div>
                    </div>
                    <div class="col-md-3 text-end">
                        <span class="badge ${forecast.rain_prediction === 'Rain' ? 'bg-primary' : 'bg-warning'} fs-6">
                            ${forecast.rain_prediction}
                        </span>
                    </div>
                </div>
            </div>
        `;
    });
    
    resultDiv.innerHTML = forecastHTML;
}

// Load session data
async function loadSessionData() {
    try {
        const response = await fetch(`${API_BASE_URL}/session`);
        const result = await response.json();
        
        if (result.status === 'success') {
            sessionData = result.session_data;
            updateSessionDisplay();
        }
    } catch (error) {
        console.error('Error loading session data:', error);
    }
}

// Update session display
function updateSessionDisplay() {
    const sessionCount = document.getElementById('sessionCount');
    const sessionRecords = document.getElementById('sessionRecords');
    
    sessionCount.textContent = sessionData.length;
    
    if (sessionData.length === 0) {
        sessionRecords.innerHTML = '<p class="text-muted text-center">No predictions in current session</p>';
        return;
    }
    
    let sessionHTML = '';
    sessionData.slice().reverse().forEach((record, index) => {
        const selectedModel = record.model_used;
        const prediction = record.predictions[selectedModel];
        
        sessionHTML += `
            <div class="session-item">
                <div class="row align-items-center">
                    <div class="col-md-2">
                        <small class="text-muted">${formatDateTime(record.timestamp)}</small>
                    </div>
                    <div class="col-md-2">
                        <strong>${getModelDisplayName(selectedModel)}</strong>
                    </div>
                    <div class="col-md-2">
                        <span class="h5 ${getTemperatureClass(prediction.temperature)}">${prediction.temperature}°C</span>
                    </div>
                    <div class="col-md-2">
                        <span class="badge ${prediction.rain_prediction === 'Rain' ? 'bg-primary' : 'bg-warning'}">${prediction.rain_prediction}</span>
                    </div>
                    <div class="col-md-4">
                        <small class="text-muted">
                            Temp: ${record.input_data.tavg}°C | Rain: ${record.input_data.prcp}mm | Wind: ${record.input_data.wspd}km/h
                        </small>
                    </div>
                </div>
            </div>
        `;
    });
    
    sessionRecords.innerHTML = sessionHTML;
}

// Clear session
async function clearSession() {
    if (confirm('Are you sure you want to clear all session data?')) {
        try {
            const response = await fetch(`${API_BASE_URL}/session/clear`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                loadSessionData();
            } else {
                alert('Failed to clear session: ' + result.message);
            }
        } catch (error) {
            console.error('Error clearing session:', error);
            alert('Failed to clear session. Please check server connection.');
        }
    }
}

// Load weather data for analytics
async function loadWeatherData() {
    try {
        const response = await fetch(`${API_BASE_URL}/data`);
        const result = await response.json();
        
        if (result.status === 'success') {
            updateAnalytics(result.data);
            createWeatherChart(result.data);
        } else {
            console.error('Failed to load weather data:', result.message);
        }
    } catch (error) {
        console.error('Error loading weather data:', error);
    }
}

// Update analytics statistics
function updateAnalytics(data) {
    if (!data || data.length === 0) return;
    
    const temperatures = data.map(item => item.tmax).filter(temp => temp !== null);
    const avgTemp = temperatures.reduce((a, b) => a + b, 0) / temperatures.length;
    const maxTemp = Math.max(...temperatures);
    const rainyDays = data.filter(item => item.prcp > 0).length;
    
    document.getElementById('totalRecords').textContent = data.length;
    document.getElementById('avgTemp').textContent = `${avgTemp.toFixed(1)}°C`;
    document.getElementById('maxTemp').textContent = `${maxTemp.toFixed(1)}°C`;
    document.getElementById('rainyDays').textContent = rainyDays;
}

// Create weather chart
function createWeatherChart(data) {
    if (!data || data.length === 0) return;
    
    const ctx = document.getElementById('weatherChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (weatherChart) {
        weatherChart.destroy();
    }
    
    const labels = data.map(item => formatDate(item.date));
    const maxTemps = data.map(item => item.tmax);
    const minTemps = data.map(item => item.tmin);
    const avgTemps = data.map(item => item.tavg);
    const precipitation = data.map(item => item.prcp);
    
    weatherChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Max Temperature (°C)',
                    data: maxTemps,
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Average Temperature (°C)',
                    data: avgTemps,
                    borderColor: '#f39c12',
                    backgroundColor: 'rgba(243, 156, 18, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Min Temperature (°C)',
                    data: minTemps,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Precipitation (mm)',
                    data: precipitation,
                    borderColor: '#9b59b6',
                    backgroundColor: 'rgba(155, 89, 182, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1',
                    type: 'bar'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Weather Trends Over Time'
                },
                legend: {
                    position: 'top',
                }
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Temperature (°C)'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Precipitation (mm)'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                }
            }
        }
    });
}

// Utility functions
function showLoading(elementId) {
    document.getElementById(elementId).style.display = 'block';
}

function hideLoading(elementId) {
    document.getElementById(elementId).style.display = 'none';
}

function hideResult(elementId) {
    document.getElementById(elementId).innerHTML = '';
}

function showError(elementId, message) {
    document.getElementById(elementId).innerHTML = `
        <div class="alert alert-danger" role="alert">
            <i class="fas fa-exclamation-triangle me-2"></i>
            ${message}
        </div>
    `;
}

function getWeatherIcon(temperature) {
    if (temperature >= 30) {
        return '<i class="fas fa-sun"></i>';
    } else if (temperature >= 20) {
        return '<i class="fas fa-cloud-sun"></i>';
    } else if (temperature >= 10) {
        return '<i class="fas fa-cloud"></i>';
    } else {
        return '<i class="fas fa-snowflake"></i>';
    }
}

function getTemperatureClass(temperature) {
    if (temperature >= 30) {
        return 'text-danger';
    } else if (temperature >= 20) {
        return 'text-warning';
    } else if (temperature >= 10) {
        return 'text-info';
    } else {
        return 'text-primary';
    }
}

function getModelDisplayName(modelName) {
    const modelNames = {
        'random_forest': 'Random Forest',
        'ann': 'Neural Network',
        'svm': 'SVM',
        'knn': 'KNN'
    };
    return modelNames[modelName] || modelName;
}

function getModelAccuracy(modelName) {
    const accuracies = {
        'random_forest': 'R²: 0.71',
        'ann': 'R²: 0.68',
        'svm': 'R²: 0.65',
        'knn': 'R²: 0.62'
    };
    return accuracies[modelName] || 'R²: 0.60';
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric' 
    });
}

function formatDateTime(dateTimeString) {
    const date = new Date(dateTimeString);
    return date.toLocaleString('en-US', { 
        month: 'short', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Train model function (can be called from console for testing)
async function trainModel() {
    try {
        const response = await fetch(`${API_BASE_URL}/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const result = await response.json();
        console.log('Training result:', result);
        
        if (result.status === 'success') {
            alert('Models trained successfully!');
        } else {
            alert('Failed to train models: ' + result.message);
        }
    } catch (error) {
        console.error('Error training models:', error);
        alert('Failed to train models. Please check server connection.');
    }
}

// Show model information modal
function showModelInfo() {
    const modal = new bootstrap.Modal(document.getElementById('modelInfoModal'));
    modal.show();
}

// Download weather data functions
function showCustomDownload() {
    const section = document.getElementById('customDownloadSection');
    section.style.display = section.style.display === 'none' ? 'block' : 'none';
}

function downloadFromMeteostat() {
    const url = 'https://meteostat.net/en/place/my/kuala-lumpur?s=48647&t=2025-07-01/2025-07-31';
    window.open(url, '_blank');
}

function downloadFromUrl(customUrl = null) {
    const url = customUrl || document.getElementById('weatherUrl').value;
    
    if (!url) {
        alert('Please enter a valid URL');
        return;
    }
    
    window.open(url, '_blank');
} 
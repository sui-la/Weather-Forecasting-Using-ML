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
    
    // Initialize analytics with placeholder
    showAnalyticsPlaceholder();
    showModelComparisonPlaceholder();
    
    // Load session data (but not weather data initially)
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
    
    // Dataset dropdown click event - load datasets only when clicked
    document.getElementById('datasetSelect').addEventListener('click', function() {
        // Check if datasets are already loaded
        if (this.children.length === 1 && this.children[0].value === '') {
            loadAvailableDatasets();
        }
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

// Analytics placeholder management
function showAnalyticsPlaceholder() {
    document.getElementById('analyticsPlaceholder').style.display = 'block';
    document.getElementById('analyticsContent').style.display = 'none';
}

function showAnalyticsContent() {
    document.getElementById('analyticsPlaceholder').style.display = 'none';
    document.getElementById('analyticsContent').style.display = 'block';
}

// Model comparison placeholder management
function showModelComparisonPlaceholder() {
    document.getElementById('modelComparisonPlaceholder').style.display = 'block';
    document.getElementById('modelComparisonContent').style.display = 'none';
}

function showModelComparisonContent() {
    document.getElementById('modelComparisonPlaceholder').style.display = 'none';
    document.getElementById('modelComparisonContent').style.display = 'block';
}

// Load weather data for analytics
async function loadWeatherData() {
    try {
        console.log('Loading weather data...');
        const response = await fetch(`${API_BASE_URL}/data`);
        const result = await response.json();
        
        console.log('Weather data response:', result);
        
        if (result.status === 'success') {
            console.log('Data received:', result.data.length, 'records');
            showAnalyticsContent(); // Show analytics content when data is loaded
            showModelComparisonContent(); // Show model comparison content when data is loaded
            updateAnalytics(result.data);
            createWeatherChart(result.data);
        } else {
            console.error('Failed to load weather data:', result.message);
            showAnalyticsPlaceholder(); // Show placeholder if data loading fails
            showModelComparisonPlaceholder(); // Show placeholder if data loading fails
        }
    } catch (error) {
        console.error('Error loading weather data:', error);
    }
}

// Update analytics statistics
function updateAnalytics(data) {
    console.log('updateAnalytics called with:', data);
    
    if (!data || data.length === 0) {
        console.log('No data to update analytics');
        return;
    }
    
    const temperatures = data.map(item => item.tmax).filter(temp => temp !== null);
    const avgTemp = temperatures.reduce((a, b) => a + b, 0) / temperatures.length;
    const maxTemp = Math.max(...temperatures);
    const rainyDays = data.filter(item => item.prcp > 0).length;
    
    console.log('Analytics calculated:', {
        totalRecords: data.length,
        avgTemp: avgTemp.toFixed(1),
        maxTemp: maxTemp.toFixed(1),
        rainyDays: rainyDays
    });
    
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
        'ann': 'Neural Network',
        'svm': 'SVM',
        'knn': 'KNN'
    };
    return modelNames[modelName] || modelName;
}

function getModelAccuracy(modelName) {
    const accuracies = {
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

// Model Comparison Functions
let accuracyChart = null;
let errorChart = null;

async function loadModelComparison() {
    showLoading('modelComparisonLoading');
    hideElement('modelComparisonResults');
    hideElement('modelInfoResults');
    
    try {
        const response = await fetch(`${API_BASE_URL}/models/compare`);
        const data = await response.json();
        
        if (data.status === 'success') {
            displayModelComparison(data.comparison);
            showElement('modelComparisonResults');
        } else {
            showError('modelComparisonResults', 'Model Comparison Error: ' + data.message);
        }
    } catch (error) {
        console.error('Error loading model comparison:', error);
        showError('modelComparisonResults', 'Failed to load model comparison. Please check if models are trained.');
    } finally {
        hideLoading('modelComparisonLoading');
    }
}

async function loadModelInfo() {
    showLoading('modelComparisonLoading');
    hideElement('modelComparisonResults');
    hideElement('modelInfoResults');
    
    try {
        const response = await fetch(`${API_BASE_URL}/models/info`);
        const data = await response.json();
        
        if (data.status === 'success') {
            displayModelInfo(data.models, data.training_status);
            showElement('modelInfoResults');
        } else {
            showError('modelInfoResults', 'Model Info Error: ' + data.message);
        }
    } catch (error) {
        console.error('Error loading model info:', error);
        showError('modelInfoResults', 'Failed to load model information.');
    } finally {
        hideLoading('modelComparisonLoading');
    }
}

function displayModelComparison(comparison) {
    // Display best model recommendation
    const bestModelInfo = document.getElementById('bestModelInfo');
    bestModelInfo.innerHTML = `
        <strong>${comparison.best_model.name}</strong> is the recommended model.<br>
        <small>${comparison.best_model.reason}</small>
    `;
    
    // Display performance rankings
    const rankingsContainer = document.getElementById('modelRankings');
    rankingsContainer.innerHTML = '';
    
    comparison.performance_ranking.forEach(model => {
        const rankCard = document.createElement('div');
        rankCard.className = 'model-card mb-3';
        
        const rankIcon = getRankIcon(model.rank);
        const performanceColor = getPerformanceColor(model.accuracy_percentage);
        
        rankCard.innerHTML = `
            <div class="row align-items-center">
                <div class="col-md-1">
                    <h4 class="text-center">${rankIcon} ${model.rank}</h4>
                </div>
                <div class="col-md-3">
                    <h6>${model.name}</h6>
                    <small class="text-muted">${model.model}</small>
                </div>
                <div class="col-md-2">
                    <div style="color: ${performanceColor};">
                        <strong>${model.accuracy_percentage}%</strong>
                        <br><small>Accuracy (R²)</small>
                    </div>
                </div>
                <div class="col-md-2">
                    <strong>${model.mae}</strong>
                    <br><small>MAE</small>
                </div>
                <div class="col-md-2">
                    <strong>${model.rmse}</strong>
                    <br><small>RMSE</small>
                </div>
                <div class="col-md-2">
                    <span class="badge" style="background-color: ${performanceColor}; color: white;">
                        ${model.error_percentage.toFixed(1)}% Error
                    </span>
                </div>
            </div>
        `;
        
        rankingsContainer.appendChild(rankCard);
    });
    
    // Create charts
    createAccuracyChart(comparison.performance_ranking);
    createErrorChart(comparison.performance_ranking);
    
    // Display detailed analysis
    displayDetailedAnalysis(comparison.model_analysis);
}

function displayModelInfo(models, trainingStatus) {
    const container = document.getElementById('modelInfoCards');
    container.innerHTML = '';
    
    Object.keys(models).forEach(modelKey => {
        const model = models[modelKey];
        const status = trainingStatus[modelKey];
        
        const card = document.createElement('div');
        card.className = 'col-md-6 mb-4';
        
        const statusBadge = status.trained ? 
            '<span class="badge bg-success">Trained</span>' : 
            '<span class="badge bg-warning">Not Trained</span>';
        
        card.innerHTML = `
            <div class="card h-100">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h5><i class="fas fa-cogs model-icon"></i>${model.name}</h5>
                    ${statusBadge}
                </div>
                <div class="card-body">
                    <p class="card-text">${model.description}</p>
                    
                    <h6 class="text-success"><i class="fas fa-check-circle"></i> Strengths:</h6>
                    <ul class="list-unstyled">
                        ${model.pros.slice(0, 3).map(pro => `<li><i class="fas fa-plus text-success me-2"></i>${pro}</li>`).join('')}
                    </ul>
                    
                    <h6 class="text-warning"><i class="fas fa-exclamation-triangle"></i> Limitations:</h6>
                    <ul class="list-unstyled">
                        ${model.cons.slice(0, 3).map(con => `<li><i class="fas fa-minus text-warning me-2"></i>${con}</li>`).join('')}
                    </ul>
                    
                    <h6 class="text-info"><i class="fas fa-lightbulb"></i> Best For:</h6>
                    <ul class="list-unstyled">
                        ${model.best_for.slice(0, 2).map(use => `<li><i class="fas fa-arrow-right text-info me-2"></i>${use}</li>`).join('')}
                    </ul>
                    
                    <div class="mt-3">
                        <small class="text-muted">
                            <strong>Algorithm Type:</strong> ${model.algorithm_type}<br>
                            <strong>Parameters:</strong> ${JSON.stringify(model.parameters)}
                        </small>
                    </div>
                </div>
            </div>
        `;
        
        container.appendChild(card);
    });
    
    // Wrap cards in row
    container.className = 'row';
}

function displayDetailedAnalysis(modelAnalysis) {
    const container = document.getElementById('modelAnalysis');
    container.innerHTML = '';
    
    Object.keys(modelAnalysis).forEach(modelKey => {
        const analysis = modelAnalysis[modelKey];
        
        const analysisCard = document.createElement('div');
        analysisCard.className = 'model-explanation mb-4';
        
        analysisCard.innerHTML = `
            <h5>${analysis.name} Analysis</h5>
            
            <div class="row mb-3">
                <div class="col-md-3">
                    <strong>Rank:</strong> #${analysis.performance.rank}
                </div>
                <div class="col-md-3">
                    <strong>Accuracy:</strong> ${analysis.performance.accuracy_percentage}%
                </div>
                <div class="col-md-3">
                    <strong>Complexity:</strong> ${analysis.complexity}
                </div>
                <div class="col-md-3">
                    <strong>Interpretability:</strong> ${analysis.interpretability}
                </div>
            </div>
            
            <div class="alert alert-info">
                <strong>Recommendation:</strong> ${analysis.recommendation}
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <h6 class="text-success">Strengths:</h6>
                    <ul>
                        ${analysis.strengths.map(strength => `<li>${strength}</li>`).join('')}
                    </ul>
                </div>
                <div class="col-md-6">
                    <h6 class="text-warning">Weaknesses:</h6>
                    <ul>
                        ${analysis.weaknesses.map(weakness => `<li>${weakness}</li>`).join('')}
                    </ul>
                </div>
            </div>
            
            <div class="mt-3">
                <h6 class="text-info">Best Use Cases:</h6>
                <ul>
                    ${analysis.use_cases.map(useCase => `<li>${useCase}</li>`).join('')}
                </ul>
            </div>
        `;
        
        container.appendChild(analysisCard);
    });
}

function createAccuracyChart(performanceData) {
    const ctx = document.getElementById('accuracyChart').getContext('2d');
    
    if (accuracyChart) {
        accuracyChart.destroy();
    }
    
    const colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'];
    
    accuracyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: performanceData.map(model => model.name),
            datasets: [{
                label: 'Accuracy (R²)',
                data: performanceData.map(model => model.accuracy_percentage),
                backgroundColor: colors,
                borderColor: colors.map(color => color + '80'),
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

function createErrorChart(performanceData) {
    const ctx = document.getElementById('errorChart').getContext('2d');
    
    if (errorChart) {
        errorChart.destroy();
    }
    
    const colors = ['#FF5722', '#E91E63', '#9C27B0', '#673AB7'];
    
    errorChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: performanceData.map(model => model.name),
            datasets: [{
                label: 'Mean Absolute Error',
                data: performanceData.map(model => model.mae),
                backgroundColor: colors[0] + '20',
                borderColor: colors[0],
                borderWidth: 3,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Helper functions
function getRankIcon(rank) {
    switch(rank) {
        case 1: return '🏆';
        case 2: return '🥈';
        case 3: return '🥉';
        default: return '📊';
    }
}

function getPerformanceColor(accuracy) {
    if (accuracy >= 90) return '#4CAF50';
    if (accuracy >= 80) return '#2196F3';
    if (accuracy >= 70) return '#FF9800';
    return '#F44336';
}

function showElement(elementId) {
    document.getElementById(elementId).style.display = 'block';
}

function hideElement(elementId) {
    document.getElementById(elementId).style.display = 'none';
}

function showLoading(elementId) {
    document.getElementById(elementId).style.display = 'block';
}

function hideLoading(elementId) {
    document.getElementById(elementId).style.display = 'none';
}

// Dataset Management Functions
async function refreshAnalytics() {
    try {
        console.log('Refreshing analytics...');
        
        // Show placeholder while loading
        showAnalyticsPlaceholder();
        showModelComparisonPlaceholder();
        
        // Load fresh data
        await loadWeatherData();
        
        // Scroll to analytics section to show the update
        const analyticsSection = document.getElementById('analytics');
        if (analyticsSection) {
            analyticsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    } catch (error) {
        console.error('Error refreshing analytics:', error);
        showAnalyticsPlaceholder(); // Show placeholder if error occurs
        showModelComparisonPlaceholder(); // Show placeholder if error occurs
    }
}

async function loadAvailableDatasets() {
    try {
        const response = await fetch(`${API_BASE_URL}/datasets`);
        const result = await response.json();
        
        if (result.status === 'success') {
            updateDatasetSelect(result.datasets);
            updateCurrentDataset(result.current_dataset);
        } else {
            console.error('Failed to load datasets:', result.message);
        }
    } catch (error) {
        console.error('Error loading datasets:', error);
    }
}

function updateDatasetSelect(datasets) {
    const select = document.getElementById('datasetSelect');
    
    if (datasets.length === 0) {
        select.innerHTML = '<option value="">No datasets available</option>';
        return;
    }
    
    // Clear existing options and add placeholder
    select.innerHTML = '<option value="">Select a dataset to switch...</option>';
    
    datasets.forEach(dataset => {
        const option = document.createElement('option');
        option.value = dataset.path;
        option.textContent = `${dataset.filename} (${dataset.records} records)`;
        select.appendChild(option);
    });
}

function updateCurrentDataset(currentPath) {
    const currentDatasetSpan = document.getElementById('currentDataset');
    const filename = currentPath ? currentPath.split('/').pop() : 'Unknown';
    currentDatasetSpan.textContent = filename;
}

async function switchDataset() {
    const select = document.getElementById('datasetSelect');
    const selectedPath = select.value;
    
    if (!selectedPath) {
        alert('Please select a dataset first');
        return;
    }
    
    if (confirm('Are you sure you want to switch datasets? This will retrain all models.')) {
        try {
            const response = await fetch(`${API_BASE_URL}/datasets/switch`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ dataset_path: selectedPath })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                // Update UI
                updateCurrentDataset(selectedPath);
                
                // Show success message
                alert(`Successfully switched to ${result.dataset_info.filename}!\n\nDataset Info:\n- Records: ${result.dataset_info.total_records}\n- Date Range: ${result.dataset_info.date_range}\n\nModels have been retrained with the new dataset.`);
                
                // Refresh data
                await loadWeatherData();
                loadSessionData();
                
                // Clear any existing results
                hideResult('predictionResult');
                hideResult('forecastResult');
                hideResult('uploadResult');
                
                // Force refresh analytics section with delay
                setTimeout(() => {
                    refreshAnalytics();
                }, 1000);
                
            } else {
                alert('Failed to switch dataset: ' + result.message);
            }
        } catch (error) {
            console.error('Error switching dataset:', error);
            alert('Failed to switch dataset. Please check server connection.');
        }
    }
}
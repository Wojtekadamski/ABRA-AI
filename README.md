# ABRA-AI Microservice

Neural Network-based traffic prediction microservice for the ABRA system.

## Overview

ABRA-AI is a FastAPI-based microservice that uses a Multi-Layer Perceptron (MLP) Neural Network to predict traffic loads and suggest scaling actions (SCALE_UP, SCALE_DOWN, MAINTAIN) for servers.

## Features

- **Neural Network Prediction**: Uses scikit-learn's MLPRegressor with 3 hidden layers (64, 32, 16 neurons)
- **Automatic Training**: Model trains on synthetic data during startup using FastAPI's lifespan context
- **RESTful API**: Clean endpoints for predictions, health checks, and model info
- **Batch Processing**: Support for predicting multiple servers at once
- **Docker Ready**: Containerized for easy deployment
- **Comprehensive Logging**: Detailed logging for monitoring and debugging

## Project Structure

```
ABRA-AI/
├── app/
│   ├── __init__.py       # Package initialization
│   ├── main.py           # FastAPI application with lifespan
│   ├── model.py          # Neural Network (MLPRegressor) logic
│   ├── schemas.py        # Pydantic models for validation
│   └── utils.py          # Helper functions
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
└── README.md            # This file
```

## Requirements

- Python 3.9+
- FastAPI
- Uvicorn
- Scikit-learn
- NumPy
- Pydantic

## Installation

### Local Development

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the service**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Docker Deployment

1. **Build the image**:
   ```bash
   docker build -t abra-ai:latest .
   ```

2. **Run the container**:
   ```bash
   docker run -d -p 8000:8000 --name abra-ai abra-ai:latest
   ```

## API Endpoints

### Health & Info

- **GET /** - Root endpoint with service status
- **GET /health** - Health check endpoint
- **GET /model/info** - Get model information and statistics

### Predictions

- **POST /predict** - Single server prediction
  ```json
  {
    "server_id": "server-001",
    "recent_traffic": [45.2, 52.1, 48.9, 55.3, 60.7]
  }
  ```
  
  Response:
  ```json
  {
    "server_id": "server-001",
    "predicted_load": 65.8,
    "action_suggested": "SCALE_UP",
    "model_version": "1.0.0"
  }
  ```

- **POST /predict/batch** - Batch prediction for multiple servers

## Scaling Logic

The service uses simple threshold-based logic:
- **SCALE_UP**: Predicted load > 80
- **SCALE_DOWN**: Predicted load < 30
- **MAINTAIN**: Predicted load between 30-80

## Model Details

- **Algorithm**: Multi-Layer Perceptron (MLPRegressor)
- **Architecture**: 
  - Input layer: 5 features (recent traffic values)
  - Hidden layers: 64 → 32 → 16 neurons
  - Output layer: 1 neuron (predicted load)
- **Activation**: ReLU
- **Optimizer**: Adam
- **Training**: Synthetic data with daily cycles, trends, and noise
- **Regularization**: L2 (alpha=0.001)
- **Early Stopping**: Enabled with validation split

## Testing

Access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

Example curl command:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "server_id": "test-server",
    "recent_traffic": [45.0, 50.0, 55.0, 60.0, 65.0]
  }'
```

## Integration with ABRA System

This microservice is designed to work alongside:
- **ABRA-backend**: Java/Spring Boot backend
- **ABRA-frontend**: React frontend
- **ABRA-mock-servers**: Mock server infrastructure

Use Docker Compose to orchestrate all services together.

## Development

### Adding New Features

1. Model improvements: Modify `app/model.py`
2. API changes: Update `app/main.py` and `app/schemas.py`
3. Utilities: Add helpers to `app/utils.py`

### Logging

The service uses Python's standard logging module. Logs include:
- Startup and training progress
- Prediction requests and results
- Errors and warnings

## License

Part of the ABRA project for academic purposes.

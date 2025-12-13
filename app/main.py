from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import Dict

from app.model import TrafficPredictor
from app.schemas import TrafficPayload, PredictionResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

predictor: TrafficPredictor = None

SCALE_UP_THRESHOLD = 80.0
SCALE_DOWN_THRESHOLD = 30.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ABRA-AI Microservice")
    
    global predictor
    predictor = TrafficPredictor()
    predictor.train_on_synthetic_data(n_samples=1000)
    
    logger.info(f"Model ready: {predictor.get_model_info()}")
    
    yield
    
    logger.info("Shutting down ABRA-AI Microservice")


app = FastAPI(
    title="ABRA-AI Traffic Prediction Service",
    description="Neural Network-based microservice for traffic load prediction",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict:
    return {
        "service": "ABRA-AI Traffic Prediction",
        "status": "operational",
        "version": "1.0.0",
        "model_trained": predictor.is_trained if predictor else False
    }


@app.get("/health")
async def health_check() -> Dict:
    if predictor is None or not predictor.is_trained:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    return {
        "status": "healthy",
        "model_status": "trained",
        "model_version": predictor.model_version
    }


@app.get("/model/info")
async def get_model_info() -> Dict:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    return predictor.get_model_info()


@app.post("/predict", response_model=PredictionResult)
async def predict_traffic(payload: TrafficPayload) -> PredictionResult:
    if predictor is None or not predictor.is_trained:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    try:
        logger.info(f"Prediction request: {payload.server_id}")
        predicted_load = predictor.predict(payload.recent_traffic)
        
        if predicted_load > SCALE_UP_THRESHOLD:
            action = "SCALE_UP"
        elif predicted_load < SCALE_DOWN_THRESHOLD:
            action = "SCALE_DOWN"
        else:
            action = "MAINTAIN"
        
        result = PredictionResult(
            server_id=payload.server_id,
            predicted_load=round(predicted_load, 2),
            action_suggested=action,
            model_version=predictor.model_version
        )
        
        logger.info(f"{payload.server_id}: {action} ({predicted_load:.2f})")
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict/batch")
async def predict_traffic_batch(payloads: list[TrafficPayload]) -> list[PredictionResult]:
    if predictor is None or not predictor.is_trained:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    results = []
    for payload in payloads:
        try:
            predicted_load = predictor.predict(payload.recent_traffic)
            
            if predicted_load > SCALE_UP_THRESHOLD:
                action = "SCALE_UP"
            elif predicted_load < SCALE_DOWN_THRESHOLD:
                action = "SCALE_DOWN"
            else:
                action = "MAINTAIN"
            
            results.append(PredictionResult(
                server_id=payload.server_id,
                predicted_load=round(predicted_load, 2),
                action_suggested=action,
                model_version=predictor.model_version
            ))
        except Exception as e:
            logger.error(f"Error processing {payload.server_id}: {str(e)}")
            continue
    
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

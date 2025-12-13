import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficPredictor:
    def __init__(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10,
            alpha=0.001
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_version = "1.0.0"
        
    def train_on_synthetic_data(self, n_samples: int = 1000):
        logger.info(f"Generating {n_samples} synthetic samples...")
        
        X_train = []
        y_train = []
        
        for i in range(n_samples):
            time_offset = i * 0.1
            
            base_pattern = 50 + 30 * np.sin(time_offset)
            weekly_trend = 10 * np.sin(time_offset / 7)
            noise = np.random.normal(0, 5, 5)
            spike = np.random.choice([0, 20], size=5, p=[0.9, 0.1])
            
            traffic_sequence = base_pattern + weekly_trend + noise + spike
            traffic_sequence = np.maximum(traffic_sequence, 0)
            X_train.append(traffic_sequence)
            
            next_value = base_pattern + weekly_trend + np.random.normal(2, 5) + \
                        np.random.choice([0, 20], p=[0.9, 0.1])
            next_value = max(next_value, 0)
            y_train.append(next_value)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        
        train_score = self.model.score(X_train_scaled, y_train)
        logger.info(f"Training complete: R² = {train_score:.4f}, iterations = {self.model.n_iter_}")
        
    def predict(self, recent_traffic: List[float]) -> float:
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if len(recent_traffic) != 5:
            raise ValueError("Expected 5 traffic values")
        
        X_input = np.array(recent_traffic).reshape(1, -1)
        X_scaled = self.scaler.transform(X_input)
        prediction = self.model.predict(X_scaled)[0]
        
        return float(max(prediction, 0))
    
    def get_model_info(self) -> dict:
        return {
            "model_type": "MLPRegressor",
            "version": self.model_version,
            "is_trained": self.is_trained,
            "hidden_layers": self.model.hidden_layer_sizes,
            "activation": self.model.activation,
            "solver": self.model.solver,
            "n_iterations": self.model.n_iter_ if self.is_trained else None,
            "loss": float(self.model.loss_) if self.is_trained else None
        }

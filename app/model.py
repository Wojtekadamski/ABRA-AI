import numpy as np
from sklearn.neural_network import MLPRegressor # Wielowarstwowy Perceptron do regresji
from sklearn.preprocessing import StandardScaler # Skalowanie cech
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficPredictor:
    def __init__(self):
        self.window_size = 20 # Liczba ostatnich punktów używanych do predykcji (Musi być taka sama jak HISTORY_WINDOW_SIZE w Javie!)

        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16), # Architektura sieci (neurony w warstwach ukrytych)
            activation='relu', # Funkcja aktywacji ReLU 
            solver='adam', # Optymalizator 
            max_iter=1000, # Maksymalna liczba iteracji treningu
            random_state=42, # Ustawienie ziarna losowości dla powtarzalności
            early_stopping=True, # Wczesne zatrzymanie, aby uniknąć przeuczenia
            validation_fraction=0.2, # Frakcja danych walidacyjnych do wczesnego zatrzymania 
            n_iter_no_change=10, # Liczba iteracji bez poprawy przed zatrzymaniem 
            alpha=0.001 # Współczynnik regularyzacji L2 
        )
        self.scaler = StandardScaler() # Skalowanie cech 
        self.is_trained = False # Flaga wskazująca, czy model jest wytrenowany
        self.model_version = "1.0.0" # Wersja modelu 
        
    def train_on_synthetic_data(self, n_samples: int = 1000):
        logger.info(f"Generating {n_samples} synthetic samples (WINDOW={self.window_size})...")
        
        X_train = []
        y_train = []
        
        for i in range(n_samples): # Generowanie syntetycznych danych
            time_offset = i * 0.1
            
            base_pattern = 50 + 30 * np.sin(time_offset)
            weekly_trend = 10 * np.sin(time_offset / 7)
            
            # Używamy zmiennej self.window_size zamiast sztywnej liczby
            noise = np.random.normal(0, 5, self.window_size) 
            spike = np.random.choice([0, 20], size=self.window_size, p=[0.9, 0.1]) 
            
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
        
        # --- DYNAMICZNE DOPASOWANIE ROZMIARU ---
        current_size = len(recent_traffic)
        
        # Jeśli Java wysłała za mało, uzupełnij zerami z przodu
        if current_size < self.window_size:
            padding = [0.0] * (self.window_size - current_size)
            recent_traffic = padding + recent_traffic
            
        # Jeśli Java wysłała za dużo, weź tylko ostatnie N elementów
        elif current_size > self.window_size:
            recent_traffic = recent_traffic[-self.window_size:]
            
        # Teraz mamy pewność, że rozmiar pasuje do self.window_size
        # ----------------------------------------
        
        X_input = np.array(recent_traffic).reshape(1, -1)
        X_scaled = self.scaler.transform(X_input)
        prediction = self.model.predict(X_scaled)[0]
        
        return float(max(prediction, 0))
    
    def get_model_info(self) -> dict:
        return {
            "model_type": "MLPRegressor",
            "version": self.model_version,
            "is_trained": self.is_trained,
            "window_size": self.window_size, # Dodajemy informację o oknie do API
            "hidden_layers": self.model.hidden_layer_sizes,
            "activation": self.model.activation,
            "solver": self.model.solver,
            "n_iterations": self.model.n_iter_ if self.is_trained else None,
            "loss": float(self.model.loss_) if self.is_trained else None
        }
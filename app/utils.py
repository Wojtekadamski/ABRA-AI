import numpy as np
from typing import List, Tuple
from datetime import datetime


def calculate_traffic_statistics(traffic_data: List[float]) -> dict:
    if not traffic_data:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "trend": "stable"}
    
    data = np.array(traffic_data)
    
    if len(data) > 1:
        slope = np.polyfit(range(len(data)), data, 1)[0]
        trend = "increasing" if slope > 1 else "decreasing" if slope < -1 else "stable"
    else:
        trend = "stable"
    
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "trend": trend
    }


def detect_traffic_anomaly(recent_traffic: List[float], threshold: float = 2.5) -> Tuple[bool, str]:
    if len(recent_traffic) < 2:
        return False, "Insufficient data"
    
    data = np.array(recent_traffic)
    mean = np.mean(data[:-1])
    std = np.std(data[:-1])
    latest = data[-1]
    
    if std == 0:
        return False, "No variation in data"
    
    z_score = abs((latest - mean) / std)
    
    if z_score > threshold:
        msg = "spike" if latest > mean else "drop"
        return True, f"Traffic {msg} (z={z_score:.2f})"
    
    return False, "Normal"


def format_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def normalize_traffic_values(traffic_data: List[float], target_range: Tuple[float, float] = (0, 100)) -> List[float]:
    if not traffic_data:
        return []
    
    data = np.array(traffic_data)
    data_min = np.min(data)
    data_max = np.max(data)
    
    if data_max == data_min:
        return [target_range[0]] * len(data)
    
    normalized = (data - data_min) / (data_max - data_min)
    scaled = normalized * (target_range[1] - target_range[0]) + target_range[0]
    
    return scaled.tolist()


def calculate_confidence_score(recent_traffic: List[float]) -> float:
    if len(recent_traffic) < 5:
        return 0.5
    
    data = np.array(recent_traffic)
    
    completeness_score = 0.7 if np.any(data <= 0) else 1.0
    
    cv = np.std(data) / (np.mean(data) + 1e-6)
    if cv < 0.2:
        consistency_score = 1.0
    elif cv < 0.5:
        consistency_score = 0.8
    else:
        consistency_score = 0.6
    
    return round((completeness_score + consistency_score) / 2, 2)

"""Machine learning models for risk prediction.

Supports:
- LSTM (requires tensorflow/keras)
- XGBoost (requires xgboost)
- Heuristic baseline (no dependencies)
"""

from typing import List, Dict, Tuple, Optional
import json

try:
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    np = None

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    HAS_LSTM = True
except ImportError:
    HAS_LSTM = False
    keras = None

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None


class LSTMPredictor:
    """LSTM model for risk trajectory prediction."""
    
    def __init__(self, seq_length: int = 7, feature_dim: int = 8):
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.model = None
        self.scaler = None
        
    def build(self):
        if not HAS_LSTM:
            raise RuntimeError("tensorflow/keras not installed")
        self.model = Sequential([
            LSTM(32, input_shape=(self.seq_length, self.feature_dim)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # binary: escalation or not
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return self.model
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32):
        if self.model is None:
            self.build()
        if HAS_SKLEARN and self.scaler is None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X.reshape(-1, self.feature_dim)).reshape(X.shape)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not built")
        if HAS_SKLEARN and self.scaler:
            X = self.scaler.transform(X.reshape(-1, self.feature_dim)).reshape(X.shape)
        return self.model.predict(X, verbose=0)


class XGBoostPredictor:
    """XGBoost model for risk prediction."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def build(self):
        if not HAS_XGBOOST:
            raise RuntimeError("xgboost not installed")
        self.model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
        return self.model
    
    def train(self, X: np.ndarray, y: np.ndarray):
        if self.model is None:
            self.build()
        if HAS_SKLEARN and self.scaler is None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not built")
        if HAS_SKLEARN and self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict(X)


class HeuristicPredictor:
    """Fallback heuristic predictor (no ML dependencies)."""
    
    def predict(self, features_list: List[Dict]) -> List[float]:
        """Given a list of feature dicts, return escalation probabilities."""
        predictions = []
        for feats in features_list:
            # simple rule-based: high freq_slope + high risk_slope -> escalation
            freq_slope = feats.get("freq_slope", 0)
            risk_slope = feats.get("risk_slope", 0)
            high_risk = feats.get("high_risk_count", 0)
            new_conns = feats.get("new_connections", 0)
            
            # heuristic score (0-1)
            score = 0.5
            score += 0.2 * min(1.0, abs(risk_slope) / 0.5)
            score += 0.15 * min(1.0, freq_slope / 2.0)
            score += 0.15 * min(1.0, high_risk / 5.0)
            score += 0.1 * min(1.0, new_conns / 10.0)
            score = min(0.99, max(0.01, score))
            predictions.append(score)
        return predictions


def create_labeled_sequences(
    events_df: List[Dict],
    sequence_length: int = 7,
    predict_window: int = 7
) -> Tuple[List, List]:
    """Create time-series sequences from events for training.
    
    Args:
        events_df: list of event dicts with 'person', 'risk_score', 'timestamp'
        sequence_length: look-back window length (days)
        predict_window: forecast window (days)
    
    Returns:
        (X, y) where X is feature sequences and y is labels (0/1 for escalation)
    """
    X, y = [], []
    # placeholder implementation: expects structured event data
    return X, y


def train_model(model_type: str, train_data: List[Dict], epochs: int = 10) -> object:
    """Train a model given data and model type.
    
    Args:
        model_type: 'lstm' | 'xgboost' | 'heuristic'
        train_data: list of feature dicts
        epochs: training epochs (for LSTM)
    
    Returns:
        Trained model object
    """
    if model_type == "lstm":
        model = LSTMPredictor()
        model.build()
        if HAS_SKLEARN and HAS_LSTM and np:
            # create dummy sequences
            X = np.random.randn(len(train_data), 7, 8)
            y = np.random.randint(0, 2, len(train_data))
            model.train(X, y, epochs=epochs)
        return model
    elif model_type == "xgboost":
        model = XGBoostPredictor()
        model.build()
        if HAS_SKLEARN and HAS_XGBOOST and np:
            X = np.random.randn(len(train_data), 8)
            y = np.random.randint(0, 2, len(train_data))
            model.train(X, y)
        return model
    else:
        return HeuristicPredictor()

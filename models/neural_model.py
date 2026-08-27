"""
Neural Network Architecture for Order Flow Time-Series.

Deep Multi-Layer Perceptron (MLP) with Batch Normalization, Dropout,
and Cosine Annealing learning rate schedule.
Provides PyTorch implementation with fallback to sklearn MLP.
"""

import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not found. Using sklearn MLP fallback.")


if HAS_TORCH:
    class PyTorchMLP(nn.Module):
        def __init__(self, input_dim: int, num_classes: int = 3, is_classification: bool = True):
            super().__init__()
            self.is_classification = is_classification
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes if is_classification else 1)
            )

        def forward(self, x):
            return self.net(x)


class NeuralNetworkModel:
    """
    Deep Neural Network wrapper supporting PyTorch with Scikit-Learn MLP fallback.
    """

    def __init__(
        self,
        is_classification: bool = True,
        num_classes: int = 3,
        epochs: int = 25,
        batch_size: int = 64,
        lr: float = 0.001,
        seed: int = 42
    ):
        self.is_classification = is_classification
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)

        if HAS_TORCH:
            torch.manual_seed(self.seed)
            input_dim = X.shape[1]
            self.model = PyTorchMLP(input_dim, self.num_classes, self.is_classification)
            
            if self.is_classification:
                criterion = nn.CrossEntropyLoss()
                y_tensor = torch.tensor(y, dtype=torch.long)
            else:
                criterion = nn.MSELoss()
                y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

            self.model.train()
            for epoch in range(self.epochs):
                for b_x, b_y in loader:
                    optimizer.zero_grad()
                    out = self.model(b_x)
                    loss = criterion(out, b_y)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
        else:
            if self.is_classification:
                self.model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=self.epochs*5, random_state=self.seed)
            else:
                self.model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=self.epochs*5, random_state=self.seed)
            self.model.fit(X_scaled, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        if HAS_TORCH:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                out = self.model(X_tensor)
                if self.is_classification:
                    preds = torch.argmax(out, dim=1).numpy()
                else:
                    preds = out.squeeze(1).numpy()
            return preds
        else:
            return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_classification:
            raise ValueError("predict_proba is only available for classification models.")
        X_scaled = self.scaler.transform(X)
        if HAS_TORCH:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                out = self.model(X_tensor)
                probs = torch.softmax(out, dim=1).numpy()
            return probs
        else:
            return self.model.predict_proba(X_scaled)

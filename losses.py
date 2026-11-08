from abc import ABC, abstractmethod
import math

from understand_ml.types import LossType


class BaseLoss(ABC):
    @property
    @abstractmethod
    def type(self) -> LossType: pass

    @abstractmethod
    def forward(self, pred, target): pass

    @abstractmethod
    def backward(self, pred, target): pass


# --- Конкретные математические реализации ---
class MSELoss(BaseLoss):
    @property
    def type(self):
        return LossType.MSE_LOSS

    def forward(self, pred, target):
        return 0.5 * (pred - target) ** 2

    def backward(self, pred, target):
        return pred - target


class MAELoss(BaseLoss):
    @property
    def type(self):
        return LossType.MAE_LOSS

    def forward(self, pred, target):
        return abs(pred - target)

    def backward(self, pred, target):
        if pred > target: return 1.0
        elif pred < target: return -1.0
        return 0.0


class RMSELoss(BaseLoss):
    @property
    def type(self):
        return LossType.RMSE_LOSS

    def forward(self, pred, target):
        self.current_mse = 0.5 * (pred - target) ** 2
        return math.sqrt(self.current_mse + 1e-15)

    def backward(self, pred, target):
        loss_val = math.sqrt(self.current_mse + 1e-15)
        return (0.5 * (pred - target)) / loss_val


class BCELoss(BaseLoss):
    @property
    def type(self):
        return LossType.BCE_LOSS

    def forward(self, pred, target):
        eps = 1e-15
        p = max(eps, min(1.0 - eps, pred))
        return -(target * math.log(p) + (1.0 - target) * math.log(1.0 - p))

    def backward(self, pred, target):
        eps = 1e-15
        p = max(eps, min(1.0 - eps, pred))
        return (p - target) / (p * (1.0 - p))


class CategoricalCrossEntropyLoss(BaseLoss):
    @property
    def type(self):
        return LossType.CCE_LOSS

    def forward(self, pred, target):
        if not isinstance(pred, list) or not isinstance(target, list):
            raise TypeError("[CCE Loss] Входные данные должны быть списками.")
        eps = 1e-15
        loss_sum = 0.0
        for p, y in zip(pred, target):
            p_clipped = max(eps, min(1.0 - eps, p))
            loss_sum += y * math.log(p_clipped)
        return -loss_sum

    def backward(self, pred, target):
        eps = 1e-15
        gradients = []
        for p, y in zip(pred, target):
            p_clipped = max(eps, min(1.0 - eps, p))
            gradients.append(-y / p_clipped)
        return gradients


# --- Классы-контейнеры (Фабрики) ---
class Regression:
    @staticmethod
    def mse() -> BaseLoss: return MSELoss()

    @staticmethod
    def mae() -> BaseLoss: return MAELoss()

    @staticmethod
    def rmse() -> BaseLoss: return RMSELoss()


class BinaryClassification:
    @staticmethod
    def bce() -> BaseLoss: return BCELoss()


class MultiClassClassification:
    @staticmethod
    def cce() -> BaseLoss: return CategoricalCrossEntropyLoss()

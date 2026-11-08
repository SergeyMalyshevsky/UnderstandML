from enum import Enum, auto

class LayerType(Enum):
    # Параметрические слои
    LINEAR = auto()
    CONV2D = auto()
    BATCH_NORM_1D = auto()
    BATCH_NORM_2D = auto()

    # Непараметрические слои / активации
    DROPOUT = auto()
    FLATTEN = auto()
    LINEAR_ACT = auto()
    RELU_ACT = auto()
    SIGMOID_ACT = auto()
    SOFTMAX_ACT = auto()

    # Сложные блоки
    CONV_RESIDUAL_BLOCK = auto()

class LossType(Enum):
    MSE_LOSS = auto()
    MAE_LOSS = auto()
    RMSE_LOSS = auto()

    BCE_LOSS = auto()
    CCE_LOSS = auto()

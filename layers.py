from abc import ABC, abstractmethod
import random
import math
from understand_ml.types import LayerType


class BaseLayer(ABC):
    @property
    @abstractmethod
    def type(self) -> LayerType: pass

    @abstractmethod
    def forward(self, x): pass

    @abstractmethod
    def backward(self, next_gradient): pass

    @abstractmethod
    def update_params(self, optimizer): pass

    def zero_grad(self): pass

    def average_grad(self, batch_size): pass

    def check_grad(self): return True

    def update_layer(self, layer, learning_rate, weight_decay_l1, weight_decay_l2): pass


class LinearLayer(BaseLayer):
    @property
    def type(self):
        return LayerType.LINEAR

    def __init__(self, in_features, out_features):
        limit = math.sqrt(6.0 / (in_features + out_features))
        self.w = [[random.uniform(-limit, limit) for _ in range(in_features)] for _ in range(out_features)]
        self.b = [0.0] * out_features
        self.zero_grad()

    def _regularizate(self, dw, w_val, weight_decay_l1, weight_decay_l2):
        if weight_decay_l2 > 0.0:
            dw += weight_decay_l2 * w_val

        if weight_decay_l1 > 0.0:
            # copysign(1.0, x) вернет 1.0 или -1.0,
            # а при w_val == 0.0 сохраняем 0.0
            sign = math.copysign(1.0, w_val) if w_val != 0.0 else 0.0
            dw += weight_decay_l1 * sign

    def zero_grad(self):
        out_features = len(self.w)
        in_features = len(self.w[0]) if out_features > 0 else 0
        self.w_grad_acc = [[0.0] * in_features for _ in range(out_features)]
        self.b_grad_acc = [0.0] * out_features

    def forward(self, x):
        self.x = x
        return [sum(wij * xj for wij, xj in zip(wi, self.x)) + bi for wi, bi in zip(self.w, self.b)]

    def backward(self, next_gradient):
        for i in range(len(self.w)):
            for j in range(len(self.w[i])):
                self.w_grad_acc[i][j] += next_gradient[i] * self.x[j]
            self.b_grad_acc[i] += next_gradient[i]
        return [sum(next_gradient[i] * self.w[i][j] for i in range(len(self.w))) for j in range(len(self.w[0]))]

    def average_grad(self, batch_size):
        self.w_grad = [[val / batch_size for val in row] for row in self.w_grad_acc]
        self.b_grad = [val / batch_size for val in self.b_grad_acc]

    def update_params(self, optimizer):
        for i in range(len(self.w)):
            for j in range(len(self.w[i])):
                dw = self.w_grad[i][j]
                if optimizer.l2 > 0.0: dw += optimizer.l2 * self.w[i][j]
                if optimizer.l1 > 0.0:
                    sign = 1.0 if self.w[i][j] > 0 else (-1.0 if self.w[i][j] < 0 else 0.0)
                    dw += optimizer.l1 * sign
                self.w[i][j] -= optimizer.learning_rate * dw
            self.b[i] -= optimizer.learning_rate * self.b_grad[i]

    def update_layer(self, layer, learning_rate, weight_decay_l1, weight_decay_l2):
        for i, row in enumerate(layer.w):
            for j, w_val in enumerate(row):
                dw = layer.w_grad[i][j]
                self._regularizate(dw, w_val, weight_decay_l1, weight_decay_l2)
                layer.w[i][j] -= learning_rate * dw

            # Обновление смещений (B) вынесено из внутреннего цикла j, но остается в i
            layer.b[i] -= learning_rate * layer.b_grad[i]

        return layer


class BatchNormLayer(BaseLayer):
    @property
    def type(self):
        return LayerType.BATCH_NORM_1D

    def __init__(self, num_features, eps=1e-5):
        self.eps = eps
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features
        self.momentum = 0.1
        self.zero_grad()

    def zero_grad(self):
        self.gamma_acc = [0.0] * len(self.gamma)
        self.beta_acc = [0.0] * len(self.beta)

    def forward(self, x):
        self.x = x
        self.x_norm = [0.0] * len(x)
        out = [0.0] * len(x)
        for i in range(len(x)):
            self.running_mean[i] = (1 - self.momentum) * self.running_mean[i] + self.momentum * x[i]
            dev = x[i] - self.running_mean[i]
            self.running_var[i] = (1 - self.momentum) * self.running_var[i] + self.momentum * (dev ** 2)
            self.x_norm[i] = (x[i] - self.running_mean[i]) / math.sqrt(self.running_var[i] + self.eps)
            out[i] = self.gamma[i] * self.x_norm[i] + self.beta[i]
        return out

    def backward(self, next_gradient):
        for i in range(len(next_gradient)):
            self.gamma_acc[i] += next_gradient[i] * self.x_norm[i]
            self.beta_acc[i] += next_gradient[i] * 1.0
        return [next_gradient[i] * self.gamma[i] / math.sqrt(self.running_var[i] + self.eps) for i in
                range(len(next_gradient))]

    def average_grad(self, batch_size):
        self.gamma_grad = [val / batch_size for val in self.gamma_acc]
        self.beta_grad = [val / batch_size for val in self.beta_acc]

    def update_params(self, optimizer):
        for i in range(len(self.gamma)):
            self.gamma[i] -= optimizer.learning_rate * self.gamma_grad[i]
            self.beta[i] -= optimizer.learning_rate * self.beta_grad[i]

    def update_layer(self, layer, learning_rate, weight_decay_l1, weight_decay_l2):
        for i in range(len(layer.gamma)):
            layer.gamma[i] -= learning_rate * layer.gamma_grad[i]
            layer.beta[i] -= learning_rate * layer.beta_grad[i]


class DropoutLayer(BaseLayer):
    @property
    def type(self):
        return LayerType.DROPOUT

    def __init__(self, p=0.2):
        self.p = p
        self.is_training = True

    def forward(self, x):
        if not self.is_training or self.p == 0.0: return x
        self.mask = [1.0 if random.random() >= self.p else 0.0 for _ in x]
        return [xi * m / (1.0 - self.p) for xi, m in zip(x, self.mask)]

    def backward(self, next_gradient):
        if not self.is_training or self.p == 0.0: return next_gradient
        return [grad * m / (1.0 - self.p) for grad, m in zip(next_gradient, self.mask)]

    # ИСПРАВЛЕНО: Добавлена пустая реализация абстрактного метода
    def update_params(self, optimizer):
        pass


class Conv2DLayer(BaseLayer):
    @property
    def type(self):
        return LayerType.CONV2D

    def __init__(self, kernel_size=3):
        self.k_size = kernel_size
        limit = math.sqrt(6.0 / (kernel_size * kernel_size + kernel_size * kernel_size))
        self.w = [[random.uniform(-limit, limit) for _ in range(kernel_size)] for _ in range(kernel_size)]
        self.b = 0.0
        self.zero_grad()

    def zero_grad(self):
        self.w_grad_acc = [[0.0] * self.k_size for _ in range(self.k_size)]
        self.b_grad_acc = 0.0

    def forward(self, x_matrix):
        self.x = x_matrix
        h = len(x_matrix)
        w = len(x_matrix[0]) if h > 0 else 0
        out_h, out_w = h - self.k_size + 1, w - self.k_size + 1
        self.z = [[0.0] * out_w for _ in range(out_h)]
        for i in range(out_h):
            for j in range(out_w):
                self.z[i][j] = sum(x_matrix[i + ki][j + kj] * self.w[ki][kj] for ki in range(self.k_size) for kj in
                                   range(self.k_size)) + self.b
        return self.z

    def backward(self, next_gradient):
        out_h = len(next_gradient)
        out_w = len(next_gradient[0]) if out_h > 0 else 0
        self.w_grad = [[0.0] * self.k_size for _ in range(self.k_size)]
        self.b_grad = 0.0
        for i in range(out_h):
            for j in range(out_w):
                delta = next_gradient[i][j]
                self.b_grad_acc += delta
                for ki in range(self.k_size):
                    for kj in range(self.k_size):
                        self.w_grad_acc[ki][kj] += delta * self.x[i + ki][j + kj]
        h_in = len(self.x)
        w_in = len(self.x[0]) if h_in > 0 else 0
        dL_dx = [[0.0] * w_in for _ in range(h_in)]
        for i in range(out_h):
            for j in range(out_w):
                delta = next_gradient[i][j]
                for ki in range(self.k_size):
                    for kj in range(self.k_size):
                        dL_dx[i + ki][j + kj] += delta * self.w[ki][kj]
        return dL_dx

    def average_grad(self, batch_size):
        self.w_grad = [[val / batch_size for val in row] for row in self.w_grad_acc]
        self.b_grad = self.b_grad_acc / batch_size

    def update_params(self, optimizer):
        for ki in range(len(self.w)):
            for kj in range(len(self.w[ki])):
                self.w[ki][kj] -= optimizer.learning_rate * self.w_grad[ki][kj]
        self.b -= optimizer.learning_rate * self.b_grad

    def update_layer(self, layer, learning_rate, weight_decay_l1, weight_decay_l2):
        for ki, row in enumerate(layer.w):
            for kj in range(len(row)):
                layer.w[ki][kj] -= learning_rate * layer.w_grad[ki][kj]
        layer.b -= learning_rate * layer.b_grad


class FlattenLayer(BaseLayer):
    @property
    def type(self): return LayerType.FLATTEN

    def forward(self, x_matrix):
        self.in_h = len(x_matrix)
        self.in_w = len(x_matrix[0]) if self.in_h > 0 else 0
        return [cell for row in x_matrix for cell in row]

    def backward(self, next_gradient_vector):
        return [next_gradient_vector[i: i + self.in_w] for i in range(0, len(next_gradient_vector), self.in_w)]

    # ИСПРАВЛЕНО: Добавлена пустая реализация абстрактного метода
    def update_params(self, optimizer): pass


class ConvResidualBlock(BaseLayer):
    @property
    def type(self):
        return LayerType.CONV_RESIDUAL_BLOCK

    def __init__(self, internal_layers):
        self.sub_layers = internal_layers

    def forward(self, x_matrix):
        self.x_shortcut = x_matrix
        current_signal = x_matrix
        for layer in self.sub_layers:
            current_signal = layer.forward(current_signal)
        return [[c + s for c, s in zip(r_c, r_s)] for r_c, r_s in zip(current_signal, self.x_shortcut)]

    def backward(self, next_gradient_matrix):
        current_gradient = next_gradient_matrix
        for layer in reversed(self.sub_layers):
            current_gradient = layer.backward(current_gradient)
        return [[c + n for c, n in zip(r_c, r_n)] for r_c, r_n in zip(current_gradient, next_gradient_matrix)]

    def zero_grad(self):
        for layer in self.sub_layers: layer.zero_grad()

    def average_grad(self, batch_size):
        for layer in self.sub_layers: layer.average_grad(batch_size)

    def update_params(self, optimizer):
        for layer in self.sub_layers: layer.update_params(optimizer)

import math
from understand_ml.layers import BaseLayer
from understand_ml.types import LayerType

class LinearActivation(BaseLayer):
    @property
    def type(self): return LayerType.LINEAR_ACT

    def forward(self, z_data):
        return z_data

    def backward(self, next_gradient):
        return next_gradient

    def update_params(self, optimizer): pass


class ReLUActivation(BaseLayer):
    @property
    def type(self): return LayerType.RELU_ACT

    def forward(self, z_data):
        self.z = z_data
        if isinstance(z_data, list) and len(z_data) > 0 and isinstance(z_data[0], list):
            return [[max(0.0, cell) for cell in row] for row in z_data]
        return [max(0.0, zi) for zi in z_data]

    def backward(self, next_gradient):
        if isinstance(next_gradient, list) and len(next_gradient) > 0 and isinstance(next_gradient[0], list):
            return [
                [grad if zi > 0 else 0.0 for grad, zi in zip(grad_row, z_row)]
                for grad_row, z_row in zip(next_gradient, self.z)
            ]
        return [grad if zi > 0 else 0.0 for grad, zi in zip(next_gradient, self.z)]

    def update_params(self, optimizer): pass


class SigmoidActivation(BaseLayer):
    @property
    def type(self): return LayerType.SIGMOID_ACT

    def forward(self, z_data):
        def _sig(zi):
            zi_clipped = max(-500.0, min(500.0, zi))
            return 1.0 / (1.0 + math.exp(-zi_clipped))

        if isinstance(z_data, list) and len(z_data) > 0 and isinstance(z_data[0], list):
            self.a = [[_sig(cell) for cell in row] for row in z_data]
        else:
            self.a = [_sig(zi) for zi in z_data]
        return self.a

    def backward(self, next_gradient):
        if isinstance(next_gradient, list) and len(next_gradient) > 0 and isinstance(next_gradient[0], list):
            return [
                [grad * (ai * (1.0 - ai)) for grad, ai in zip(grad_row, a_row)]
                for grad_row, a_row in zip(next_gradient, self.a)
            ]
        return [grad * (ai * (1.0 - ai)) for grad, ai in zip(next_gradient, self.a)]

    def update_params(self, optimizer): pass


class SoftmaxActivation(BaseLayer):
    @property
    def type(self): return LayerType.SOFTMAX_ACT

    def forward(self, z_vector):
        max_z = max(z_vector)
        shift_z = [zi - max_z for zi in z_vector]
        exp_z = [math.exp(zi) for zi in shift_z]
        sum_exp = sum(exp_z)
        self.a = [ex / (sum_exp + 1e-15) for ex in exp_z]
        return self.a

    def backward(self, next_gradient):
        return next_gradient

    def update_params(self, optimizer): pass


class Regression:
    @staticmethod
    def linear(): return LinearActivation()

    @staticmethod
    def relu(): return ReLUActivation()


class BinaryClassification:
    @staticmethod
    def sigmoid(): return SigmoidActivation()


class MultiClassClassification:
    @staticmethod
    def softmax(): return SoftmaxActivation()

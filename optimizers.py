import math

from understand_ml.types import LayerType

class SGD:
    def __init__(self, learning_rate=0.1, weight_decay_l1=0.0, weight_decay_l2=0.0, early_stopping=False, patience=3, tol=1e-6):
        self.learning_rate = learning_rate
        self.l1 = weight_decay_l1
        self.l2 = weight_decay_l2
        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol
        self.best_loss = float('inf')
        self.patience_counter = 0

    def _update_layer(self, layer):
        if not layer.check_grad():
            return

        if layer.type == LayerType.CONV_RESIDUAL_BLOCK:
            # Символ | позволяет объединить несколько кейсов в один блок
                for sub_layer in layer.sub_layers:
                    self._update_layer(sub_layer)
        else:
            layer.update_layer(layer=layer, learning_rate=self.learning_rate, weight_decay_l1=self.l1,
                           weight_decay_l2=self.l2)

    def step(self, model):
        for layer in model.layers:
            self._update_layer(layer)

    def should_stop(self, current_loss):
        if not self.early_stopping: return False
        if current_loss < self.best_loss - self.tol:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        else: self.patience_counter += 1
        if self.patience_counter >= self.patience:
            print(f"\n[Early Stopping] Прервано.")
            return True
        return False

import json

from understand_ml.types import LayerType, LossType


class Sequential:
    def __init__(self, layers_list, loss_module, optimizer_module):
        self.layers = layers_list
        self.loss_fn = loss_module
        self.optimizer = optimizer_module
        self.loss_history = []
        self.is_training = True

    def forward(self, x):
        current_signal = x
        for layer in self.layers:
            if hasattr(layer, 'is_training'):
                layer.is_training = self.is_training
            current_signal = layer.forward(current_signal)
        self.output = current_signal
        return self.output

    def backward(self, y):
        current_gradient = self.loss_fn.backward(self.output, y)
        if self.loss_fn.type == LossType.CCE_LOSS and self.layers[-1].type == LayerType.SOFTMAX_ACT:
            current_gradient = [ai - yi for ai, yi in zip(self.output, y)]
            layers_to_process = self.layers[:-1]
        else:
            layers_to_process = self.layers

        for layer in reversed(layers_to_process):
            current_gradient = layer.backward(current_gradient)

    def fit(self, dataloader, epochs=10):
        self.loss_history = []
        print(f"=== ЗАПУСК MINI-BATCH ОБУЧЕНИЯ (Батч: {dataloader.batch_size}) ===")

        for epoch in range(1, epochs + 1):
            self.is_training = True
            epoch_loss = 0.0
            samples_counted = 0

            for batch_X, batch_Y in dataloader:
                # Перед началом нового батча просим все слои обнулить свои локальные копилки градиентов
                for layer in self.layers:
                    layer.zero_grad()

                # Прогоняем элементы батча построчно
                for x_sample, y_sample in zip(batch_X, batch_Y):
                    pred = self.forward(x_sample)
                    epoch_loss += self.loss_fn.forward(pred, y_sample)
                    samples_counted += 1
                    self.backward(y_sample)  # Каждый слой внутри себя аккумулирует градиент

                # Батч закончился. Просим все слои усреднить накопленные градиенты по размеру батча
                b_len = len(batch_X)
                for layer in self.layers:
                    layer.average_grad(b_len)

                # Делаем ОДИН шаг обновления весов на весь батч
                self.optimizer.step(self)

            avg_loss = epoch_loss / samples_counted
            self.loss_history.append(avg_loss)

            if epochs <= 10 or epoch % 5 == 0 or epoch == epochs:
                print(f"Эпоха {epoch:2d}: Средний Loss = {avg_loss:.6f}")

            if self.optimizer.should_stop(avg_loss):
                break
        print(f"Обучение завершено. Итоговый Loss: {avg_loss:.6f}\n")

    def save_weights(self, filepath):
        model_weights = []
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, 'w'):
                model_weights.append(
                    {'layer_idx': idx, 'layer_type': layer.type.__class__.__name__, 'w': layer.w, 'b': layer.b})
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_weights, f, indent=4)

    def load_weights(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f: model_weights = json.load(f)
        for layer_data in model_weights:
            idx = layer_data['layer_idx']
            self.layers[idx].w = layer_data['w']
            self.layers[idx].b = layer_data['b']

    def print_json_weights(self, filepath, max_items=2):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for layer in data:
            print(f"\n[Слой {layer['layer_idx']}: {layer['layer_type']}]")
            print("  Матрица весов (W): [")
            for i, row in enumerate(layer['w'][:max_items]):
                print(
                    f"    Нейрон {i}: [{', '.join([f'{v:.4f}' for v in row[:max_items]])}{', ...' if len(row) > max_items else ''}]")
            print("  ]")

    def evaluate_confusion_matrix(self, X, Y_one_hot, class_names=None):
        self.is_training = False
        num_classes = len(Y_one_hot)
        matrix = [[0] * num_classes for _ in range(num_classes)]

        for x_sample, y_sample in zip(X, Y_one_hot):
            pred_probs = self.forward(x_sample)
            pred_class = pred_probs.index(max(pred_probs))
            true_class = y_sample.index(1.0)
            matrix[true_class][pred_class] += 1

        if class_names is None:
            class_names = [f"Класс {i}" for i in range(num_classes)]

        print("=== КРАСИВАЯ МАТРИЦА ОШИБОК (CONFUSION MATRIX) ===")
        header = f"{'Истинный / Предск.':<20}" + "".join([f"{name:>15}" for name in class_names])
        print(header)
        print("-" * len(header))
        for i, row in enumerate(matrix):
            print(f"{class_names[i]:<20}" + "".join([f"{cell:>15}" for cell in row]))
        print("-" * len(header))
        return matrix

    def summary(self):
        print("=" * 80)
        print(f"{'ГЛУБОКАЯ НЕЙРОСЕТЬ (ОБЗОР АРХИТЕКТУРЫ)':^80}")
        print("=" * 80)
        header = f"{'Индекс':<8}{'Тип Слоя':<25}{'Размерность параметров':<25}{'Обучаемые параметры':<15}"
        print(header)
        print("-" * 80)
        total_params = 0

        for idx, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            if layer.type == LayerType.LINEAR and hasattr(layer, 'w'):
                out_f = len(layer.w)
                in_f = len(layer.w[0]) if out_f > 0 else 0
                shape_str = f"W: [{out_f}x{in_f}], b: [{out_f}]"
                layer_params = (out_f * in_f) + out_f
                total_params += layer_params
                params_str = f"{layer_params}"
            elif layer.type == LayerType.CONV2D and hasattr(layer, 'w'):
                k = layer.k_size
                shape_str = f"Kernel: [{k}x{k}], b: 1"
                layer_params = (k * k) + 1
                total_params += layer_params
                params_str = f"{layer_params}"
            elif layer.type == LayerType.BATCH_NORM_1D and hasattr(layer, 'gamma'):
                num_f = len(layer.gamma)
                shape_str = f"gamma: [{num_f}], beta: [{num_f}]"
                layer_params = num_f * 2
                total_params += layer_params
                params_str = f"{layer_params}"
            elif layer.type == LayerType.DROPOUT:
                shape_str = f"p = {getattr(layer, 'p', 0.0)}"
                params_str = "0"
            else:
                shape_str = "Нет (Непараметрический)"
                params_str = "0"
            print(f"{idx:<8}{layer_name:<25}{shape_str:<25}{params_str:<15}")

        print("-" * 80)
        print(f"Всего обучаемых параметров в сети: {total_params}")
        print(f"Функция потерь (Loss): {self.loss_fn.__class__.__name__}")
        print(f"Алгоритм оптимизации: {self.optimizer.__class__.__name__}")
        l1_val = getattr(self.optimizer, 'l1', 0.0)
        l2_val = getattr(self.optimizer, 'l2', 0.0)
        reg_status = []
        if l1_val > 0.0: reg_status.append(f"L1 = {l1_val}")
        if l2_val > 0.0: reg_status.append(f"L2 = {l2_val}")
        print(f"Регуляризация весов: {', '.join(reg_status) if reg_status else 'Выключена'}")
        print("=" * 80)

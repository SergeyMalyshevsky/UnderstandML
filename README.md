# understand_ml 🧠

[![Python Version](https://shields.io)](https://python.org)
[![License](https://shields.io)](LICENSE)

**understand_ml** — это учебный, но полностью функциональный модульный мини-фреймворк глубокого обучения, написанный с нуля на Python. Синтаксис, логика разделения обязанностей слоев, принципы обратного распространения ошибки и динамического построения вычислительного графа спроектированы в строгом соответствии с промышленным стандартом **PyTorch**.

Основная цель проекта — продемонстрировать «подкапотную» математику и ООП-архитектуру современных библиотек машинного обучения (слои, батч-нормализация, дропаут, остаточные связи ResNet, автоматическое накопление градиентов) без использования «черных ящиков».

---

## 🚀 Ключевые особенности и возможности

*   **Полный полиморфизм (Паттерн LEGO):** Все вычислительные узлы наследуют единый контракт `BaseLayer` с методами `forward()` и `backward()`, что позволяет собирать архитектуры любой глубины.
*   **Строгая типизация (Прочь "магические строки"):** Управление типами слоев и логика ветвления переведены на безопасные перечисления `LayerType` (`Enum`).
*   **Инкапсулированный бэкпроп (Децентрализация градиентов):** Оптимизатор избавлен от проверок внутренних полей слоев. Каждый параметрический слой сам управляет своими накопителями градиентов батча (`zero_grad()`, `average_grad()`, `update_params()`).
*   **Гибридный граф вычислений (1D и 2D):** Непараметрические слои активаций (ReLU, Sigmoid) обладают инвариантностью к размерности — они автоматически адаптируются как под одномерные векторы, так и под двумерные карты признаков свёрточных блоков.
*   **Продвинутая регуляризация и стабилизация:** Поддержка инвертированного дропаута (`Inverted Dropout`), слоев `BatchNormLayer`, а также L1 (Lasso) и L2 (Ridge Weight Decay) штрафов весов прямо внутри шага `SGD`.
*   **Умная инспекция архитектуры:** Наглядный метод `.summary()`, рассчитывающий общее количество обучаемых параметров и генерирующий пошаговую интерактивную подсказку движения сигналов и дельт.
*   **Промышленная сериализация:** Сохранение состояния матриц параметров в человекочитаемый JSON-формат (`save_weights`) с жесткой валидацией структуры графа при обратной загрузке (`load_weights`).

---

## 📁 Структура проекта

Фреймворк спроектирован по принципу строгого разделения обязанностей (Separation of Concerns):

```text
├── types.py          # Enum-перечисления типов слоев (LayerType)
├── layers.py         # Слои: BaseLayer, LinearLayer, BatchNormLayer, DropoutLayer, Conv2DLayer, FlattenLayer, ConvResidualBlock
├── activations.py    # Функции активации: Linear, ReLU, Sigmoid, Softmax (с фабриками Regression/Classification)
├── losses.py         # Функции потерь и их производные (MSE, MAE, RMSE, BCE, CCE)
├── optimizers.py     # Оптимизатор SGD с Weight Decay и Early Stopping
├── data.py           # Пайплайн данных: DataLoader со встроенным Mini-batch и Shuffle
└── models.py         # Контейнер модели Sequential, валидация (Confusion Matrix), summary и JSON-сериализация
```

---

## 📊 Математика и Граф вычислений

### 1. Архитектура Свёрточного Остаточного Блока (ConvResidualBlock)
Реализует концепцию Skip Connection (ResNet), решая проблему затухания градиентов (\(Output = F(x) + x\)):

```text
 ПРЯМОЙ ПРОХОД (FORWARD):
  Вход [8x8] -> [Conv2D 3x3] -> ReLU -> [ConvResidualBlock (внутри Conv1x1)] -> Flatten -> [Linear 10] -> Softmax
                                            │                          ▲
                                            └───── (Skip Connection) ──┘

 ОБРАТНЫЙ ПРОХОД (BACKWARD):
  Градиент W0 <- [Слой 0] <- ReLU <- [dL/dx + dL/dF (Градиент ResNet)] <- [Flatten] <- [Linear 10] <- Градиент Loss (CCE)
                                            ▲                          │
                                            └───── (Shortcut градиент) ┘
```

### 2. Анатомия полносвязного параметрического слоя (LinearLayer)

```text
                                   [ FORWARD PASS ]
 Входной вектор X  =======================================================>  Выходной вектор Z
 (Сохраняется в self.x)                                                      (z_i = sum(w_ij * x_j) + b_i)
 
 
                                   [ BACKWARD PASS ]
 Градиент по входу dL/dx  <================================================  Пришедший градиент delta
 (dL/dx_j = sum(delta_i * w_ij))                                            |
                                                                            | (Использует self.x)
                                                                            V
                                                                  Градиенты параметров:
                                                                  self.w_grad = delta * x
                                                                  self.b_grad = delta * 1
```

---

## 💻 Быстрый старт (Пример: Свёрточная ResNet на MNIST)

Ниже представлен пример сборки, инспекции и обучения гибридной свёрточно-остаточной сети на датасете рукописных цифр `load_digits` с мини-батчами, L2-регуляризацией и валидацией через Матрицу ошибок:

```python
import random
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from understand_ml.layers import LinearLayer, Conv2DLayer, FlattenLayer, ConvResidualBlock
from understand_ml.models import Sequential
from understand_ml.optimizers import SGD
from understand_ml.data import DataLoader
from understand_ml import activations, losses

# 1. Подготовка данных (MNIST 8x8)
digits = load_digits()
X_digits = [(sample / 16.0).reshape(8, 8).tolist() for sample in digits.data] # 2D матрицы
Y_digits = [[1.0 if i == y else 0.0 for i in range(10)] for y in digits.target.tolist()] # One-Hot

train_loader = DataLoader(X_digits[:200], Y_digits[:200], batch_size=16, shuffle=True)

# 2. Инициализация оптимизатора с затуханием весов
opt = SGD(learning_rate=0.01, weight_decay_l2=1e-5)

# 3. Модульная сборка ResNet
my_resnet = Sequential(
    layers_list=[
        Conv2DLayer(kernel_size=3),                       # Сжатие 8x8 -> 6x6
        activations.Regression.relu(),
        
        ConvResidualBlock(internal_layers=[
            Conv2DLayer(kernel_size=1),                   # Фильтр 1x1 сохраняет размер 6x6
            activations.Regression.relu(),
            Conv2DLayer(kernel_size=1),
            activations.Regression.relu()
        ]),
        
        FlattenLayer(),                                   # Матрица 6x6 -> Вектор (36)
        LinearLayer(in_features=36, out_features=10),
        activations.MultiClassClassification.softmax()
    ],
    loss_module=losses.MultiClassClassification.cce(),
    optimizer_module=opt
)

# 4. Вывод карты архитектуры и параметров
my_resnet.summary()

# 5. Запуск обучения
my_resnet.fit(train_loader, epochs=5)

# 6. Сохранение обученных весов
my_resnet.save_weights("saved_resnet_weights.json")
```

---

## 🛠 Пример вывода метода `.summary()`

```text
================================================================================
                     ГЛУБОКАЯ НЕЙРОСЕТЬ (ОБЗОР АРХИТЕКТУРЫ)                     
================================================================================
Индекс  Тип Слоя                 Размерность параметров   Обучаемые параметры
--------------------------------------------------------------------------------
0       CONV2D                   Kernel: [3x3], b: 1      10             
1       RELU_ACT                 Нет (Непараметрический)  0              
2       CONV_RESIDUAL_BLOCK      Внутренних слоев: 4      Инкапсулированы
3       FLATTEN                  Нет (Непараметрический)  0              
4       LINEAR                   W: [10x36], b: [10]      370            
5       SOFTMAX_ACT              Нет (Непараметрический)  0              
--------------------------------------------------------------------------------
Всего обучаемых параметров в сети: 380
Функция потерь (Loss): CategoricalCrossEntropyLoss
Алгоритм оптимизации: SGD
Регуляризация весов: L2 = 1e-05
================================================================================
```

---

## 📈 Сравнительный анализ интерфейсов классификации

| Критерий | `BinaryClassification` | `MultiClassClassification` |
| :--- | :--- | :--- |
| **Количество классов** | Строго 2 класса | 3 класса и более |
| **Формат таргета (\(y\))** | Скаляр (`0` или `1`) | Вектор One-Hot (например, `[0.0, 1.0, 0.0]`) |
| **Базовая активация** | `Sigmoid` (\(a = \frac{1}{1 + e^{-z}}\)) | `Softmax` (\(a_i = \frac{e^{z_i}}{\sum e^{z_j}}\)) |
| **Финальный Loss** | `BCE` (Binary Cross-Entropy) | `CCE` (Categorical Cross-Entropy) |

---

## 📝 Лицензия

Проект распространяется под лицензией MIT. Подробности см. в файле [LICENSE](LICENSE).

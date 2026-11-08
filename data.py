import random

class DataLoader:
    def __init__(self, X, Y, batch_size=32, shuffle=True):
        """
        DataLoader с поддержкой мини-батчей на чистом Python.
        :param X: Матрица признаков (список списков)
        :param Y: Вектор/матрица ответов (список)
        :param batch_size: Размер мини-пакета (сколько строк выдавать за раз)
        :param shuffle: Нужно ли перемешивать данные перед каждой эпохой
        """
        if len(X) != len(Y):
            raise ValueError(f"Размеры X ({len(X)}) и Y ({len(Y)}) должны совпадать!")

        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        """Возвращает общее количество БАТЧЕЙ в эпохе (а не строк)."""
        # Округляем в большую сторону, чтобы захватить последний неполный батч
        num_samples = len(self.X)
        return (num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """
        Генерирует и выдает мини-батчи данных.
        """
        indices = list(range(len(self.X)))
        if self.shuffle:
            random.shuffle(indices)

        # Нарезаем индексы на кусочки размером batch_size
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]

            # Собираем подматрицу X и подматрицу Y для текущего батча
            batch_X = [self.X[idx] for idx in batch_indices]
            batch_Y = [self.Y[idx] for idx in batch_indices]

            yield batch_X, batch_Y

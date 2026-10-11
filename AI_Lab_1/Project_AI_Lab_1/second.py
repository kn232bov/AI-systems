import numpy as np


# Порогова функція активації f(v) = 1, якщо v > 0; 0, якщо v < 0
def step_function(v):
    return 1 if v > 0 else 0


# 1. Нейрон OR: w1 = 1, w2 = 1, w0 (поріг/bias) = -0.5
def neuron_or(x1, x2):
    v = 1 * x1 + 1 * x2 - 0.5
    return step_function(v)


# 2. Нейрон AND: w1 = 1, w2 = 1, w0 (поріг/bias) = -1.5
def neuron_and(x1, x2):
    v = 1 * x1 + 1 * x2 - 1.5
    return step_function(v)


# 3. Нейрон NOT (для інверсії): w = -1, w0 = 0.5 (або реалізація через лінійний класифікатор)
# Для XOR у просторі (y1, y2): y1 (OR) - 2*y2 (AND) - 0.5 > 0
def neuron_xor(x1, x2):
    y1 = neuron_or(x1, x2)  # Вихід OR
    y2 = neuron_and(x1, x2)  # Вихід AND

    # Розділяюча пряма у просторі (y1, y2): g(y) = y1 - 2*y2 - 0.5
    v_xor = 1 * y1 - 2 * y2 - 0.5
    return step_function(v_xor)


# Перевірка роботи та вивід таблиць істинності
if __name__ == "__main__":
    dataset = [(0, 0), (0, 1), (1, 0), (1, 1)]

    print(f"{'x1':<5} | {'x2':<5} | {'OR (y1)':<8} | {'AND (y2)':<8} | {'XOR':<5}")
    print("-" * 40)

    for x1, x2 in dataset:
        y1 = neuron_or(x1, x2)
        y2 = neuron_and(x1, x2)
        xor_out = neuron_xor(x1, x2)
        print(f"{x1:<5} | {x2:<5} | {y1:<8} | {y2:<8} | {xor_out:<5}")
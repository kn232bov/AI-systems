import random


# 1. Клас одношарового нейрона (Персептрон)
class Perceptron:

    def __init__(self, input_size, lr=0.1, epochs=100):
        self.lr = lr  # Швидкість навчання
        self.epochs = epochs  # Кількість епох (ітерацій)
        # Ініціалізація ваг та порогу випадковими малими значеннями
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(input_size)]
        self.bias = random.uniform(-0.5, 0.5)

    # Порогова функція активації: 1 якщо sum + bias > 0, інакше 0
    def predict(self, x):
        total_sum = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return 1 if total_sum > 0 else 0

    # Процес навчання (знаходження ваг)
    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                output = self.predict(xi)
                error = target - output

                # Якщо є помилка, коригуємо ваги та bias (поріг)
                if error != 0:
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * xi[i]
                    self.bias += self.lr * error


# 2. Вхідні дані (Таблиця істинності)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]  # Вхідні вектори x1, x2

y_or = [0, 1, 1, 1]  # Цільові значення для OR
y_and = [0, 0, 0, 1]  # Цільові значення для AND

# 3. Навчаємо нейрон OR
neuron_or = Perceptron(input_size=2)
neuron_or.train(X, y_or)

# 4. Навчаємо нейрон AND
neuron_and = Perceptron(input_size=2)
neuron_and.train(X, y_and)

# 5. Навчаємо вихідний нейрон XOR (на входах y1 та y2)
# Створюємо новий навчальний набір для вихідного нейрона з результатів OR та AND
X_xor_hidden = []
y_xor = [0, 1, 1, 0]  # Цільові значення для XOR

for xi in X:
    y1 = neuron_or.predict(xi)  # Вихід OR
    y2 = neuron_and.predict(xi)  # Вихід AND
    X_xor_hidden.append([y1, y2])

neuron_xor_output = Perceptron(input_size=2)
neuron_xor_output.train(X_xor_hidden, y_xor)

# 6. Вивід знайдених ваг та перевірка результатів
print("=== ЗНАЙДЕНІ ВАГИ ТА ПОРОГИ В ПРОЦЕСІ НАВЧАННЯ ===")
print(
    f"Нейрон OR:  W1={neuron_or.weights[0]:.2f}, W2={neuron_or.weights[1]:.2f}, W0(bias)={neuron_or.bias:.2f}"
)
print(
    f"Нейрон AND: W1={neuron_and.weights[0]:.2f}, W2={neuron_and.weights[1]:.2f}, W0(bias)={neuron_and.bias:.2f}"
)
print(
    f"Нейрон XOR: W1={neuron_xor_output.weights[0]:.2f}, W2={neuron_xor_output.weights[1]:.2f}, W0(bias)={neuron_xor_output.bias:.2f}"
)

print("\n=== ПЕРЕВІРКА РЕЗУЛЬТАТІВ ===")
print(f"{'x1':<5} | {'x2':<5} | {'OR':<5} | {'AND':<5} | {'XOR':<5}")
print("-" * 35)

for xi in X:
    y1 = neuron_or.predict(xi)
    y2 = neuron_and.predict(xi)
    res_xor = neuron_xor_output.predict([y1, y2])
    print(
        f"{xi[0]:<5} | {xi[1]:<5} | {y1:<5} | {y2:<5} | {res_xor:<5}"
    )
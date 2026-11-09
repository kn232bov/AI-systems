import numpy as np
import matplotlib.pyplot as plt

# Вихідні дані (Варіант 1)
x = np.array([0, 5, 10, 15, 20, 25], dtype=float)
y = np.array([21, 39, 51, 63, 70, 90], dtype=float)
n = len(x)

# 1. Проміжні суми
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_x2 = np.sum(x**2)
sum_xy = np.sum(x * y)

# 2. Обчислення коефіцієнтів МНК
beta1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
beta0 = (sum_y - beta1 * sum_x) / n

# 3. Обчислення значень регресії та похибок
y_pred = beta0 + beta1 * x
errors = y - y_pred
sq_errors = errors**2
S = np.sum(sq_errors)

# Вивід результатів у консоль для переносу в звіт
print("=== ТАБЛИЦЯ ПРОМІЖНИХ СУМ ===")
print(f"n = {n}")
print(f"sum(X) = {sum_x:.2f}, sum(Y) = {sum_y:.2f}")
print(f"sum(X^2) = {sum_x2:.2f}, sum(X*Y) = {sum_xy:.2f}\n")

print("=== КОЕФІЦІЄНТИ РЕГРЕСІЇ ===")
print(f"beta0 (зсув) = {beta0:.4f}")
print(f"beta1 (нахил) = {beta1:.4f}")
print(f"Рівняння: y = {beta0:.4f} + {beta1:.4f} * x\n")

print("=== ТАБЛИЦЯ ПОХИБОК ===")
print("i |  X   |  Y_експ |  Y_розр  |   e_i   |   e_i^2  ")
print("-" * 50)
for i in range(n):
    print(f"{i+1} | {x[i]:4.1f} | {y[i]:6.1f} | {y_pred[i]:8.4f} | {errors[i]:7.4f} | {sq_errors[i]:8.4f}")
print("-" * 50)
print(f"Сума квадратів похибок S = {S:.4f}\n")

# Побудова графіка
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='red', s=60, label='Експериментальні точки (X, Y)', zorder=5)

x_dense = np.linspace(min(x) - 1, max(x) + 1, 100)
y_dense = beta0 + beta1 * x_dense
plt.plot(x_dense, y_dense, color='blue', linewidth=2, label=f'Апроксимуюча пряма: y = {beta0:.2f} + {beta1:.2f}x')

plt.title("Апроксимація МНК", fontsize=12)
plt.xlabel("Величина X", fontsize=11)
plt.ylabel("Величина Y", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
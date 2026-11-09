import numpy as np
import matplotlib.pyplot as plt

# Вхідні вектори даних (вузли інтерполяції)
x_nodes = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y_nodes = np.array([3.2, 3.0, 1.0, 1.8, 1.9])

# КРОК 1. Заповнення матриці X (матриця Вандермонда степеня 4)
X_matrix = np.vander(x_nodes, N=5, increasing=True)

# КРОК 2. Отримання коефіцієнтів полінома [a0, a1, a2, a3, a4]
coeffs = np.linalg.solve(X_matrix, y_nodes)

# КРОК 3. Визначення функції полінома P4(x) та формування його текстового рівняння
def P4(x):
    return sum(c * (x**i) for i, c in enumerate(coeffs))

# Формування рядкового вигляду рівняння полінома для виводу в консоль
terms = []
for i, c in enumerate(coeffs):
    if i == 0:
        terms.append(f"{c:.6f}")
    elif c >= 0:
        terms.append(f"+ {c:.6f}*x^{i}")
    else:
        terms.append(f"- {abs(c):.6f}*x^{i}")
poly_equation = " ".join(terms)

# КРОК 5. Визначення значень функції у проміжних точках (0.2 та 0.5)
x_eval = np.array([0.2, 0.5])
y_eval = P4(x_eval)

# =========================================================
# ВИВІД УСІХ КРОКІВ У КОНСОЛЬ PYCHARM ДЛЯ ЗВІТУ
# =========================================================
np.set_printoptions(precision=6, suppress=True)

print("=== Заповнення матриці X (Вандермонда) ===")
print(X_matrix)

print("\n=== Коефіцієнти полінома ===")
for i, c in enumerate(coeffs):
    print(f"a_{i} = {c:12.6f}")

print("\n=== Визначення рівняння полінома P4(x) ===")
print(f"P4(x) = {poly_equation}")

print("\n=== Значення функції в проміжних точках ===")
print(f"P4(0.2) = {y_eval[0]:.6f}")
print(f"P4(0.5) = {y_eval[1]:.6f}")

# КРОК 4. Побудова графіка функції для інтерполюючого полінома
plt.figure(figsize=(8, 5))

# Плавна крива полінома
x_dense = np.linspace(0.08, 0.72, 300)
plt.plot(x_dense, P4(x_dense), color='purple', linewidth=2, label='Інтерполяційний поліном $P_4(x)$')

# Вихідні вузли інтерполяції
plt.scatter(x_nodes, y_nodes, color='red', s=70, label='Вузли інтерполяції', zorder=5)

# Проміжні обчислені точки
plt.scatter(x_eval, y_eval, color='green', marker='s', s=80, label='Проміжні точки (0.2 та 0.5)', zorder=6)

plt.title("Поліноміальна інтерполяція (4-й степінь)", fontsize=12)
plt.xlabel("X", fontsize=11)
plt.ylabel("Y", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
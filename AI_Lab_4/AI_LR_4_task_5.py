import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as sm

# 1. Генерація даних відповідно до Варіанта 1
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * (X**2) + X + 2 + np.random.randn(m, 1)

# 2. Додавання поліноміальних ознак (2-й степінь)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# 3. Вивід початкових та перетворених ознак для першого елемента
print("=== ПЕРЕТВОРЕННЯ ОЗНАК (КРОК 1) ===")
print("Вхідне значення X[0]:", np.round(X[0], 4))
print("Перетворене значення X_poly[0] [x, x^2]:", np.round(X_poly[0], 4))

# 4. Навчання моделі LinearRegression на розширених даних
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Отримання коефіцієнтів
w1, w2 = lin_reg.coef_[0]
w0 = lin_reg.intercept_[0]

print("\n=== ЗНАЙДЕНІ КОЕФІЦІЄНТИ МОДЕЛІ ===")
print(f"Вільний член (intercept_ w0) = {w0:.4f}")
print(f"Коефіцієнти при x та x^2 (coef_ [w1, w2]) = [{w1:.4f}, {w2:.4f}]")

# 5. Порівняння математичних рівнянь
print("\n=== ПОРІВНЯННЯ МАТЕМАТИЧНИХ МОДЕЛЕЙ ===")
print("Теоретична модель: y = 0.5000 * x^2 + 1.0000 * x + 2.0000 + noise")
print(f"Отримана модель:    y = {w2:.4f} * x^2 + {w1:.4f} * x + {w0:.4f}")

# 6. Оцінка якості регресії
y_pred = lin_reg.predict(X_poly)
print("\n=== МЕТРИКИ ЯКОСТІ ===")
print("R2 score =", round(sm.r2_score(y, y_pred), 4))
print("MAE =", round(sm.mean_absolute_error(y, y_pred), 4))
print("MSE =", round(sm.mean_squared_error(y, y_pred), 4))

# 7. Побудова графіка
X_new = np.linspace(-5.1, 1.1, 200).reshape(-1, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)

plt.figure(figsize=(8, 5))
plt.plot(X, y, "b.", label="Згенеровані дані (x, y)")
plt.plot(X_new, y_new, "r-", linewidth=2.5, label=f"Прогноз: y = {w2:.2f}$x^2$ + {w1:.2f}$x$ + {w0:.2f}")
plt.xlabel("$x_1$", fontsize=12)
plt.ylabel("$y$", fontsize=12)
plt.title("Завдання 2.5: Поліноміальна регресія (Варіант 1)", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
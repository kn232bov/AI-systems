import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Генерація даних відповідно до Варіанта 1
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * (X ** 2) + X + 2 + np.random.randn(m, 1)


# 2. Функція для побудови кривих навчання
def plot_learning_curves(model, X, y, title_text):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_errors, val_errors = [], []

    for m_len in range(1, len(X_train)):
        model.fit(X_train[:m_len], y_train[:m_len])
        y_train_predict = model.predict(X_train[:m_len])
        y_val_predict = model.predict(X_val)

        train_errors.append(mean_squared_error(y_train[:m_len], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.figure(figsize=(8, 5))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Навчальний набір (train)")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Перевірочний набір (val)")
    plt.xlabel("Розмір навчального набору", fontsize=11)
    plt.ylabel("RMSE", fontsize=11)
    plt.title(title_text, fontsize=12)
    plt.ylim(0, 3.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


# --- 1) Криві навчання для Лінійного регресора ---
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y, "Криві навчання: Лінійна модель (Underfitting)")

# --- 2) Криві навчання для Поліноміальної моделі 10-го степеня ---
polynomial_regression_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression())
])
plot_learning_curves(polynomial_regression_10, X, y, "Криві навчання: Поліном 10-го степеня (Overfitting)")

# --- 3) Криві навчання для Поліноміальної моделі 2-го степеня ---
polynomial_regression_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression())
])
plot_learning_curves(polynomial_regression_2, X, y, "Криві навчання: Поліном 2-го степеня (Оптимальна модель)")
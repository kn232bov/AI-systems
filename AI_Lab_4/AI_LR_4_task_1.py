import os
import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 0. Перевірка наявності входного файлу (якщо файлу немає, створюємо його)
# -------------------------------------------------------------
input_file = 'data_singlevar_regr.txt'

if not os.path.exists(input_file):
    print(f"Файл '{input_file}' не знайдено. Створюємо тестовий файл...")
    np.random.seed(42)
    x_synth = np.random.uniform(0, 10, 100)
    y_synth = 2.5 * x_synth + 1.2 + np.random.normal(0, 1.5, 100)
    np.savetxt(input_file, np.column_stack((x_synth, y_synth)), delimiter=',')

# -------------------------------------------------------------
# 1. Завантаження даних з файлу
# -------------------------------------------------------------
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# -------------------------------------------------------------
# 2. Розбивка даних на навчальний та тестовий набори (80% / 20%)
# -------------------------------------------------------------
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Тренувальні дані
X_train, y_train = X[:num_training], y[:num_training]
# Тестові дані
X_test, y_test = X[num_training:], y[num_training:]

# -------------------------------------------------------------
# 3. Створення об'єкта лінійного регресора та його навчання
# -------------------------------------------------------------
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# -------------------------------------------------------------
# 4. Прогнозування результату для тестового набору
# -------------------------------------------------------------
y_test_pred = regressor.predict(X_test)

# -------------------------------------------------------------
# 5. Побудова графіка
# -------------------------------------------------------------
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.title("Завдання 2.1: Регресор однієї змінної")
plt.show()

# -------------------------------------------------------------
# 6. Обчислення та вивід метричних параметрів
# -------------------------------------------------------------
print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# -------------------------------------------------------------
# 7. Збереження моделі у файл pickle
# -------------------------------------------------------------
output_model_file = 'model.pkl'
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

# -------------------------------------------------------------
# 8. Завантаження моделі з файлу та повторне прогнозування
# -------------------------------------------------------------
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

y_test_pred_new = regressor_model.predict(X_test)
print("\nNew mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.metrics as sm

# 1. Визначення імені файлу для Варіанта 1
input_file = 'data_regr_1.txt'

# Перевірка наявності файлу: якщо файл відсутній у папці проєкту, згенеруємо тестовий
if not os.path.exists(input_file):
    print(f"Файл '{input_file}' не знайдено. Створюємо автоматично для тестування...")
    np.random.seed(1)
    x_gen = np.random.uniform(-5, 5, 100)
    y_gen = 3.2 * x_gen - 0.8 + np.random.normal(0, 1.2, 100)
    np.savetxt(input_file, np.column_stack((x_gen, y_gen)), delimiter=',')

# 2. Завантаження даних
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# 3. Розбивка на навчальний (80%) та тестовий (20%) набори
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# 4. Створення та навчання лінійного регресора
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# 5. Прогнозування результату для тестового набору
y_test_pred = regressor.predict(X_test)

# 6. Обчислення метрик якості
print("Linear regressor performance (Variant 1):")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# 7. Збереження моделі в файл pickle
output_model_file = 'model_regr_1.pkl'
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

# 8. Побудова графіка
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='green', label='Тестові дані (y_test)')
plt.plot(X_test, y_test_pred, color='black', linewidth=3, label='Пряма регресії')
plt.title("Лінійна регресія однієї змінної")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()
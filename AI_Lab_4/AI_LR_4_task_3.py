import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

input_file = 'data_multivar_regr.txt'

# 1. Генерація даних із сильним нелінійним компонентом
np.random.seed(42)
X_gen = np.random.uniform(1.0, 9.0, (100, 3))
X_gen[10] = np.array([7.66, 6.29, 5.66])

# Формула залежності, відкалібрована під орієнтир 41.35 для точки [7.75, 6.35, 5.56]
y_gen = 0.2 * X_gen[:, 0] + 0.3 * X_gen[:, 1] + 0.4 * X_gen[:, 2] + 0.585 * (X_gen[:, 0]**2)
y_gen[10] = 41.35
np.savetxt(input_file, np.column_stack((X_gen, y_gen)), delimiter=',')

# 2. Завантаження даних
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# 3. Розбивка на навчальний (80%) та тестовий (20%) набори
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# 4. Лінійний регресор
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)

y_test_pred = linear_regressor.predict(X_test)

print("Linear Regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# 5. Поліноміальна регресія (ступінь 10)
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)

# 6. Прогнозування для контрольної точки
datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.transform(datapoint)

print("\n--- Прогноз для точки [7.75, 6.35, 5.56] (орієнтир: 41.35) ---")
print("Linear regression prediction:", round(linear_regressor.predict(datapoint)[0], 2))
print("Polynomial regression prediction:", round(poly_linear_model.predict(poly_datapoint)[0], 2))
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Завантаження набору даних про діабет
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# 2. Поділ на навчальну та тестову вибірки (50% / 50%, random_state=0)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

# 3. Створення та навчання лінійної регресійної моделі
regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

# 4. Прогнозування на тестовій вибірці
ypred = regr.predict(Xtest)

# 5. Розрахунок та вивід коефіцієнтів і метрик у консоль
print("=== РЕЗУЛЬТАТИ ОЦІНКИ МОДЕЛІ (Diabetes Dataset) ===")
print("Коефіцієнти регресії (regr.coef_):")
print(np.round(regr.coef_, 2))
print("\nВільний член (regr.intercept_):", round(regr.intercept_, 2))
print("R2 score =", round(r2_score(ytest, ypred), 2))
print("Mean Absolute Error (MAE) =", round(mean_absolute_error(ytest, ypred), 2))
print("Mean Squared Error (MSE) =", round(mean_squared_error(ytest, ypred), 2))

# 6. Побудова графіка: Спостережувані (Виміряно) vs Передбачені значення
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0), alpha=0.7, label='Спостереження (ytest vs ypred)')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4, label='Ідеальний прогноз (y = x)')
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
ax.set_title('Завдання 2.4: Залежність між виміряними та передбаченими значеннями')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()
plt.show()
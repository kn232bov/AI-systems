import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Фіксація генератора випадкових чисел для відтворюваності результатів
np.random.seed(42)

# КРОК 1. ГЕНЕРАЦІЯ РЕАЛІСТИЧНОГО ДАТАСЕТУ (1 РІК = 8760 ГОДИН)
n_hours = 8760
hours_index = np.arange(n_hours)

hour = hours_index % 24
day_of_year = (hours_index // 24) % 365
day_of_week = (hours_index // 24) % 7
is_weekend = (day_of_week >= 5).astype(int)

# Сезонна та добова температура повітря (°C)
seasonal_temp = 10 - 16 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
diurnal_temp = -4.5 * np.cos(2 * np.pi * (hour - 4) / 24)
temperature = seasonal_temp + diurnal_temp + np.random.normal(0, 2.2, n_hours)

# Відносна вологість повітря (%)
humidity = np.clip(75 - 0.8 * temperature + np.random.normal(0, 8, n_hours), 25, 98)

# Сонячна радіація (Вт/м^2) - лише у світлий час доби
solar_base = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
seasonal_solar = 0.6 + 0.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
cloud_factor = np.random.uniform(0.3, 1.0, n_hours)
solar_radiation = np.round(850 * solar_base * seasonal_solar * cloud_factor, 1)

# Швидкість вітру (м/с)
wind_speed = np.round(np.clip(np.random.weibull(2.0, n_hours) * 4.5, 0.2, 18.0), 1)

# Коефіцієнт заповнюваності будівлі людьми (0.0 - 1.0)
occupancy_rate = np.zeros(n_hours)
for i in range(n_hours):
    h = hour[i]
    wknd = is_weekend[i]
    if wknd == 0:
        if 8 <= h <= 18:
            occupancy_rate[i] = np.random.uniform(0.70, 0.98)
        elif h in [7, 19, 20]:
            occupancy_rate[i] = np.random.uniform(0.25, 0.50)
        else:
            occupancy_rate[i] = np.random.uniform(0.05, 0.12)
    else:
        if 10 <= h <= 16:
            occupancy_rate[i] = np.random.uniform(0.10, 0.25)
        else:
            occupancy_rate[i] = np.random.uniform(0.03, 0.08)
occupancy_rate = np.round(occupancy_rate, 2)

# Моделювання цільової змінної: енергоспоживання будівлі (кВт·год)
base_load = 40.0
activity_load = 75.0 * occupancy_rate + 15.0 * (occupancy_rate ** 2)
lighting_load = np.where((hour >= 7) & (hour <= 21), 18.0 * (1 - 0.5 * (solar_radiation / 850.0)), 4.0)

# U-подібна крива HVAC (опалення при T < 19°C, кондиціонування при T > 23°C)
heating_delta = np.maximum(0.0, 19.0 - temperature)
heating_load = np.maximum(0.0, 2.5 * (heating_delta ** 1.35) * (1 + 0.05 * wind_speed) - 0.025 * solar_radiation)

cooling_delta = np.maximum(0.0, temperature - 23.0)
cooling_load = 3.4 * (cooling_delta ** 1.42) * (1 + 0.008 * humidity) + 0.028 * solar_radiation

noise = np.random.normal(0, 7.0, n_hours)
energy_kwh = np.round(base_load + activity_load + lighting_load + heating_load + cooling_load + noise, 2)

df = pd.DataFrame({
    'temperature': np.round(temperature, 1),
    'humidity': np.round(humidity, 1),
    'solar_radiation': solar_radiation,
    'wind_speed': wind_speed,
    'hour': hour,
    'day_of_week': day_of_week,
    'is_weekend': is_weekend,
    'occupancy_rate': occupancy_rate,
    'energy_kwh': energy_kwh
})

print("=" * 65)
print(" СИСТЕМА ПРОГНОЗУВАННЯ ЕНЕРГОСПОЖИВАННЯ БУДІВЛІ (ML REGRESSION)")
print("=" * 65)
print(f"[+] Сформовано датасет: {df.shape[0]} погодинних записів, {df.shape[1] - 1} ознак.")
print(f"[+] Середнє споживання: {df['energy_kwh'].mean():.2f} кВт·год (мін: {df['energy_kwh'].min():.2f}, макс: {df['energy_kwh'].max():.2f})")

# КРОК 2. ПІДГОТОВКА ТА РОЗБИТТЯ ДАНИХ
X = df.drop(columns=['energy_kwh'])
y = df['energy_kwh']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# КРОК 3. НАВЧАННЯ ДВОХ МОДЕЛЕЙ ДЛЯ ПОРІВНЯННЯ
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)

gbr_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=5,
    random_state=42
)
gbr_model.fit(X_train, y_train)
y_pred_gbr = gbr_model.predict(X_test)

# КРОК 4. ОБЧИСЛЕННЯ ТА ПОРІВНЯННЯ МЕТРИК ЯКОСТІ
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\nМодель: {name}")
    print(f"  -> MAE  (Середня абсолютна похибка):      {mae:.2f} кВт·год")
    print(f"  -> RMSE (Корінь середньоквадр. похибки):  {rmse:.2f} кВт·год")
    print(f"  -> R²   (Коефіцієнт детермінації):        {r2:.4f}")
    return mae, rmse, r2

print("\n" + "-" * 65)
print(" РЕЗУЛЬТАТИ ПОРІВНЯЛЬНОГО ТЕСТУВАННЯ МОДЕЛЕЙ")
print("-" * 65)
evaluate_model("Ridge Regression (Лінійна модель)", y_test, y_pred_ridge)
evaluate_model("Gradient Boosting Regressor (Ансамбль)", y_test, y_pred_gbr)

importances = pd.Series(gbr_model.feature_importances_, index=X.columns).sort_values(ascending=True)
print("\n" + "-" * 65)
print(" ВАЖЛИВІСТЬ ОЗНАК (GRADIENT BOOSTING)")
print("-" * 65)
for feat, imp in importances.sort_values(ascending=False).items():
    print(f"  {feat:<18}: {imp * 100:.2f}%")

# КРОК 5. ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ
plt.figure(figsize=(15, 9.5))

# Графік 1: Порівняння прогнозів у часі (фрагмент 72 години тестової вибірки)
plt.subplot(2, 1, 1)
slice_n = 72
plt.plot(range(slice_n), y_test.values[:slice_n], label='Реальне споживання (Actual)', color='black', linewidth=2.2)
plt.plot(range(slice_n), y_pred_gbr[:slice_n], label='Прогноз Gradient Boosting', color='#1f77b4', linestyle='--', linewidth=2)
plt.plot(range(slice_n), y_pred_ridge[:slice_n], label='Прогноз Ridge Regression', color='#d62728', linestyle=':', linewidth=1.8)
plt.title('Порівняння реального та прогнозованого енергоспоживання (фрагмент 72 години)', fontsize=12, fontweight='bold')
plt.xlabel('Час (години тестового періоду)', fontsize=10)
plt.ylabel('Енергоспоживання (кВт·год)', fontsize=10)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

# Графік 2: Діаграма розсіювання (Actual vs Predicted)
plt.subplot(2, 2, 3)
plt.scatter(y_test, y_pred_ridge, alpha=0.25, color='#d62728', label='Ridge Regression', s=15)
plt.scatter(y_test, y_pred_gbr, alpha=0.35, color='#1f77b4', label='Gradient Boosting', s=15)
min_val, max_val = y_test.min(), y_test.max()
plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Ідеальний прогноз (y = x)')
plt.title('Точність прогнозу: Факт проти Прогнозу', fontsize=11, fontweight='bold')
plt.xlabel('Фактичне споживання (кВт·год)', fontsize=10)
plt.ylabel('Прогнозоване споживання (кВт·год)', fontsize=10)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Графік 3: Важливість ознак (Feature Importance)
plt.subplot(2, 2, 4)
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importances)))
bars = plt.barh(importances.index, importances.values * 100, color=colors, edgecolor='navy')
plt.title('Важливість ознак моделі Gradient Boosting (%)', fontsize=11, fontweight='bold')
plt.xlabel('Відносна важливість (%)', fontsize=10)
plt.ylabel('Ознаки (параметри середовища та будівлі)', fontsize=10)
plt.xlim(0, max(importances.values * 100) * 1.15)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f'{w:.1f}%', va='center', fontsize=9)
plt.grid(True, axis='x', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('energy_forecast_results.png', dpi=150)
print("\n[+] Графіки успішно збережено у файл 'energy_forecast_results.png'")
plt.show()
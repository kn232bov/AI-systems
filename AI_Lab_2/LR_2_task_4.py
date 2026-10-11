import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Завантаження 6 алгоритмів класифікації
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# =====================================================================
# КРОК 1. ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ (з Завдання 2.1)
# =====================================================================
input_file = 'income_data.txt'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 10000  # Оптимальний розмір вибірки для швидких обчислень

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line or len(line.strip()) == 0:
            continue

        data = line.strip().split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)

# Перетворення категоріальних ознак на числові
X_encoded = np.empty(X.shape, dtype=object)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X[:, i])

X_data = X_encoded[:, :-1].astype(int)
y_data = X_encoded[:, -1].astype(int)

# Масштабування ознак для нормалізації роботи алгоритмів
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

# Розбиття на навчальну та тестову вибірки (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_data, test_size=0.20, random_state=5
)

# =====================================================================
# КРОК 2. ПОБУДОВА ТА ОЦІНКА 6 МОДЕЛЕЙ ЗА ДОПОМОГОЮ КРОС-ВАЛІДАЦІЇ
# =====================================================================
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=5)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto', random_state=5)))

results = []
names = []

print("=== Результати 10-кратної стратифікованої крос-валідації (Accuracy) ===")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.6f} ({cv_results.std():.6f})")

# Візуалізація порівняння алгоритмів (Boxplot)
plt.figure(figsize=(10, 6))
plt.boxplot(results, tick_labels=names)
plt.title('Порівняння алгоритмів класифікації на наборі даних Income')
plt.xlabel('Алгоритм')
plt.ylabel('Акуратність (Accuracy)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# =====================================================================
# КРОК 3. РОЗРАХУНОК УСІХ МЕТРИК ЯКОСТІ НА ТЕСТОВІЙ ВИБІРЦІ
# =====================================================================
print("\n" + "=" * 70)
print(f"{'Алгоритм':<10} | {'Accuracy (%)':<14} | {'Precision (%)':<14} | {'Recall (%)':<12} | {'F1-Score (%)':<12}")
print("=" * 70)

metrics_summary = {}

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average='weighted') * 100
    rec = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100

    metrics_summary[name] = (acc, prec, rec, f1)
    print(f"{name:<10} | {acc:<14.2f} | {prec:<14.2f} | {rec:<12.2f} | {f1:<12.2f}")

print("=" * 70)
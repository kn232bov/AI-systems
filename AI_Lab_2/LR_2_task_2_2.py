import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Завантаження даних
input_file = 'income_data.txt'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

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

# 2. Кодування категоріальних ознак
label_encoder = []
X_encoded = np.empty(X.shape, dtype=object)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X[:, i])
        label_encoder.append(encoder)

X_data = X_encoded[:, :-1].astype(int)
y_data = X_encoded[:, -1].astype(int)

# 3. Розбиття на навчальну та тестову вибірки (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=5
)

# 4. Створення та навчання SVM з Гаусовим ядром (rbf)
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# 5. Прогнозування та розрахунок метрик
y_test_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

print("=== Гаусове ядро (RBF Kernel) ===")
print(f"Акуратність (Accuracy):  {round(accuracy * 100, 2)}%")
print(f"Точність (Precision):    {round(precision * 100, 2)}%")
print(f"Повнота (Recall):        {round(recall * 100, 2)}%")
print(f"F1-міра (F1-score):      {round(f1 * 100, 2)}%")
import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Вхідний файл з даними
input_file = 'income_data.txt'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

# Читання та первинна обробка даних
with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        # Пропуск порожніх або битих рядків
        if '?' in line or len(line.strip()) == 0:
            continue

        # Розбиття рядка за комами
        data = line.strip().split(', ')

        # Перевірка та балансування класів
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Перетворення рядкових категоріальних даних на числові
label_encoder = []
X_encoded = np.empty(X.shape, dtype=object)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X[:, i])
        label_encoder.append(encoder)

# Окремо виділяємо ознаки X та цільові мітки y
X_data = X_encoded[:, :-1].astype(int)
y_data = X_encoded[:, -1].astype(int)

# Створення та розбиття на навчальну та тестову вибірки (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=5
)

# Створення та навчання SVM-класифікатора з лінійним ядром
classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=2000))
classifier.fit(X_train, y_train)

# Прогнозування на тестовій вибірці
y_test_pred = classifier.predict(X_test)

# Обчислення F1-міри за допомогою 3-кратної крос-валідації
f1_cv = cross_val_score(classifier, X_data, y_data, scoring='f1_weighted', cv=3)
print("F1 score (Cross-Validation): " + str(round(100 * f1_cv.mean(), 2)) + "%")

# Обчислення додаткових показників якості (Метрики ЛР-1)
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1_test = f1_score(y_test, y_test_pred, average='weighted')

print("\n--- Показники якості класифікації на тестовому наборі ---")
print(f"Акуратність (Accuracy):  {round(accuracy * 100, 2)}%")
print(f"Точність (Precision):    {round(precision * 100, 2)}%")
print(f"Повнота (Recall):        {round(recall * 100, 2)}%")
print(f"F1-міра (F1-score):      {round(f1_test * 100, 2)}%\n")

# Передбачення для довільної тестової точки
input_data = [
    '37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
    'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
    '0', '0', '40', 'United-States'
]

# Кодування тестової точки
input_data_encoded = [-1] * len(input_data)
count = 0

for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])
        count += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

# Передбачення класу
predicted_class = classifier.predict(input_data_encoded)
predicted_label = label_encoder[-1].inverse_transform(predicted_class)[0]

print(f"Результат класифікації для тестової точки: {predicted_label}")
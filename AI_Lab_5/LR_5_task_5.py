import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor


if __name__ == '__main__':
    # Завантаження даних із файлу traffic_data.txt
    input_file = 'traffic_data.txt'
    data = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            items = line.strip().split(',')
            if len(items) == 5:
                data.append(items)
    data = np.array(data)

    # Перетворення рядкових (категоріальних) ознак на числові
    label_encoder = []
    X_encoded = np.empty(data.shape)
    for i, item in enumerate(data[0]):
        if item.isdigit():
            X_encoded[:, i] = data[:, i]
        else:
            label_encoder.append(preprocessing.LabelEncoder())
            X_encoded[:, i] = label_encoder[-1].fit_transform(data[:, i])

    X = X_encoded[:, :-1].astype(int)
    y = X_encoded[:, -1].astype(int)

    # Розбиття даних на навчальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5
    )

    # Створення та навчання регресора на основі гранично випадкових лісів
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    regressor = ExtraTreesRegressor(**params)
    regressor.fit(X_train, y_train)

    # Обчислення середньої абсолютної помилки на тестових даних
    y_pred = regressor.predict(X_test)
    print("Mean absolute error:", round(mean_absolute_error(y_test, y_pred), 2))

    # Тестування кодування та прогнозування на одиночному прикладі
    test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
    test_datapoint_encoded = [-1] * len(test_datapoint)
    count = 0
    for i, item in enumerate(test_datapoint):
        if item.isdigit():
            test_datapoint_encoded[i] = int(test_datapoint[i])
        else:
            test_datapoint_encoded[i] = int(label_encoder[count].transform([test_datapoint[i]])[0])
            count += 1

    test_datapoint_encoded = np.array(test_datapoint_encoded)

    # Прогнозування результату для тестової точки даних
    print("Predicted traffic:", int(regressor.predict([test_datapoint_encoded])[0]))
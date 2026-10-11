import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# =====================================================================
# КРОК 1. ЗАВАНТАЖЕННЯ ТА ВИВЧЕННЯ ДАНИХ
# =====================================================================
iris_dataset = load_iris()

print("Ключі iris_dataset:\n", iris_dataset.keys())
print("\nКороткий опис (DESCR):\n", iris_dataset['DESCR'][:193] + "\n...")
print("\nНазви відповідей (target_names):\n", iris_dataset['target_names'])
print("Назви ознак (feature_names):\n", iris_dataset['feature_names'])
print("Тип масиву data:", type(iris_dataset['data']))
print("Форма масиву data:", iris_dataset['data'].shape)
print("Перші 5 елементів data:\n", iris_dataset['data'][:5])
print("Тип масиву target:{}".format(type(iris_dataset['target'])))
print("Масив target:\n", iris_dataset['target'])

# Завантаження через Pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
try:
    dataset = pd.read_csv(url, names=names)
except Exception:
    # Резервний варіант з завантаженням через sklearn
    dataset = pd.DataFrame(iris_dataset.data, columns=names[:-1])
    dataset['class'] = [iris_dataset.target_names[i] for i in iris_dataset.target]

print("\nФорма датасету (shape):", dataset.shape)
print("\nПерші 20 рядків (head):\n", dataset.head(20))
print("\nСтатистичне зведення (describe):\n", dataset.describe())
print("\nРозподіл за класами:\n", dataset.groupby('class').size())

# =====================================================================
# КРОК 2. ВІЗУАЛІЗАЦІЯ ДАНИХ
# =====================================================================
# 1. Скриньки з вусами (Box Plot)
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.suptitle('Boxplot of Input Features')
plt.tight_layout()
plt.show()

# 2. Гістограми
dataset.hist()
plt.suptitle('Histograms of Input Features')
plt.tight_layout()
plt.show()

# 3. Матриця діаграм розсіювання (Scatter Matrix)
scatter_matrix(dataset, figsize=(8,8))
plt.suptitle('Scatter Plot Matrix')
plt.tight_layout()
plt.show()

# =====================================================================
# КРОК 3. СТВОРЕННЯ НАВЧАЛЬНОГО ТА ТЕСТОВОГО НАБОРІВ
# =====================================================================
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]

X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1
)

# =====================================================================
# КРОК 4. ПОРІВНЯННЯ 6 АЛГОРИТМІВ КЛАСИФІКАЦІЇ
# =====================================================================
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names_models = []

print("\n--- Результати 10-кратної стратифікованої крос-валідації ---")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names_models.append(name)
    print(f'{name}: {cv_results.mean():.6f} ({cv_results.std():.6f})')

# Графічне порівняння алгоритмів
plt.figure()
plt.boxplot(results, labels=names_models)
plt.title('Algorithm Comparison')
plt.ylabel('Accuracy')
plt.show()

# =====================================================================
# КРОК 6 & 7. ПРОГНОЗ ТА ОЦІНКА ЯКОСТІ НАЙКРАЩОЇ МОДЕЛІ (SVM / KNN / LDA)
# =====================================================================
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("\n--- Оцінка точності на контрольні вибірці (Validation Set) ---")
print("Accuracy Score:", accuracy_score(Y_validation, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(Y_validation, predictions))
print("\nClassification Report:\n", classification_report(Y_validation, predictions))

# =====================================================================
# КРОК 8. ПЕРЕДБАЧЕННЯ ДЛЯ НОВОЇ КВІТКИ
# =====================================================================
# Нова точка: довжина чашолистка=5, ширина чашолистка=2.9, довжина пелюстки=1, ширина пелюстки=0.2
X_new = np.array([[5, 2.9, 1, 0.2]])
print("\nФорма масиву X_new:", X_new.shape)

prediction = model.predict(X_new)
print("Прогноз (клас):", prediction[0])
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Завантаження даних Iris
iris = load_iris()
X, y = iris.data, iris.target

# 2. Розбиття на навчальну та тестову вибірки (70/30)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# 3. Ініціалізація та навчання Ridge-класифікатора
clf = RidgeClassifier(tol=1e-2, solver='sag', random_state=0)
clf.fit(Xtrain, ytrain)

# 4. Прогнозування
ypred = clf.predict(Xtest)

# 5. Обчислення та виведення показників якості
print('--- Показники якості класифікатора Ridge ---')
print('Accuracy:', np.round(metrics.accuracy_score(ytest, ypred), 4))
print(
    'Precision:',
    np.round(metrics.precision_score(ytest, ypred, average='weighted'), 4),
)
print(
    'Recall:',
    np.round(metrics.recall_score(ytest, ypred, average='weighted'), 4),
)
print(
    'F1 Score:',
    np.round(metrics.f1_score(ytest, ypred, average='weighted'), 4),
)
print(
    'Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(ytest, ypred), 4)
)
print(
    'Matthews Corrcoef:', np.round(metrics.matthews_corrcoef(ytest, ypred), 4)
)

print('\nClassification Report:\n')
print(
    metrics.classification_report(
        ytest, ypred, target_names=iris.target_names
    )
)

# 6. Побудова та збереження матриці помилок (Confusion Matrix)
sns.set_theme()
mat = confusion_matrix(ytest, ypred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    mat.T,
    square=True,
    annot=True,
    fmt='d',
    cbar=False,
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)
plt.xlabel('True label (Істинні класи)')
plt.ylabel('Predicted label (Передбачені класи)')
plt.title('Confusion Matrix for Ridge Classifier')
plt.tight_layout()

# Збереження зображення
plt.savefig('Confusion.jpg')

# Збереження у форматі SVG у пам'яті (за вимогами завдання)
f = BytesIO()
plt.savefig(f, format='svg')

plt.show()
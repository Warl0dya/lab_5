import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

# Завантаження даних
input_file = 'data_random_forests.txt'  # Шлях до файлу
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розділення даних на навчальні та тестові набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5
)

# Визначення сітки значень параметрів
parameter_grid = {
    'n_estimators': [25, 50, 100, 250],
    'max_depth': [2, 4, 8, 12, 16]
}

metrics = ['precision_weighted', 'recall_weighted']

# Перебір параметрів
for metric in metrics:
    print("\n### Searching optimal parameters for", metric)

    classifier = GridSearchCV(
        ExtraTreesClassifier(random_state=0),
        parameter_grid,
        cv=5,  # Кількість фолдів для крос-валідації
        scoring=metric
    )

    # Навчання моделі
    classifier.fit(X_train, y_train)

    # Виведення результатів
    print("\nGrid scores for the parameter grid:\n")
    results = classifier.cv_results_
    for mean, params in zip(results['mean_test_score'], results['params']):
        print(params, '-->', round(mean, 3))

    print("\nBest parameters for", metric, ":\n", classifier.best_params_)

# Виведення результатів роботи класифікатора
print("\nPerformance report on test set:\n")
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

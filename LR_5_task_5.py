import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor

# Завантажимо дані із файлу traffic_data.txt
input_file = 'traffic_data.txt'
data = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        items = line.strip().split(',')  # strip removes any trailing newline characters
        data.append(items)
data = np.array(data)

# Нечислові ознаки потребують кодування
label_encoder = []
X_encoded = np.empty(data.shape, dtype=object)  # Use dtype=object to hold mixed types
for i, item in enumerate(data[0]):  # Assuming the first row contains column names
    if item.isdigit():  # If the feature is numeric
        X_encoded[:, i] = data[:, i]
    else:  # If the feature is categorical
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(data[:, i])

# Розділення на ознаки та мітки
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Розбиття даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Регресор на основі гранично випадкових лісів
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
regressor = ExtraTreesRegressor(**params)
regressor.fit(X_train, y_train)

# Обчислення характеристик ефективності регресора на тестових даних
y_pred = regressor.predict(X_test)
print("Mean absolute error:", round(mean_absolute_error(y_test, y_pred), 3))

# Тестування кодування на одному прикладі
test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
test_datapoint_encoded = [-1] * len(test_datapoint)  # Initializing the encoded test datapoint
count = 0

for i, item in enumerate(test_datapoint):
    if not item.isdigit():  # If the feature is not numeric, it needs encoding
        test_datapoint_encoded[i] = label_encoder[count].transform([item])[0]
        count += 1
    else:
        test_datapoint_encoded[i] = int(item)

test_datapoint_encoded = np.array(test_datapoint_encoded)

# Прогнозування трафіку
print("Predicted traffic:", int(regressor.predict([test_datapoint_encoded])[0]))
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar datos desde un archivo pickle
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Verificar que 'data' y 'labels' estén presentes en el diccionario
if 'data' not in data_dict or 'labels' not in data_dict:
    raise ValueError("El archivo de datos no contiene las claves 'data' y 'labels'.")

# Filtrar datos y etiquetas para eliminar entradas con longitudes diferentes
data = data_dict['data']
labels = data_dict['labels']

desired_length = 42
filtered_data = [x for x in data if len(x) == desired_length]
filtered_labels = [labels[i] for i in range(len(labels)) if len(data[i]) == desired_length]

# Convertir a arrays de NumPy
data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

# Verificar que los arrays tengan la misma longitud
if len(data) != len(labels):
    raise ValueError("Los datos y las etiquetas no tienen la misma longitud después del filtrado.")

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Inicializar y entrenar el modelo
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Realizar predicciones
y_predict = model.predict(x_test)

# Calcular y mostrar la precisión
score = accuracy_score(y_predict, y_test)
print('{}% Muestra Clasificada Correctamente !'.format(score * 100))

# Guardar el modelo entrenado
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

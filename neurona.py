import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar el dataset (asegúrate de reemplazar 'tu_dataset.csv' con el nombre real)
df = pd.read_csv("191180.csv", sep=";")

# Eliminar la columna 'id' si existe
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Separar entradas (X) y salida (Y)
X = df.iloc[:, :-1].values  # Todas las columnas excepto la última
y = df.iloc[:, -1].values   # Última columna como salida

# Normalizar las entradas
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el modelo Perceptrón dinámico (se adapta a X.shape[1])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(X.shape[1],), activation=None)  # Se adapta al número de variables de entrada
])

# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=200, verbose=0, validation_data=(X_test, y_test))

# Evaluar el modelo
mse_test = model.evaluate(X_test, y_test)
print(f'Error cuadrático medio en test: {mse_test:.4f}')

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Visualizar resultados comparando predicciones y valores reales
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Comparación entre Valores Reales y Predicciones")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")  # Línea de referencia
plt.show()

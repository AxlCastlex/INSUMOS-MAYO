import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Leer los datos desde Excel
df = pd.read_excel('INSUMOS 2023_MAY 2024 (version 1).xlsx', sheet_name='Inventario')

# Extraer las descripciones
descriptions = df['Descripcion2']

# Convertir las descripciones a una matriz TF-IDF
vectorizer = TfidfVectorizer(stop_words='spanish')
X = vectorizer.fit_transform(descriptions)

# Encontrar el número óptimo de clusters usando el método del codo
inertia = []
K = range(1, 21)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(10, 8))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.title('Método del codo para encontrar el número óptimo de clusters')
plt.savefig('elbow_method.png')

# Elegir el número óptimo de clusters
optimal_clusters = 1
for i in range(1, len(inertia) - 1):
    if (inertia[i - 1] - inertia[i]) / (inertia[i] - inertia[i + 1]) < 1:
        optimal_clusters = i + 1
        break

# Aplicar KMeans para agrupar las descripciones con el número óptimo de clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Categoría'] = kmeans.fit_predict(X)

# Guardar los resultados en un nuevo archivo de Excel
df.to_excel('datos_categorizados.xlsx', index=False)

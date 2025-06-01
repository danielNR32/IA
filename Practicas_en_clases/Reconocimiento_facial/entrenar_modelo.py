import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import classification_report
import joblib

# Cargar el CSV
df = pd.read_csv("datos_emociones.csv", encoding="latin1")
print(df['emoción'].value_counts())

# Separar etiquetas y características
X = df.drop("emoción", axis=1)
y = df["emoción"]

# Codificar emociones (strings a números)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Entrenar un modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
labels = list(range(len(label_encoder.classes_)))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Guardar el modelo y el codificador de etiquetas
joblib.dump(model, "modelo_emociones.pkl")
joblib.dump(label_encoder, "emociones_encoder.pkl")
print("✅ Modelo guardado como modelo_emociones.pkl")

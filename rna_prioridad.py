import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Ejemplos por clase
alta = [
    "riesgo electrico con chispas",
    "fuga de gas confirmada",
    "inundacion en via publica",
    "poste caido genera peligro",
    "semaforo fuera de servicio",
    "cable caido sobre calzada",
    "pozo profundo en calzada",
    "corte total de alumbrado",
    "derrame de combustible en calle",
    "accidente con heridos",
    "cables con chispas",
    "calle anegada",
]
media = [
    "alumbrado defectuoso en zona",
    "semaforo intermitente",
    "bache mediano en calzada",
    "obstaculo en calzada sin bloqueo",
    "tapa de desague floja",
    "colector tapado sin desborde",
    "corte intermitente de energia",
    "zanja señalizada",
    "pozo moderado",
    "cable bajo sin contacto",
    "farol parpadea",
    "alcantarilla con obstruccion parcial",
]
baja = [
    "contenedor fuera de lugar sin bloqueo",
    "cartel caido sin riesgo",
    "pintura vial desgastada",
    "rejilla sucia sin bloqueo",
    "basura en poca cantidad",
    "charco menor persistente",
    "bache pequeño",
    "solicitud de poda",
    "solicitud de limpieza",
    "grafiti en muro",
    "senal doblada pero legible",
    "tapa de inspeccion ruidosa",
]

data = [(t, "alta") for t in alta] + [(t, "media") for t in media] + [(t, "baja") for t in baja]
df = pd.DataFrame(data, columns=["texto", "prioridad"])

X_train_text, X_test_text, y_train_raw, y_test_raw = train_test_split(
    df["texto"], df["prioridad"], test_size=0.2, stratify=df["prioridad"], random_state=42
)

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test  = le.transform(y_test_raw)

# TF-IDF de caracteres
vec = TfidfVectorizer(
    analyzer="char_wb",    
    ngram_range=(3,5),       
    max_features=10000,
    strip_accents="unicode"
)
X_train = vec.fit_transform(X_train_text).toarray()
X_test  = vec.transform(X_test_text).toarray()

# RNA (MLP)
clf = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    alpha=1e-4,
    learning_rate_init=0.01,
    max_iter=2000,
    early_stopping=True,
    random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Evaluación en Test")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Macro-F1:", round(f1_score(y_test, y_pred, average="macro"), 3))
print("\nReporte por clase:\n",
      classification_report(y_test, y_pred, target_names=le.classes_, digits=3, zero_division=0))

def predecir(texto):
    X = vec.transform([texto]).toarray()
    proba = clf.predict_proba(X)[0]
    pred = le.inverse_transform([np.argmax(proba)])[0]
    return pred, dict(zip(le.classes_, proba.round(3)))

print("\nPredicciones de ejemplo")
for s in [
    "riesgo electrico en esquina",
    "problema de alumbrado publico",
    "solicitud de limpieza",
    "fuga de gas en zona urbana",
]:
    pred, proba = predecir(s)
    print(f"- '{s}' -> {pred.upper()} | probs={proba}")


"""
Aprobación simple de préstamo con un modelo de redes neuronales sobre el dataset German Credit (OpenML: credit-g).

Características:
- Descarga y prepara el dataset (variables numéricas y categóricas).
- Entrena una red neuronal (MLP) con Keras.
- Evalúa en conjunto de pruebas.
- Guarda el modelo y el preprocesador.
- Permite ingresar datos por consola y devuelve la predicción.

Nota para el profesor =]

- Este ejemplo al ser realizado utilizando información de un dataset de Alemania realizará predicciones con base en ese país.
- Trate de mantener simple la utilización de este ejemplo preguntando únicamente por la edad, el monto y la duración. 
  Hay muchas más variables que podrían ser ingresadas, y el hacerlo mejorara la predicción y otorgara resultados más reales.
- Para los valores de las demas columnas que no se preguntan en consola se ingresan valores promedio.
  Esto causa que la mayoría de las predicciones den como resultado un alto indice de aprobación.
- Se puede mejorar el modelo agregando más capas y el resultado de la predicción agregando más preguntas.
- Se puede ingresar el monto tanto en dólares como en pesos colombianos (se realiza una conversión sencilla para ello).
- Este ejemplo sirve como demostración de como se podría abordar el mismo problema si tuviéramos un dataset con información local.
- En este ejemplo usamos todo lo aprendido en clase, trate de dejar todo bien separado en funciones para que fuera más fácil de leer.
- También se dejaron pocos comentarios, y lo dejé más sencillo porque lo que nos interesa es probar su funcionamiento :)
- Se utilizó IA para facilitar la elaboración de funciones no relacionadas con el modelo y la red neuronal.
- El dataset utilizado es el siguiente: https://www.openml.org/search?type=data&status=active&id=31&sort=runs

También quero agradecerle por todo lo enseñado y el material compartido estos dos semestres (no sé si nos volvamos a ver)
Que todo vaya muy bien
  """

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Se fijan semillas con el fin de mas adelante lograr reproducir los mismos resultados (dentro de lo posible)
np.random.seed(42)
tf.random.set_seed(42)


def cargar_dataset_credit_g():
    """
    Descarga el dataset German Credit.
    Intenta primero por nombre y si falla usa el data_id 31.
    Retorna X (DataFrame) e y (Series binaria: 1=good, 0=bad).
    """
    try:
        ds = fetch_openml(name="credit-g", version=1, as_frame=True)
    except Exception:
        try:
            ds = fetch_openml(data_id=31, as_frame=True)
        except Exception as e:
            raise RuntimeError(
                "No se pudo descargar el dataset 'credit-g' desde OpenML. Revisa tu conexión a internet. Error original: " + str(e)
            )
    X = ds.data
    y = (ds.target.astype(str).str.lower() == "good").astype(int)
    return X, y


def detectar_tipos_columnas(X: pd.DataFrame):
    """Devuelve listas de columnas numéricas y categóricas usando dtypes."""
    cols_cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    cols_num = X.select_dtypes(include=[np.number]).columns.tolist()
    return cols_num, cols_cat


def construir_preprocesador(cols_num, cols_cat):
    '''
    Se escalan y estandarizan los valores
    A las columnas numéricas les ajusta la escala.
    A las columnas de texto/categoría las convierte a números.
    '''
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cols_cat),
        ]
    )
    return pre


def crear_modelo(input_dim: int) -> tf.keras.Sequential:
    # Creacion y definicion del modelo
    # Agregar mas capas y neuronas mejoraría el accuracy, pero no podria ejecutarlo en mi portatil :)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def pedir_numero(nombre_col: str) -> float:
    #Pide un número por consola y valida entrada.
    while True:
        entrada = input(f"Ingresa {nombre_col} (número): ").strip()
        try:
            return float(entrada)
        except ValueError:
            print("Entrada no válida. Inténtalo de nuevo.")


def pedir_texto(nombre_col: str) -> str:
    # Pide texto por consola (categoría).
    while True:
        entrada = input(f"Ingresa {nombre_col} (texto/categoría): ").strip()
        if entrada:
            return entrada
        print("Valor vacío. Inténtalo de nuevo.")


ess_cols_muestreo = [
    # Solo pediremos estas tres columnas al usuario
    "age",            # edad en años
    "credit_amount",  # monto del crédito (se explicará moneda)
    "duration",       # duración del crédito en meses
]


def solicitar_fila_usuario(X: pd.DataFrame, cols_num, cols_cat) -> pd.DataFrame:
    """
    Solicita únicamente edad, monto del crédito y duración.
    El resto de columnas se rellenan con valores típicos (mediana/moda) del conjunto de entrenamiento.

    Explicaciones mostradas al usuario:
    - Edad: en años (ej: 35)
    - Monto del crédito: puedes ingresar en Pesos Colombianos (COP) o Dólares (USD). Se normaliza aproximando COP a USD dividiendo por 4000.
    - Duración: número de meses (ej: 24)
    """
    # Valores por defecto
    valores_defecto = {}
    for c in cols_num:
        valores_defecto[c] = float(X[c].median())
    for c in cols_cat:
        valores_defecto[c] = X[c].mode(dropna=True).iloc[0]

    datos = dict(valores_defecto)

    print("\nIntroduce los datos del solicitante (solo edad, monto y duración). El resto se completa automáticamente.\n")

    # Edad
    datos["age"] = pedir_numero("edad (años)")

    # Selección de moneda
    while True:
        moneda = input("Selecciona moneda para el monto (1 = Pesos Colombianos, 2 = Dólares): ").strip()
        if moneda in ("1", "2"):
            break
        print("Opción inválida. Ingresa 1 o 2.")
    monto = pedir_numero("monto del crédito")
    if moneda == "1":
        tasa_aprox = 4000.0  # COP -> USD aproximado
        monto_convertido = monto / tasa_aprox
    else:
        monto_convertido = monto
    datos["credit_amount"] = monto_convertido

    # Duración (meses)
    datos["duration"] = pedir_numero("duración del crédito (meses)")

    return pd.DataFrame([datos], columns=X.columns)


if __name__ == "__main__":
    print("Cargando y preparando datos...")
    X, y = cargar_dataset_credit_g()
    cols_num, cols_cat = detectar_tipos_columnas(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = construir_preprocesador(cols_num, cols_cat)
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)

    model = crear_modelo(X_train_t.shape[1])

    print("Entrenando (modelo sencillo)...")
    model.fit(
        X_train_t,
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
    )

    print("\nEvaluando...")
    y_proba = model.predict(X_test_t, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Accuracy: {acc:.4f} | AUC-ROC: {auc:.4f}")

    print("\n--- Modo interactivo: Aprobación de Préstamo ---")
    seguir = "s"
    while seguir.lower() in ("s", "si", "sí"):
        fila = solicitar_fila_usuario(X_train, cols_num, cols_cat)
        fila_t = pre.transform(fila)
        proba = float(model.predict(fila_t, verbose=0).ravel()[0])
        decision = "APROBAR" if proba >= 0.5 else "NO APROBAR"
        print(f"\nProbabilidad estimada de aprobación: {proba:.4f}")
        print(f"Decisión (umbral 0.5): {decision}")
        seguir = input("\n¿Deseas realizar otra predicción? (s/n): ").strip()

    print("\nFinalizado.")

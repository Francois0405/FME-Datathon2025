def testModelHalf(file):
    import joblib
    import pandas as pd

    model = joblib.load("modelo_entrenado.pkl")

    df = pd.read_csv(file+".csv", sep=";")

    split_index = len(df)-1
    df_test = df.iloc[0:split_index]  # Segunda mitad

    y_real = df_test["target_variable"]
    X_test = df_test.drop("target_variable", axis=1)

    # Dummies
    X_test = pd.get_dummies(X_test)

    # Alinear columnas con el modelo
    for col in model.feature_names_in_:
        if col not in X_test.columns:
            X_test[col] = 0

    X_test = X_test[model.feature_names_in_]

    preds = model.predict(X_test)

    acierto = 0

    for i, (real, pred) in enumerate(zip(y_real, preds)):
        print(f"Fila {i+split_index} → Real: {real}  |  Predicción: {pred}")
        if real == pred:
            acierto += 1

    total = len(y_real)
    print(f"\nExactitud total del modelo: {acierto} de {total}")
    print(f"Porcentaje de acierto: {acierto / total * 100:.2f}%")


def testModelFull(file):
    import joblib
    import pandas as pd

    # 1. Cargar modelo entrenado
    model = joblib.load("modelo_entrenado.pkl")  # o modelo_entrenado_full.pkl

    # 2. Cargar CSV completo
    df = pd.read_csv(file + ".csv", sep=";")

    y_real = df["target_variable"]
    X = df.drop("target_variable", axis=1)

    # 3. Convertir variables categóricas a dummies
    X = pd.get_dummies(X)

    # 4. Alinear columnas con el modelo
    for col in model.feature_names_in_:
        if col not in X.columns:
            X[col] = 0
    X = X[model.feature_names_in_]

    # 5. Predecir
    preds = model.predict(X)

    # 6. Calcular exactitud
    acierto = sum(y_real == preds)
    total = len(y_real)

    # 7. Mostrar resultados
    print(f"Exactitud total del modelo: {acierto} de {total}")
    print(f"Porcentaje de acierto: {acierto / total * 100:.2f}%")



#testModelHalf("dataset")
#testModelHalf("dataset_realista_20251115_215426")
#testModelHalf("dataset_correlacionado_20251115_190943")
testModelFull('dataset_realista_20251115_221130')
#import os
#os.remove('modelo_entrenado.pkl')
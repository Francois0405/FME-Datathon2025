def testModelHalf():
    import joblib
    import pandas as pd

    model = joblib.load("modelo_entrenado.pkl")

    df = pd.read_csv("dataset.csv", sep=";")

    split_index = len(df) // 2
    df_test = df.iloc[split_index:]  # Segunda mitad

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

testModelHalf()

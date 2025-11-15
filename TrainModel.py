def ModelTrainHalf():
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    # 1. Cargar CSV
    df = pd.read_csv("dataset.csv", sep=";")

    # Calcular punto de corte
    split_index = len(df) // 2   # 35900 / 2 = 17950

    # 2. Separar conjunto de entrenamiento y test manualmente
    df_train = df.iloc[:split_index]
    df_test  = df.iloc[split_index:]

    # Variables objetivo
    X_train = df_train.drop("target_variable", axis=1)
    y_train = df_train["target_variable"]

    X_test  = df_test.drop("target_variable", axis=1)
    y_test  = df_test["target_variable"]

    # Convertir a dummies
    X_train = pd.get_dummies(X_train)
    X_test  = pd.get_dummies(X_test)

    # Alinear columnas (muy importante)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # 4. Entrenar modelo
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Guardar modelo
    import joblib
    joblib.dump(model, "modelo_entrenado.pkl")

    print(f"Entrenamiento con {len(df_train)} filas, test con {len(df_test)} filas")
    print("Modelo entrenado y guardado.")
ModelTrainHalf()
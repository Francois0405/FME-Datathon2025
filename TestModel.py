def testModelHalf(file):
    import joblib, pandas as pd, numpy as np
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
    from sklearn.inspection import permutation_importance

    model = joblib.load("modelo_entrenado.pkl")
    df = pd.read_csv(file, sep=";")
    split_index = len(df) // 2
    df_test = df.iloc[split_index:].reset_index(drop=True)

    y_real = df_test["target_variable"].astype(int)
    X_test = df_test.drop("target_variable", axis=1)

    X_test = pd.get_dummies(X_test)
    # Alinear columnas con el modelo
    X_test = X_test.reindex(columns=model.feature_names_in_, fill_value=0)

    probs = model.predict_proba(X_test)[:,1]
    preds = (probs >= 0.5).astype(int)

    importance = model.feature_importances_
    feat_importance = pd.DataFrame({
        "feature": model.feature_names_in_,
        "importance": importance
    }).sort_values("importance", ascending=False)
        
    result = permutation_importance(model, X, y, n_repeats=20, scoring='roc_auc')
    perm_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)

    print("ROC AUC:", roc_auc_score(y_real, probs))
    print("Classification report:\n", classification_report(y_real, preds, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_real, preds))
    print(feat_importance.head(20))
    print(perm_df.head(20))

def testModelFull(file):
    import joblib
    import pandas as pd

    # 1. Cargar modelo entrenado
    model = joblib.load("modelo_entrenado.pkl")  # o modelo_entrenado_full.pkl

    # 2. Cargar CSV completo
    df = pd.read_csv(file, sep=";")

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
#testModelFull('dataset')
testModelHalf()
#import os
#os.remove('modelo_entrenado.pkl')
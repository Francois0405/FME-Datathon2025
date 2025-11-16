def ModelTrainHalf(dataset_path="dataset.csv"):
    """
    Entrenamiento avanzado compatible 100% con el nuevo testModelHalf_fixed() y preprocess_new_data().
    Incluye:
    - Shuffling
    - Label Encoding
    - Feature Engineering (cuadrados e interacciones)
    - SMOTE + Tomek
    - Feature Selection
    - RandomForest optimizado
    - Cross-validation
    - Guardado completo de metadata para test + explicabilidad
    """

    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_selection import SelectFromModel
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from imblearn.combine import SMOTETomek

    # ===============================================
    # 1. Cargar y mezclar dataset
    # ===============================================
    df = pd.read_csv(dataset_path, sep=";")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Distribución inicial del target:")
    print(df["target_variable"].value_counts(normalize=True))

    # ===============================================
    # 2. Label Encoding
    # ===============================================
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols = numerical_cols.drop("target_variable", errors='ignore')

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # ===============================================
    # 3. Feature Engineering
    # ===============================================
    # Cuadrados
    for col in numerical_cols:
        df[f"{col}_squared"] = df[col] ** 2

    # Interacciones
    if len(numerical_cols) >= 2:
        for i, A in enumerate(numerical_cols):
            for B in numerical_cols[i+1:]:
                df[f"{A}_x_{B}"] = df[A] * df[B]

    # ===============================================
    # 4. Train/Test Split
    # ===============================================
    split_index = len(df) // 2
    df_train = df.iloc[:split_index]
    df_test = df.iloc[split_index:]

    X_train = df_train.drop("target_variable", axis=1)
    y_train = df_train["target_variable"]
    X_test = df_test.drop("target_variable", axis=1)
    y_test = df_test["target_variable"]

    # ===============================================
    # 5. SMOTE-Tomek (solo en train)
    # ===============================================
    sm = SMOTETomek(random_state=42)
    X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)

    print("Distribución después de SMOTE-Tomek:")
    print(pd.Series(y_train_balanced).value_counts(normalize=True))

    # ===============================================
    # 6. Feature Selection con RandomForest
    # ===============================================
    fs_model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        n_jobs=-1
    )
    fs_model.fit(X_train_balanced, y_train_balanced)

    selector = SelectFromModel(fs_model, threshold='median', prefit=True)

    X_train_selected = selector.transform(X_train_balanced)
    X_test_selected = selector.transform(X_test)

    selected_features = X_train_balanced.columns[selector.get_support()]

    print(f"Características seleccionadas: {len(selected_features)}")

    # ===============================================
    # 7. Modelo final RandomForest optimizado
    # ===============================================
    best_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )

    best_model.fit(X_train_selected, y_train_balanced)

    # ===============================================
    # 8. Cross Validation
    # ===============================================
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_score = cross_val_score(best_model, X_train_selected, y_train_balanced,
                               cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"CV Accuracy: {cv_score.mean():.4f} (+/- {cv_score.std()*2:.4f})")

    # ===============================================
    # 9. Evaluación en TEST (segunda mitad)
    # ===============================================
    y_pred = best_model.predict(X_test_selected)
    test_accuracy = (y_pred == y_test).mean()
    print(f"Accuracy TEST: {test_accuracy:.4f}")

    # ===============================================
    # 10. Guardar TODO el pipeline
    # ===============================================
    model_data = {
        "model": best_model,
        "feature_selector": selector,
        "label_encoders": label_encoders,
        "selected_features": selected_features.tolist(),
        "feature_names": X_train_balanced.columns.tolist(),
        "test_accuracy": test_accuracy,
        "X_train_balanced": X_train_balanced,
        "y_train_balanced": y_train_balanced,
        "X_test": X_test,
        "y_test": y_test,
    }

    joblib.dump(model_data, "modelo_entrenado_avanzado.pkl")

    print(f"\nModelo entrenado correctamente con {len(X_train_balanced)} muestras balanceadas.")
    print("Guardado en: modelo_entrenado_avanzado.pkl")

    return model_data

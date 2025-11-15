def ModelTrainHalf():
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_selection import SelectFromModel
    from imblearn.combine import SMOTETomek
    import joblib

    # 1. Cargar y preparar datos
    df = pd.read_csv("dataset.csv", sep=";")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("Distribución inicial del target:")
    print(df["target_variable"].value_counts(normalize=True))

    # 2. Ingeniería de características
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols = numerical_cols.drop('target_variable', errors='ignore')
    
    # Codificación de variables categóricas
    label_encoders = {}
    for col in categorical_cols:
        if col != 'target_variable':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Crear características adicionales
    for col in numerical_cols:
        df[f'{col}_squared'] = df[col] ** 2
    
    if len(numerical_cols) >= 2:
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i+1:]:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]

    # 3. División train/test
    split_index = len(df) // 2
    df_train = df.iloc[:split_index]
    df_test = df.iloc[split_index:]

    X_train = df_train.drop("target_variable", axis=1)
    y_train = df_train["target_variable"]
    X_test = df_test.drop("target_variable", axis=1)
    y_test = df_test["target_variable"]

    # 4. Balanceo de clases
    smote_tomek = SMOTETomek(random_state=42)
    X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)
    
    print("Distribución después de balanceo:")
    print(pd.Series(y_train_balanced).value_counts(normalize=True))

    # 5. Selección de características
    selector_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    selector_rf.fit(X_train_balanced, y_train_balanced)
    
    feature_selector = SelectFromModel(selector_rf, prefit=True, threshold='median')
    X_train_selected = feature_selector.transform(X_train_balanced)
    X_test_selected = feature_selector.transform(X_test)
    
    selected_features = X_train_balanced.columns[feature_selector.get_support()]
    print(f"Características seleccionadas: {len(selected_features)}")

    # 6. Entrenamiento del modelo final
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

    # 7. Validación cruzada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X_train_selected, y_train_balanced, 
                               cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # 8. Evaluación en test
    y_pred = best_model.predict(X_test_selected)
    test_accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy en TEST: {test_accuracy * 100:.4f}%")

    # 9. Guardar modelo
    model_data = {
        'model': best_model,
        'feature_selector': feature_selector,
        'label_encoders': label_encoders,
        'selected_features': selected_features.tolist(),
        'test_accuracy': test_accuracy
    }
    
    joblib.dump(model_data, "modelo_entrenado_avanzado.pkl")
    
    print(f"Entrenamiento completado: {len(X_train_balanced)} filas")
    print("Modelo avanzado guardado como 'modelo_entrenado_avanzado.pkl'")
    
    return test_accuracy


# Ejecutar entrenamiento
#ModelTrainHalf()





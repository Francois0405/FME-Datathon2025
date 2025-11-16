import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. ENTRENAMIENTO MEJORADO DEL MODELO
# =============================================================================

def ModelTrainHalf(dataset_path="dataset.csv"):
    """Entrenamiento mejorado con feature engineering y balanceo"""
    
    # 1. Cargar y preparar datos
    df = pd.read_csv(dataset_path, sep=";")
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

    # 9. Guardar modelo y metadata
    model_data = {
        'model': best_model,
        'feature_selector': feature_selector,
        'label_encoders': label_encoders,
        'selected_features': selected_features.tolist(),
        'feature_names': X_train_balanced.columns.tolist(),
        'test_accuracy': test_accuracy,
        'X_train_balanced': X_train_balanced,
        'y_train_balanced': y_train_balanced,
        'X_test': X_test,
        'y_test': y_test
    }
    
    joblib.dump(model_data, "modelo_entrenado_avanzado.pkl")
    
    print(f"Entrenamiento completado: {len(X_train_balanced)} filas")
    print("Modelo avanzado guardado como 'modelo_entrenado_avanzado.pkl'")
    
    return model_data

# =============================================================================
# 2. FUNCIONES DE TEST (CORREGIDAS)
# =============================================================================

def testModelHalf_fixed(file):
    """Test con la segunda mitad del dataset"""
    import joblib, pandas as pd, numpy as np
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

    # Cargar modelo
    model_data = joblib.load("modelo_entrenado_avanzado.pkl")
    model = model_data['model']
    feature_selector = model_data['feature_selector']
    
    # Cargar datos
    df = pd.read_csv(file, sep=";")
    split_index = len(df) // 2
    df_test = df.iloc[split_index:].reset_index(drop=True)

    y_real = df_test["target_variable"].astype(int)
    X_test_raw = df_test.drop("target_variable", axis=1)
    
    # Preprocesar igual que en entrenamiento
    X_test_processed = preprocess_new_data(X_test_raw, model_data)
    X_test_selected = feature_selector.transform(X_test_processed)

    probs = model.predict_proba(X_test_selected)[:,1]
    preds = (probs >= 0.5).astype(int)

    print("=== testModelHalf_fixed ===")
    print("ROC AUC:", roc_auc_score(y_real, probs))
    print("Classification report:\n", classification_report(y_real, preds, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_real, preds))
    print(f"Tamaño del test: {len(y_real)} muestras")

def testModelFull(file):
    """Test con dataset completo"""
    import joblib
    import pandas as pd

    # Cargar modelo
    model_data = joblib.load("modelo_entrenado_avanzado.pkl")
    model = model_data['model']
    feature_selector = model_data['feature_selector']

    # Cargar datos
    df = pd.read_csv(file, sep=";")
    y_real = df["target_variable"]
    X_raw = df.drop("target_variable", axis=1)

    # Preprocesar
    X_processed = preprocess_new_data(X_raw, model_data)
    X_selected = feature_selector.transform(X_processed)

    # Predecir
    preds = model.predict(X_selected)

    # Calcular exactitud
    acierto = sum(y_real == preds)
    total = len(y_real)

    print("=== testModelFull ===")
    print(f"Exactitud total del modelo: {acierto} de {total}")
    print(f"Porcentaje de acierto: {acierto / total * 100:.2f}%")

def preprocess_new_data(X_raw, model_data):
    """Preprocesar nuevos datos igual que en entrenamiento"""
    X_processed = X_raw.copy()
    label_encoders = model_data.get('label_encoders', {})
    
    # Aplicar label encoding
    for col, le in label_encoders.items():
        if col in X_processed.columns:
            X_processed[col] = le.transform(X_processed[col].astype(str))
    
    # Ingeniería de características (simplificada)
    numerical_cols = X_processed.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        X_processed[f'{col}_squared'] = X_processed[col] ** 2
    
    # Alinear columnas
    expected_features = model_data['feature_names']
    for col in expected_features:
        if col not in X_processed.columns:
            X_processed[col] = 0
    
    return X_processed[expected_features]

# =============================================================================
# 3. TÉCNICAS DE EXPLICABILIDAD
# =============================================================================

def explain_model_global():
    """Explicabilidad global: importancia de características"""
    model_data = joblib.load("modelo_entrenado_avanzado.pkl")
    model = model_data['model']
    feature_selector = model_data['feature_selector']
    selected_features = model_data['selected_features']
    
    # Importancia de características
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Importancia de Características - Global")
    plt.bar(range(min(20, len(importances))), 
            importances[indices][:20])
    plt.xticks(range(min(20, len(importances))), 
               [selected_features[i] for i in indices[:20]], rotation=45)
    plt.tight_layout()
    plt.show()
    
    print("Top 10 características más importantes:")
    for i in range(min(10, len(importances))):
        print(f"{i+1}. {selected_features[indices[i]]}: {importances[indices[i]]:.4f}")

def explain_with_shap():
    """Explicabilidad con SHAP (valores Shapley)"""
    try:
        import shap
        
        model_data = joblib.load("modelo_entrenado_avanzado.pkl")
        model = model_data['model']
        feature_selector = model_data['feature_selector']
        X_train_balanced = model_data['X_train_balanced']
        selected_features = model_data['selected_features']
        
        # Preparar datos
        X_train_selected = feature_selector.transform(X_train_balanced)
        
        # Calcular SHAP values (muestra más pequeña para velocidad)
        sample_size = min(100, X_train_selected.shape[0])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train_selected[:sample_size])
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):  # Para clasificación multiclase
            shap.summary_plot(shap_values[1], X_train_selected[:sample_size], 
                            feature_names=selected_features, show=False)
        else:
            shap.summary_plot(shap_values, X_train_selected[:sample_size], 
                            feature_names=selected_features, show=False)
        plt.title("SHAP Summary Plot")
        plt.tight_layout()
        plt.show()
        
        # Force plot para un ejemplo específico
        plt.figure(figsize=(12, 6))
        shap.force_plot(explainer.expected_value[1] if isinstance(shap_values, list) else explainer.expected_value,
                       shap_values[1][0] if isinstance(shap_values, list) else shap_values[0],
                       X_train_selected[0], feature_names=selected_features, matplotlib=True, show=False)
        plt.title("SHAP Force Plot - Primer ejemplo")
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("SHAP no está instalado. Instala con: pip install shap")

def explain_with_pdp():
    """Partial Dependence Plots para características importantes"""
    from sklearn.inspection import PartialDependenceDisplay
    
    model_data = joblib.load("modelo_entrenado_avanzado.pkl")
    model = model_data['model']
    feature_selector = model_data['feature_selector']
    X_train_balanced = model_data['X_train_balanced']
    selected_features = model_data['selected_features']
    
    X_train_selected = feature_selector.transform(X_train_balanced)
    
    # Tomar las 4 características más importantes
    importances = model.feature_importances_
    top_features = np.argsort(importances)[-4:][::-1]
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax = ax.ravel()
    
    for i, feature_idx in enumerate(top_features):
        if i < 4:
            PartialDependenceDisplay.from_estimator(
                model, X_train_selected, [feature_idx],
                feature_names=selected_features,
                ax=ax[i]
            )
            ax[i].set_title(f"PDP: {selected_features[feature_idx]}")
    
    plt.tight_layout()
    plt.show()

def explain_local_prediction(instance_idx=0):
    """Explicabilidad local para predicción específica"""
    model_data = joblib.load("modelo_entrenado_avanzado.pkl")
    model = model_data['model']
    feature_selector = model_data['feature_selector']
    X_test = model_data['X_test']
    y_test = model_data['y_test']
    selected_features = model_data['selected_features']
    
    X_test_selected = feature_selector.transform(X_test)
    
    # Predicción para instancia específica
    instance = X_test_selected[instance_idx]
    true_label = y_test.iloc[instance_idx]
    prediction = model.predict([instance])[0]
    probability = model.predict_proba([instance])[0]
    
    print(f"=== Explicación Local - Instancia {instance_idx} ===")
    print(f"Predicción: {prediction} (Probabilidad: {probability[1]:.3f})")
    print(f"Valor real: {true_label}")
    print(f"¿Correcto?: {'SÍ' if prediction == true_label else 'NO'}")
    
    # Características más influyentes para esta predicción
    importances = model.feature_importances_
    feature_contributions = instance * importances
    
    top_contrib_idx = np.argsort(np.abs(feature_contributions))[-10:][::-1]
    
    print("\nTop características que influyeron en esta predicción:")
    for idx in top_contrib_idx:
        feat_name = selected_features[idx]
        feat_value = instance[idx]
        contribution = feature_contributions[idx]
        print(f"  {feat_name}: {feat_value:.3f} (contribución: {contribution:.4f})")

def run_complete_pipeline(dataset_path="dataset.csv"):
    """Ejecutar pipeline completo: entrenamiento + test + explicabilidad"""
    
    print("=" * 60)
    print("INICIANDO PIPELINE COMPLETO")
    print("=" * 60)
    
    # 1. Entrenamiento
    print("\n1. ENTRENANDO MODELO...")
    model_data = ModelTrainHalf(dataset_path)
    
    # 2. Tests
    print("\n2. EJECUTANDO TESTS...")
    testModelHalf_fixed(dataset_path)
    testModelFull(dataset_path)
    
    # 3. Explicabilidad
    print("\n3. EXPLICABILIDAD DEL MODELO...")
    
    print("\n3.1 IMPORTANCIA GLOBAL")
    explain_model_global()
    
    print("\n3.2 ANÁLISIS SHAP")
    explain_with_shap()
    
    print("\n3.3 PARTIAL DEPENDENCE PLOTS")
    explain_with_pdp()
    
    print("\n3.4 EXPLICACIÓN LOCAL")
    explain_local_prediction(instance_idx=0)
    explain_local_prediction(instance_idx=1)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO")
    print("=" * 60)

# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    # Ejecutar pipeline completo
    run_complete_pipeline("dataset.csv")
    
    # O ejecutar componentes individualmente:
    # ModelTrainHalf("dataset.csv")
    # testModelHalf_fixed("dataset.csv")
    # explain_model_global()
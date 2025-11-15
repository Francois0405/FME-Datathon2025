import pandas as pd
import numpy as np
from datetime import datetime
import random

def generar_datos_similares(n_registros=1000):
    """
    Genera datos similares al dataset proporcionado
    """
    np.random.seed(42)  # Para reproducibilidad
    
    datos = {
        'id': range(1, n_registros + 1),
        
        # Variables de productos (valores similares a los observados)
        'product_A_sold_in_the_past': np.random.normal(-0.2, 1.5, n_registros),
        'product_B_sold_in_the_past': np.random.normal(-0.2, 1.2, n_registros),
        'product_A_recommended': np.random.normal(-0.1, 0.8, n_registros),
        'product_A': np.random.normal(-0.1, 1.0, n_registros),
        'product_C': np.random.normal(-0.02, 0.1, n_registros),
        'product_D': np.random.normal(-0.04, 0.05, n_registros),
        
        # Variables de cliente
        'cust_hitrate': np.random.normal(0.1, 0.8, n_registros),
        'cust_interactions': np.random.normal(0.3, 1.2, n_registros),
        'cust_contracts': np.random.normal(-0.3, 1.0, n_registros),
        
        # Variables de oportunidad
        'opp_month': np.random.normal(-0.5, 1.0, n_registros),
        'opp_old': np.random.normal(-0.3, 1.5, n_registros),
        
        # Competidores (variables binarias)
        'competitor_Z': np.random.choice([0, 1], n_registros, p=[0.7, 0.3]),
        'competitor_X': np.random.choice([0, 1], n_registros, p=[0.9, 0.1]),
        'competitor_Y': np.random.choice([0, 1], n_registros, p=[0.8, 0.2]),
        
        # Variable geográfica
        'cust_in_iberia': np.random.choice([0, 1], n_registros, p=[0.6, 0.4]),
        
        # Variable objetivo
        'target_variable': np.random.choice([0, 1], n_registros, p=[0.55, 0.45])
    }
    
    # Crear DataFrame
    df = pd.DataFrame(datos)
    
    # Aplicar algunas transformaciones para hacer los datos más realistas
    # Algunos valores extremos como en el dataset original
    for col in ['product_A_sold_in_the_past', 'product_B_sold_in_the_past', 
                'product_A_recommended', 'product_A']:
        idx_extremos = np.random.choice(n_registros, size=int(n_registros * 0.05), replace=False)
        df.loc[idx_extremos, col] = np.random.normal(5, 3, len(idx_extremos))
    
    # Algunos valores muy específicos que aparecen en el dataset
    idx_especiales = np.random.choice(n_registros, size=int(n_registros * 0.02), replace=False)
    df.loc[idx_especiales, 'product_C'] = np.random.choice([9.8194, 36.23157, 5.7181], len(idx_especiales))
    
    # Redondear a 5 decimales como en el dataset original
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].round(5)
    
    return df

# Generar datos
print("Generando datos similares...")
datos_generados = generar_datos_similares(1000)

# Mostrar información básica
print(f"Datos generados: {datos_generados.shape}")
print("\nPrimeras 10 filas:")
print(datos_generados.head(10))

print("\nEstadísticas descriptivas:")
print(datos_generados.describe())

print("\nDistribución de la variable objetivo:")
print(datos_generados['target_variable'].value_counts())

# Guardar en CSV
nombre_archivo = f"dataset_generado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
datos_generados.to_csv(nombre_archivo, index=False)
print(f"\nDatos guardados en: {nombre_archivo}")

# Función adicional para generar datos con correlaciones específicas
def generar_datos_con_correlaciones(n_registros=1000):
    """
    Versión más avanzada que intenta replicar correlaciones entre variables
    """
    np.random.seed(42)
    
    # Generar algunas variables correlacionadas
    base = np.random.normal(0, 1, n_registros)
    
    datos_avanzados = {
        'id': range(1, n_registros + 1),
        
        # Productos correlacionados
        'product_A_sold_in_the_past': base + np.random.normal(0, 0.5, n_registros),
        'product_B_sold_in_the_past': base * 0.3 + np.random.normal(-0.2, 1.0, n_registros),
        
        # Variables de cliente con cierta correlación
        'cust_hitrate': np.random.normal(0.1, 0.8, n_registros),
        'cust_interactions': np.abs(np.random.normal(0.3, 1.2, n_registros)),
        'cust_contracts': np.random.normal(-0.3, 1.0, n_registros),
        
        # Otras variables similares a la función anterior
        'product_A_recommended': np.random.normal(-0.1, 0.8, n_registros),
        'product_A': np.random.normal(-0.1, 1.0, n_registros),
        'product_C': np.random.normal(-0.02, 0.1, n_registros),
        'product_D': np.random.normal(-0.04, 0.05, n_registros),
        'opp_month': np.random.normal(-0.5, 1.0, n_registros),
        'opp_old': np.random.normal(-0.3, 1.5, n_registros),
        'competitor_Z': np.random.choice([0, 1], n_registros, p=[0.7, 0.3]),
        'competitor_X': np.random.choice([0, 1], n_registros, p=[0.9, 0.1]),
        'competitor_Y': np.random.choice([0, 1], n_registros, p=[0.8, 0.2]),
        'cust_in_iberia': np.random.choice([0, 1], n_registros, p=[0.6, 0.4]),
        
        # Variable objetivo con cierta dependencia de otras variables
        'target_variable': np.where(
            (base + np.random.normal(0, 0.3, n_registros)) > 0.2, 1, 0
        )
    }
    
    df_avanzado = pd.DataFrame(datos_avanzados)
    
    # Añadir valores extremos
    for col in ['product_A_sold_in_the_past', 'product_B_sold_in_the_past']:
        idx_extremos = np.random.choice(n_registros, size=int(n_registros * 0.03), replace=False)
        df_avanzado.loc[idx_extremos, col] = np.random.uniform(2, 8, len(idx_extremos))
    
    # Redondear
    for col in df_avanzado.columns:
        if df_avanzado[col].dtype in [np.float64, np.float32]:
            df_avanzado[col] = df_avanzado[col].round(5)
    
    return df_avanzado

# Opción para generar datos con correlaciones
print("\n" + "="*50)
print("Generando datos con correlaciones...")
datos_correlacionados = generar_datos_con_correlaciones(1000)

# Guardar segunda versión
nombre_archivo2 = f"dataset_correlacionado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
datos_correlacionados.to_csv(nombre_archivo2, index=False)
print(f"Datos correlacionados guardados en: {nombre_archivo2}")

print("\nComparación de distribuciones:")
print("Dataset original vs Generado:")
print("- Variables continuas: distribuciones normales con algunos outliers")
print("- Variables binarias: proporciones similares")
print("- Formato: mismos nombres de columnas y precisión decimal")
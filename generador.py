import pandas as pd
import numpy as np
from datetime import datetime
import random

n_registros = 35899

def generar_datos_similares_realistas(n_registros):
    """
    Genera datos similares al dataset proporcionado con valores REALES dentro de los rangos observados
    """
    np.random.seed(3929)  # Para reproducibilidad
    
    # Rangos observados en el dataset original
    rangos = {
        'product_A_sold_in_the_past': {'min': -25992, 'max': 795386, 'tipo': 'entero'},
        'product_B_sold_in_the_past': {'min': -34794, 'max': 644745, 'tipo': 'entero'},
        'product_A_recommended': {'min': -10968, 'max': 508065, 'tipo': 'entero'},
        'product_A': {'min': -891, 'max': 1878559, 'tipo': 'entero'},
        'product_C': {'min': -2372, 'max': 98194, 'tipo': 'entero'},
        'product_D': {'min': -4247, 'max': 3623157, 'tipo': 'entero'},
        'cust_hitrate': {'min': -122789, 'max': 737997, 'tipo': 'entero'},
        'cust_interactions': {'min': -6817, 'max': 734997, 'tipo': 'entero'},
        'cust_contracts': {'min': -34997, 'max': 1046037, 'tipo': 'entero'},
        'opp_month': {'min': -141464, 'max': 183486, 'tipo': 'entero'},
        'opp_old': {'min': -28185, 'max': 354793, 'tipo': 'entero'}
    }
    
    # Valores específicos observados para variables categóricas
    valores_especificos = {
        'product_A_sold_in_the_past': [-25992, -19248, -586, 65945, 28136, 795386, 108216, 34402, 37589, 405959],
        'product_B_sold_in_the_past': [-34794, -28599, -24182, 335306, 225596, 325755, 254732, 351662, 200541, 153069],
        'product_A_recommended': [-1097, -10968, 59619, 51314, 404258, 258928, 335053, 40933, 196644, 508065],
        'product_A': [-891, 1227707, 478455, 223537, 1878559, 115062, 455984, 920878, 1385771, 765913],
        'product_C': [-2372, 98194],
        'product_D': [-4247, 3310429, 3623157, 1331632, 88978, 28511, 362118, 2585344],
        'cust_contracts': [-34997, -3202, 28593, 60388, 92184, 123979, 155774, 251159, 346544, 473725],
        'opp_month': [-141464, -111923, -82382, -52841, -233, 6241, 65323, 94864, 124404, 153945, 183486],
        'opp_old': [-28185, 354793]
    }
    
    datos = {'id': range(1, n_registros + 1)}
    
    # Generar variables numéricas dentro de rangos reales
    for col, config in rangos.items():
        if col in valores_especificos:
            # Para columnas con valores específicos, usar esos valores con probabilidades realistas
            valores_posibles = valores_especificos[col]
            if col in ['product_A_sold_in_the_past', 'product_B_sold_in_the_past']:
                # Alta probabilidad de valores negativos específicos
                if col == 'product_A_sold_in_the_past':
                    prob = [0.6, 0.1, 0.05] + [0.25/7] * 7  # 60% -25992, 10% -19248, etc.
                else:
                    prob = [0.5, 0.1, 0.05] + [0.35/7] * 7
                datos[col] = np.random.choice(valores_posibles, n_registros, p=prob)
            elif col in ['product_A_recommended', 'product_A']:
                # Alta probabilidad de -1097 y -891
                if col == 'product_A_recommended':
                    prob = [0.8] + [0.2/9] * 9
                else:
                    prob = [0.85] + [0.15/9] * 9
                datos[col] = np.random.choice(valores_posibles, n_registros, p=prob)
            elif col in ['product_C', 'product_D']:
                # Muy alta probabilidad de valores negativos
                if col == 'product_C':
                    prob = [0.95, 0.05]
                else:
                    prob = [0.98] + [0.02/7] * 7
                datos[col] = np.random.choice(valores_posibles, n_registros, p=prob)
            elif col == 'cust_contracts':
                # Distribución más balanceada pero con -34997 frecuente
                prob = [0.4, 0.1] + [0.5/8] * 8
                datos[col] = np.random.choice(valores_posibles, n_registros, p=prob)
            elif col == 'opp_month':
                # Distribución más uniforme entre valores específicos
                prob = [1/len(valores_posibles)] * len(valores_posibles)
                datos[col] = np.random.choice(valores_posibles, n_registros, p=prob)
            elif col == 'opp_old':
                # 70% -28185, 30% 354793
                datos[col] = np.random.choice(valores_posibles, n_registros, p=[0.7, 0.3])
        else:
            # Para otras columnas, generar dentro del rango
            if col == 'cust_hitrate':
                datos[col] = np.random.normal(0, 50000, n_registros).astype(int)
            elif col == 'cust_interactions':
                datos[col] = np.abs(np.random.normal(50000, 80000, n_registros)).astype(int)
            else:
                base = np.random.normal(0, 1, n_registros)
                escalado = (base - base.min()) / (base.max() - base.min())
                datos[col] = (config['min'] + escalado * (config['max'] - config['min'])).astype(int)
    
    # Asegurar que todos los valores estén dentro de los rangos
    for col, config in rangos.items():
        datos[col] = np.clip(datos[col], config['min'], config['max'])
    
    # Variables binarias con proporciones realistas
    datos['competitor_Z'] = np.random.choice([0, 1], n_registros, p=[0.85, 0.15])
    datos['competitor_X'] = np.random.choice([0, 1], n_registros, p=[0.92, 0.08])
    datos['competitor_Y'] = np.random.choice([0, 1], n_registros, p=[0.75, 0.25])
    datos['cust_in_iberia'] = np.random.choice([0, 1], n_registros, p=[0.55, 0.45])
    
    # Variable objetivo con correlación real
    base_target = (
        0.3 * (datos['product_A_sold_in_the_past'] > 0).astype(int) +
        0.2 * (datos['product_B_sold_in_the_past'] > 0).astype(int) +
        0.1 * datos['cust_in_iberia'] +
        0.1 * (datos['cust_contracts'] > 0).astype(int) +
        np.random.normal(0, 0.4, n_registros)
    )
    datos['target_variable'] = (base_target > 0.5).astype(int)
    
    return pd.DataFrame(datos)

# Generar datos realistas
print("Generando datos REALISTAS similares al dataset original...")
datos_realistas = generar_datos_similares_realistas(n_registros)

# Mostrar información básica
print(f"Datos generados: {datos_realistas.shape}")
print("\nPrimeras 10 filas:")
print(datos_realistas.head(10))

print("\nEstadísticas descriptivas:")
print(datos_realistas.describe())

print("\nDistribución de variables binarias:")
print("competitor_Z:", datos_realistas['competitor_Z'].value_counts())
print("competitor_X:", datos_realistas['competitor_X'].value_counts())
print("competitor_Y:", datos_realistas['competitor_Y'].value_counts())
print("cust_in_iberia:", datos_realistas['cust_in_iberia'].value_counts())
print("target_variable:", datos_realistas['target_variable'].value_counts())

# Verificar rangos
print("\nVerificación de rangos:")
for col in ['product_A_sold_in_the_past', 'product_B_sold_in_the_past', 'product_A_recommended', 'product_A']:
    print(f"{col}: min={datos_realistas[col].min()}, max={datos_realistas[col].max()}")

# Mostrar valores únicos para algunas columnas clave
print("\nValores únicos en product_A_sold_in_the_past:")
print(datos_realistas['product_A_sold_in_the_past'].value_counts().head(10))

print("\nValores únicos en product_B_sold_in_the_past:")
print(datos_realistas['product_B_sold_in_the_past'].value_counts().head(10))

# Guardar en CSV con punto y coma como separador
nombre_archivo = f"dataset_realista_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
datos_realistas.to_csv(nombre_archivo, sep=';', index=False)
print(f"\nDatos guardados en: {nombre_archivo} (separador: punto y coma)")

# Versión simplificada para datos de prueba más rápidos
def generar_datos_rapidos(n_registros):
    """Versión más rápida para testing"""
    np.random.seed(42)
    
    datos = {'id': range(1, n_registros + 1)}
    
    # Variables principales con valores específicos
    datos['product_A_sold_in_the_past'] = np.random.choice(
        [-25992, 65945, 28136, 795386, 108216], n_registros, p=[0.7, 0.1, 0.1, 0.05, 0.05])
    
    datos['product_B_sold_in_the_past'] = np.random.choice(
        [-34794, 335306, 225596, 325755, 254732], n_registros, p=[0.7, 0.1, 0.1, 0.05, 0.05])
    
    datos['product_A_recommended'] = np.random.choice(
        [-1097, 404258, 258928, 335053], n_registros, p=[0.8, 0.07, 0.07, 0.06])
    
    datos['product_A'] = np.random.choice(
        [-891, 1227707, 478455, 223537], n_registros, p=[0.85, 0.05, 0.05, 0.05])
    
    # Otras variables
    datos['product_C'] = -2372
    datos['product_D'] = -4247
    
    datos['cust_hitrate'] = np.random.normal(0, 50000, n_registros).astype(int)
    datos['cust_interactions'] = np.abs(np.random.normal(50000, 80000, n_registros)).astype(int)
    datos['cust_contracts'] = np.random.choice([-34997, 28593, 60388, 92184], n_registros)
    
    datos['opp_month'] = np.random.choice([-141464, -111923, -82382, -233, 124404, 183486], n_registros)
    datos['opp_old'] = np.random.choice([-28185, 354793], n_registros, p=[0.7, 0.3])
    
    # Variables binarias
    datos['competitor_Z'] = np.random.choice([0, 1], n_registros, p=[0.85, 0.15])
    datos['competitor_X'] = np.random.choice([0, 1], n_registros, p=[0.92, 0.08])
    datos['competitor_Y'] = np.random.choice([0, 1], n_registros, p=[0.75, 0.25])
    datos['cust_in_iberia'] = np.random.choice([0, 1], n_registros, p=[0.55, 0.45])
    
    # Variable objetivo
    base_target = (
        0.3 * (datos['product_A_sold_in_the_past'] > 0).astype(int) +
        0.2 * (datos['product_B_sold_in_the_past'] > 0).astype(int) +
        0.1 * datos['cust_in_iberia'] +
        np.random.normal(0, 0.4, n_registros)
    )
    datos['target_variable'] = (base_target > 0.5).astype(int)
    
    return pd.DataFrame(datos)

# Generar versión rápida para testing
print("\n" + "="*60)
print("Generando versión RÁPIDA para testing...")
datos_rapidos = generar_datos_rapidos(n_registros)

nombre_archivo_rapido = f"dataset_rapido_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
datos_rapidos.to_csv(nombre_archivo_rapido, sep=';', index=False)
print(f"Datos rápidos guardados en: {nombre_archivo_rapido}")

print("\n✅ GENERACIÓN COMPLETADA")
print("Características de los datos generados:")
print("- Separador: punto y coma (;)")
print("- Valores dentro de rangos REALES del dataset original")
print("- Distribuciones REALISTAS replicando patrones observados")
print("- Variables binarias con proporciones realistas")
print("- Correlaciones entre variables objetivo y predictores")
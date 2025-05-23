import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Configuración de la página
st.set_page_config(page_title="Optimización Energética Industrial", layout="wide")
st.title("Optimización de Consumo Energético con Multiplicadores de Lagrange")

# Coeficientes calibrados
a, b, c, d = 0.0496, 0.0012, 0.0189, 1.9958

# Función de consumo energético
def energy_consumption(T, V, H):
    return a * T**2 + b * V**3 + c * T * V + d * H

# Función para optimización con Lagrange
def optimize_energy(P0):
    def objetivo(x):
        T, V = x
        return energy_consumption(T, V, H=1)  # Fijamos H=1 para el caso de 1 hora
    
    def restriccion(x):
        T, V = x
        return 0.5 * V * T - P0  # Restricción: P(T,V) >= P0
    
    resultado = minimize(
        objetivo,
        x0=[25, 10],
        constraints={'type': 'ineq', 'fun': restriccion},
        bounds=[(10, 50), (1, 20)]
    )
    return resultado.x[0], resultado.x[1]  # T_opt, V_opt

# Sidebar
with st.sidebar:
    st.header("Parámetros de Control")
    P0 = st.slider("Producción mínima requerida (unidades/hora)", 100, 1000, 500)
    H_fixed = 1  # Restricción de 1 hora de operación
    
    # Calcular punto óptimo
    T_opt, V_opt = optimize_energy(P0)
    E_opt = energy_consumption(T_opt, V_opt, H_fixed)
    
    st.divider()
    st.markdown(f"""
    **Resultados de Optimización**:
    - Temperatura óptima: `{T_opt:.2f} °C`
    - Velocidad óptima: `{V_opt:.2f} unidades/min`
    - Consumo mínimo: `{E_opt:.2f} kW/h`
    """)

# Pestañas principales
tab1, tab2, tab3 = st.tabs(["Optimización", "Sensibilidad", "Gradiente"])

with tab1:
    st.header("Configuración Óptima de Operación")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Punto Óptimo**:
        - **Temperatura**: Minimiza el consumo dado que altas temperaturas aumentan el término cuadrático.
        - **Velocidad**: Balancea la producción con el término cúbico de fricción.
        """)
    with col2:
        # Gráfico de la restricción VT = 2*P0 (para H=1)
        V_range = np.linspace(1, 20, 100)
        T_constraint = (2 * P0) / V_range
        
        fig, ax = plt.subplots()
        ax.plot(T_constraint, V_range, 'r-', label=f'Restricción: $VT = {2*P0}$')
        ax.scatter(T_opt, V_opt, c='green', s=100, label='Punto óptimo')
        ax.set_xlabel('Temperatura (°C)')
        ax.set_ylabel('Velocidad (unidades/min)')
        ax.legend()
        ax.grid()
        st.pyplot(fig)

with tab2:
    st.header("Análisis de Sensibilidad")
    
    # Derivadas parciales en el punto óptimo
    dEdT = 2 * a * T_opt + c * V_opt
    dEdV = 3 * b * V_opt**2 + c * T_opt
    
    st.markdown(f"""
    **Efecto de Cambios Pequeños**:
    - ∂E/∂T = `{dEdT:.2f}` kW/h por °C  
      → Aumentar 1°C eleva el consumo en `{dEdT:.2f}` kW/h.
    - ∂E/∂V = `{dEdV:.2f}` kW/h por (unidad/min)  
      → Aumentar 1 unidad/min eleva el consumo en `{dEdV:.2f}` kW/h.
    """)

with tab3:
    st.header("Campo Vectorial del Gradiente")
    
    # Generar grid para el gradiente
    T_grid = np.linspace(10, 50, 10)
    V_grid = np.linspace(1, 20, 10)
    T_mesh, V_mesh = np.meshgrid(T_grid, V_grid)
    
    # Calcular gradientes
    dEdT_grid = 2 * a * T_mesh + c * V_mesh
    dEdV_grid = 3 * b * V_mesh**2 + c * T_mesh
    
    # Gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.quiver(T_mesh, V_mesh, dEdT_grid, dEdV_grid, scale=500, color='blue', width=0.005)
    ax.scatter(T_opt, V_opt, c='red', s=100, label='Punto óptimo')
    ax.set_xlabel('Temperatura (°C)')
    ax.set_ylabel('Velocidad (unidades/min)')
    ax.set_title('Dirección de Máximo Aumento del Consumo (∇E)')
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    **Interpretación**:
    - Las flechas azules muestran cómo aumenta el consumo al cambiar $T$ y $V$.
    - En el punto óptimo (rojo), el gradiente es ortogonal a la restricción (curva roja en la pestaña 1).
    """)
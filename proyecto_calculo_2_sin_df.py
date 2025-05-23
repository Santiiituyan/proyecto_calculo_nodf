import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import minimize # Import the minimize function

# --- Configuración inicial de la página ---
st.set_page_config(page_title="Optimización Energética Industrial", layout="wide")
st.title("Análisis de Consumo Energético Industrial")

# --- Carga de datos (sin cambios) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('industrial_consumption_dataset.csv')
        if not all(col in df.columns for col in ['Temperature', 'Velocity', 'Hours']):
            st.error("El dataset no contiene las columnas requeridas. Usando datos generados.")
            raise ValueError
        return df
    except Exception as e:
        st.warning("No se pudo cargar el archivo. Generando datos sintéticos...")
        np.random.seed(42)
        data = {
            'Temperature': np.random.uniform(10, 50, 100),
            'Velocity': np.random.uniform(1, 20, 100),
            'Hours': np.random.randint(1, 24, 100)
        }
        return pd.DataFrame(data)

df = load_data()

# --- Cálculo de energía y derivadas (para el modelo original) ---
df['Energy'] = 0.5*df['Temperature']**2 + 0.1*df['Velocity']**3 + 2*df['Hours'] + 0.3*df['Temperature']*df['Velocity']
df['dEdT'] = df['Temperature'] + 0.3*df['Velocity']
df['dEdV'] = 0.3*df['Velocity']**2 + 0.3*df['Temperature']

# --- Sidebar para controles (sin cambios significativos, H ahora solo para el modelo original) ---
with st.sidebar:
    st.header("Parámetros de Control (Modelo Original)")
    T = st.slider("Temperatura (°C)", 10.0, 50.0, 25.0, 0.1, key="T_original")
    V = st.slider("Velocidad (unidades/min)", 1.0, 20.0, 10.0, 0.1, key="V_original")
    H = st.slider("Horas de operación", 1, 24, 8, key="H_original")
    st.divider()
    st.caption("Proyecto de Cálculo II - Optimización Energética")

# --- Cálculos principales (para el modelo original) ---
E_original = 0.5*T**2 + 0.1*V**3 + 2*H + 0.3*T*V
dEdT_original = T + 0.3*V
dEdV_original = 0.3*V**2 + 0.3*T

# --- Filtrado de datos similares (para el modelo original) ---
similar_data = df[
    (df['Temperature'].between(T-2, T+2)) &
    (df['Velocity'].between(V-1, V+1))
]

# --- Definición de la función de energía y restricción para el problema de optimización ---
def energy_consumption(params):
    """
    Función de consumo energético para el problema de optimización.
    params = [T, V]
    E(T,V) = 0.0496 T^2 + 0.0012 V^3 + 0.0189 TV
    """
    T_opt, V_opt = params
    return 0.0496 * T_opt**2 + 0.0012 * V_opt**3 + 0.0189 * T_opt * V_opt

def production_constraint(params):
    """
    Restricción de producción: P(T,V) = 0.5 * V * T = 500  =>  V * T - 1000 = 0
    """
    T_opt, V_opt = params
    return (0.5 * V_opt * T_opt) - 500

# --- Visualización en pestañas ---
tab1, tab2, tab3, tab4 = st.tabs(["Resultados", "Validación", "Análisis", "Optimización Mejorada"])

with tab1:
    st.header("Resultados Principales")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Consumo Energético", f"{E_original:.2f} kW/h")
    with col2:
        st.metric("Sensibilidad a Temperatura", f"{dEdT_original:.2f} kW/h/°C")
    with col3:
        st.metric("Sensibilidad a Velocidad", f"{dEdV_original:.2f} kW/h/(unidad/min)")

    st.write("Distribución del consumo en los datos:")
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['Temperature'], df['Velocity'], c=df['Energy'], cmap='viridis')
    ax.scatter(T, V, c='red', s=70, label='Punto seleccionado')
    plt.colorbar(scatter, label='Consumo Energético (kW/h)')
    ax.set_xlabel('Temperatura (°C)')
    ax.set_ylabel('Velocidad (unidades/min)')
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.header("Validación con Datos Simulados")

    if len(similar_data) > 0:
        st.write(f"Datos encontrados con parámetros similares: {len(similar_data)} registros")

        numeric_dEdT = similar_data['Energy'].diff() / similar_data['Temperature'].diff()
        numeric_dEdV = similar_data['Energy'].diff() / similar_data['Velocity'].diff()

        st.subheader("Comparación de Derivadas")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**∂E/∂T (Temperatura):**")
            st.write(f"Teórica: {dEdT_original:.2f}")
            st.write(f"Numérica: {numeric_dEdT.mean():.2f}")
        with col2:
            st.write("**∂E/∂V (Velocidad):**")
            st.write(f"Teórica: {dEdV_original:.2f}")
            st.write(f"Numérica: {numeric_dEdV.mean():.2f}")

        st.subheader("Datos Similares")
        st.dataframe(similar_data[['Temperature', 'Velocity', 'Hours', 'Energy']].head())
    else:
        st.warning("No se encontraron datos con parámetros similares")

with tab3:
    st.header("Análisis Avanzado")

    st.subheader("Campo Vectorial del Gradiente")
    st.write("Visualización de cómo cambia el consumo energético en diferentes condiciones:")

    sample = df.sample(15, random_state=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(
        sample['Temperature'],
        sample['Velocity'],
        sample['dEdT'],
        sample['dEdV'],
        scale=300,
        color='blue',
        width=0.003
    )
    ax.scatter(T, V, c='red', s=70, label='Punto actual')
    ax.set_xlabel('Temperatura (°C)')
    ax.set_ylabel('Velocidad (unidades/min)')
    ax.set_title('Campo Vectorial del Gradiente ∇E(T,V)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Distribución del Consumo")
    st.line_chart(df.groupby('Temperature')['Energy'].mean())


with tab4:
    st.header("Optimización de Consumo con Restricción de Producción")
    st.write("Aquí se resuelve el problema de optimización usando el método de Lagrange para encontrar la temperatura y velocidad óptimas que minimizan el consumo energético, dada una producción mínima.")

    st.subheader("Definición del Problema")
    st.markdown(r"""
    **Función de Consumo Energético ($E(T,V)$):**
    $$ E(T,V) = 0.0496\,T^2 + 0.0012\,V^3 + 0.0189\,TV $$
    **Restricción de Producción ($P(T,V)$):**
    Una producción mínima de 500 unidades en 1 hora, donde $P(T,V) = 0.5 \cdot V \cdot T$.
    Así, la restricción es:
    $$ 0.5 \cdot V \cdot T = 500 \quad \Rightarrow \quad V \cdot T = 1000 $$
    """)

    st.subheader("Cálculo Numérico de Optimización")

    # Definir la restricción para scipy.optimize
    constraints = ({'type': 'eq', 'fun': production_constraint})

    # Límites para T y V (ajusta según la realidad física de tu máquina)
    # Es crucial establecer límites realistas para la optimización numérica.
    # Por ejemplo, la temperatura no puede ser negativa, ni la velocidad.
    # Basado en tu problema, T podría ser entre 10-100 y V entre 1-50, por ejemplo.
    bounds = [(10, 100), (1, 50)] # (min_T, max_T), (min_V, max_V)

    # Punto inicial de guess (importante para la optimización numérica)
    # Un buen punto inicial puede ayudar a encontrar el óptimo global más rápido.
    # Podríamos usar los valores del slider del modelo original como un punto de partida.
    initial_guess = [T, V]

    try:
        # Realizar la optimización
        result = minimize(energy_consumption, initial_guess, constraints=constraints, bounds=bounds)

        if result.success:
            optimal_T, optimal_V = result.x
            min_energy = result.fun

            st.success("¡Optimización Exitosa!")
            st.write("Se encontraron los parámetros óptimos para minimizar el consumo energético bajo la restricción de producción.")

            col_opt1, col_opt2, col_opt3 = st.columns(3)
            with col_opt1:
                st.metric("Temperatura Óptima", f"{optimal_T:.2f} °C")
            with col_opt2:
                st.metric("Velocidad Óptima", f"{optimal_V:.2f} unidades/min")
            with col_opt3:
                st.metric("Consumo Energético Mínimo", f"{min_energy:.2f} kW/h")

            # Verificación de la restricción
            st.write(f"**Verificación de la restricción (V * T):** {optimal_V * optimal_T:.2f} (Esperado: 1000)")

            st.subheader("Comparación con tu Cálculo Analítico")
            st.write(f"Tu T analítica: **54.79 °C**")
            st.write(f"Tu V analítica: **18.25 unidades/min**")
            st.write(f"Tu Consumo analítico: **175.09 kW/h**")

            st.info("La optimización numérica puede variar ligeramente de los resultados analíticos debido a la precisión del algoritmo y el punto de inicio. Sin embargo, deberían ser muy cercanos.")

        else:
            st.error(f"La optimización no convergió. Mensaje: {result.message}")
            st.write("Intenta ajustar los límites o el punto inicial si la optimización falla repetidamente.")

    except Exception as e:
        st.error(f"Ocurrió un error durante la optimización: {e}")
        st.write("Asegúrate de que tus funciones y restricciones estén definidas correctamente.")

    st.markdown("---")
    st.subheader("Visualización del Espacio de Soluciones (Simplificado)")
    st.write("Esta gráfica muestra la función de consumo energético y la línea de la restricción, ayudando a visualizar el punto óptimo.")

    # Crear un rango de valores para T y V
    T_vals = np.linspace(10, 80, 100) # Rango de T para la gráfica
    V_vals = np.linspace(1, 40, 100) # Rango de V para la gráfica
    T_grid, V_grid = np.meshgrid(T_vals, V_vals)

    # Calcular la energía para cada punto en la cuadrícula
    E_grid = 0.0496 * T_grid**2 + 0.0012 * V_grid**3 + 0.0189 * T_grid * V_grid

    fig_opt, ax_opt = plt.subplots(figsize=(6, 3))
    contour = ax_opt.contourf(T_grid, V_grid, E_grid, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, label='Consumo Energético (kW/h)')

    # Plotear la restricción VT=1000
    V_constraint = 1000 / T_vals
    # Filtrar para que la línea de restricción esté dentro de los límites de V
    V_constraint_filtered = V_constraint[(V_constraint >= bounds[1][0]) & (V_constraint <= bounds[1][1])]
    T_constraint_filtered = T_vals[(V_constraint >= bounds[1][0]) & (V_constraint <= bounds[1][1])]

    ax_opt.plot(T_constraint_filtered, V_constraint_filtered, 'r--', label='Restricción: VT = 1000')

    if result.success:
        ax_opt.plot(optimal_T, optimal_V, 'ro', markersize=10, label='Punto Óptimo')
        ax_opt.text(optimal_T + 2, optimal_V + 2, f'E_min: {min_energy:.2f}', color='red')

    ax_opt.set_xlabel('Temperatura (°C)')
    ax_opt.set_ylabel('Velocidad (unidades/min)')
    ax_opt.set_title('Mapa de Consumo Energético con Restricción')
    ax_opt.legend()
    ax_opt.grid(True)
    st.pyplot(fig_opt)
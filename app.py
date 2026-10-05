import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from scipy import sparse


st.set_page_config(
    page_title="Prediccion de riesgo academico",
    page_icon="🎓",
    layout="wide",
)

MODEL_PATH = Path("random_forest_streamlit.pkl")
CONFIG_PATH = Path("configuracion_encoder.json")


@st.cache_resource
def cargar_artefactos():
    """Carga el Random Forest y la configuracion portable del encoder."""
    modelo = joblib.load(MODEL_PATH)
    with CONFIG_PATH.open("r", encoding="utf-8") as archivo:
        configuracion = json.load(archivo)
    return modelo, configuracion


try:
    modelo, configuracion = cargar_artefactos()
except FileNotFoundError as error:
    st.error(
        "No se encontro uno de los archivos requeridos. Verifica que "
        "random_forest_streamlit.pkl y configuracion_encoder.json esten "
        "en la misma carpeta que app.py."
    )
    st.exception(error)
    st.stop()
except Exception as error:
    st.error(f"No fue posible cargar los artefactos del modelo: {error}")
    st.stop()


columnas = configuracion["columnas"]
categorias_por_columna = dict(zip(columnas, configuracion["categorias"]))


def categorias(nombre_columna):
    """Retorna las categorias exactas aprendidas durante el entrenamiento."""
    return categorias_por_columna[nombre_columna]


def transformar_registro(registro):
    """
    Reproduce OneHotEncoder(drop='first', handle_unknown='ignore') sin cargar
    ColumnTransformer. El orden se obtiene del JSON exportado del encoder.
    """
    indices = []
    desplazamiento = 0

    for columna, categorias_columna in zip(
        configuracion["columnas"], configuracion["categorias"]
    ):
        valor = registro[columna]

        if valor in categorias_columna:
            posicion = categorias_columna.index(valor)
            if posicion > 0:
                indices.append(desplazamiento + posicion - 1)

        desplazamiento += max(len(categorias_columna) - 1, 0)

    datos = np.ones(len(indices), dtype=np.float64)
    filas = np.zeros(len(indices), dtype=np.int32)

    return sparse.csr_matrix(
        (datos, (filas, indices)),
        shape=(1, desplazamiento),
        dtype=np.float64,
    )


def etiqueta_si_no(valor):
    return str(valor)


st.title("🎓 Prediccion de riesgo de bajo desempeño")
st.markdown(
    """
    ### Proyecto academico
    **Maestria en Ciencia de Datos · Universidad Pontificia Bolivariana (UPB)**

    Esta aplicacion estima el riesgo de bajo desempeño en razonamiento
    cuantitativo a partir de caracteristicas academicas y socioeconomicas.
    """
)
st.info(
    "La prediccion es una herramienta de apoyo preventivo. No reemplaza la "
    "valoracion academica individual ni debe utilizarse para tomar decisiones "
    "excluyentes sobre estudiantes."
)
st.divider()

st.subheader("Informacion del estudiante")

with st.form("formulario_prediccion"):
    columna_1, columna_2 = st.columns(2)

    with columna_1:
        periodo = st.selectbox(
            "Periodo de presentacion",
            categorias("PERIODO"),
            format_func=str,
        )

        modalidad = st.selectbox(
            "Modalidad del programa",
            categorias("ESTU_METODO_PRGM"),
            format_func=str,
        )

        opciones_horas = categorias("ESTU_HORASSEMANATRABAJA")
        horas_trabajo = st.select_slider(
            "Horas de trabajo semanales",
            options=opciones_horas,
            value=opciones_horas[0],
            format_func=str,
        )

        genero = st.radio(
            "Genero registrado",
            categorias("ESTU_GENERO"),
            horizontal=True,
            format_func=str,
        )

        estrato = st.selectbox(
            "Estrato socioeconomico",
            categorias("FAMI_ESTRATOVIVIENDA"),
            format_func=str,
        )

    with columna_2:
        educacion_padre = st.selectbox(
            "Nivel educativo del padre",
            categorias("FAMI_EDUCACIONPADRE"),
            format_func=str,
        )

        educacion_madre = st.selectbox(
            "Nivel educativo de la madre",
            categorias("FAMI_EDUCACIONMADRE"),
            format_func=str,
        )

        opciones_computador = categorias("FAMI_TIENECOMPUTADOR")
        computador_predeterminado = (
            opciones_computador.index("Si")
            if "Si" in opciones_computador
            else 0
        )
        tiene_computador = st.radio(
            "¿Cuenta con computador?",
            opciones_computador,
            index=computador_predeterminado,
            horizontal=True,
            format_func=str,
        )

        opciones_internet = categorias("FAMI_TIENEINTERNET")
        internet_predeterminado = (
            opciones_internet.index("Si") if "Si" in opciones_internet else 0
        )
        tiene_internet = st.radio(
            "¿Cuenta con acceso a internet?",
            opciones_internet,
            index=internet_predeterminado,
            horizontal=True,
            format_func=str,
        )

        origen_institucion = st.selectbox(
            "Origen de la institucion",
            categorias("INST_ORIGEN"),
            format_func=str,
        )

    enviar = st.form_submit_button(
        "🔍 Realizar prediccion",
        use_container_width=True,
    )


if enviar:
    registro = {
        "PERIODO": periodo,
        "ESTU_METODO_PRGM": modalidad,
        "ESTU_HORASSEMANATRABAJA": horas_trabajo,
        "ESTU_GENERO": genero,
        "FAMI_EDUCACIONPADRE": educacion_padre,
        "FAMI_ESTRATOVIVIENDA": estrato,
        "FAMI_TIENECOMPUTADOR": tiene_computador,
        "FAMI_TIENEINTERNET": tiene_internet,
        "FAMI_EDUCACIONMADRE": educacion_madre,
        "INST_ORIGEN": origen_institucion,
    }

    try:
        entrada_transformada = transformar_registro(registro)

        if entrada_transformada.shape[1] != modelo.n_features_in_:
            raise ValueError(
                "La configuracion del encoder produce "
                f"{entrada_transformada.shape[1]} variables, pero el modelo "
                f"espera {modelo.n_features_in_}."
            )

        prediccion = int(modelo.predict(entrada_transformada)[0])
        probabilidad_riesgo = float(
            modelo.predict_proba(entrada_transformada)[0][1]
        )
    except Exception as error:
        st.error(f"No fue posible realizar la prediccion: {error}")
        st.stop()

    st.divider()
    st.subheader("Resultado de la prediccion")

    resultado_1, resultado_2 = st.columns([1, 2])

    with resultado_1:
        st.metric(
            "Probabilidad estimada de riesgo",
            f"{probabilidad_riesgo * 100:.2f}%",
        )
        st.progress(probabilidad_riesgo)

    with resultado_2:
        if prediccion == 1:
            st.error("⚠️ ESTUDIANTE CON RIESGO DE BAJO DESEMPEÑO")
            st.write(
                "El modelo identifica un perfil asociado con riesgo de bajo "
                "desempeño en razonamiento cuantitativo. Se recomienda validar "
                "el resultado y considerar acciones preventivas de acompañamiento."
            )
        else:
            st.success("✅ ESTUDIANTE SIN RIESGO DE BAJO DESEMPEÑO")
            st.write(
                "El modelo identifica un perfil asociado con desempeño esperado. "
                "El resultado no reemplaza el seguimiento academico individual."
            )

    with st.expander("Ver datos utilizados en la prediccion"):
        etiquetas = {
            "PERIODO": "Periodo",
            "ESTU_METODO_PRGM": "Modalidad",
            "ESTU_HORASSEMANATRABAJA": "Horas de trabajo",
            "ESTU_GENERO": "Genero",
            "FAMI_EDUCACIONPADRE": "Educacion del padre",
            "FAMI_ESTRATOVIVIENDA": "Estrato",
            "FAMI_TIENECOMPUTADOR": "Computador",
            "FAMI_TIENEINTERNET": "Internet",
            "FAMI_EDUCACIONMADRE": "Educacion de la madre",
            "INST_ORIGEN": "Origen de la institucion",
        }
        for clave in columnas:
            st.write(f"**{etiquetas.get(clave, clave)}:** {registro[clave]}")


st.divider()
st.caption(
    "Developed by Daniel Lezcano Rios · "
    "Maestria en Ciencia de Datos · Universidad Pontificia Bolivariana (UPB)"
)

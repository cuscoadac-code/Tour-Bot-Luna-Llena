import streamlit as st
import pandas as pd
import glob
import os
import warnings
import numpy as np
import google.generativeai as genai

# ==========================================
# 1. CONFIGURACIÓN VISUAL
# ==========================================
st.set_page_config(page_title="TourBot Pro", page_icon="🧠", layout="centered")
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# ---------------------------------------------------------
# 🔑 ZONA DE LA LLAVE
# ---------------------------------------------------------
# Tu clave configurada:
GOOGLE_API_KEY = "AIzaSyBZ4_AblsxBY3x67yRqZzsGrW-gRSPR5bU"
# ---------------------------------------------------------

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except:
    st.error("⚠️ Error: Revisa tu API Key.")

# Estilos CSS Modernos (Naranja TourBot)
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stChatInput textarea {border-radius: 20px !important;}
    .stChatInput button {color: #FF5722 !important;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CARGADOR DE DATOS + EMBEDDINGS (VECTORES)
# ==========================================
@st.cache_data
def cargar_base_conocimiento():
    """
    Lee el Excel y convierte la información en vectores matemáticos.
    """
    ruta_carpeta = "."
    todos_los_datos = []
    
    archivos = glob.glob(os.path.join(ruta_carpeta, "*.xlsx"))
    
    for archivo in archivos:
        try:
            xls = pd.ExcelFile(archivo)
            for nombre_hoja in xls.sheet_names:
                try:
                    df = pd.read_excel(xls, sheet_name=nombre_hoja, header=None)
                    df_str = df.astype(str)
                    coords = df_str[df_str.apply(lambda row: row.astype(str).str.contains('PROMPT', case=False).any(), axis=1)]
                    
                    if not coords.empty:
                        fila = coords.index[0]
                        df.columns = df.iloc[fila]
                        df = df.iloc[fila + 1:]
                        
                        cols = df.columns.astype(str).str.upper()
                        col_p_list = df.columns[cols.str.contains('PROMPT')]
                        col_c_list = df.columns[cols.str.contains('COMPLETION')]
                        
                        if len(col_p_list) > 0 and len(col_c_list) > 0:
                            temp_df = df[[col_p_list[0], col_c_list[0]]].dropna()
                            temp_df.columns = ['Pregunta', 'Respuesta']
                            temp_df['Fuente'] = nombre_hoja
                            todos_los_datos.append(temp_df)
                except:
                    continue
        except:
            pass
            
    if not todos_los_datos:
        return pd.DataFrame(), None

    base_datos = pd.concat(todos_los_datos, ignore_index=True)
    
    # Vectorización (Embeddings)
    df_vectorizar = base_datos.copy()
    textos = df_vectorizar.apply(lambda x: f"Pregunta: {x['Pregunta']} \n Respuesta: {x['Respuesta']}", axis=1).tolist()
    
    try:
        resultado = genai.embed_content(
            model="models/text-embedding-004",
            content=textos,
            task_type="retrieval_document"
        )
        vectores = np.array(resultado['embedding'])
        return df_vectorizar, vectores
    except Exception as e:
        st.error(f"Error vectores: {e}")
        return pd.DataFrame(), None

df_conocimiento, matriz_vectores = cargar_base_conocimiento()

# ==========================================
# 3. MOTOR DE BÚSQUEDA SEMÁNTICA
# ==========================================
def buscar_respuesta_inteligente(pregunta_usuario):
    if df_conocimiento.empty or matriz_vectores is None:
        return None, "Base vacía"

    try:
        vec_pregunta = genai.embed_content(
            model="models/text-embedding-004",
            content=pregunta_usuario,
            task_type="retrieval_query"
        )['embedding']
        
        puntajes = np.dot(matriz_vectores, vec_pregunta)
        mejor_indice = np.argmax(puntajes)
        mejor_match = df_conocimiento.iloc[mejor_indice]
        score = puntajes[mejor_indice]
        
        if score < 0.45: 
            return None, f"Confianza baja ({score:.2f})"
            
        contexto = f"FUENTE: {mejor_match['Fuente']}\nINFO TÉCNICA: {mejor_match['Respuesta']}"
        return contexto, f"Match: {mejor_match['Fuente']} ({score:.2f})"
        
    except Exception as e:
        return None, f"Error: {e}"

# ==========================================
# 4. CEREBRO EXPERTO (PROMPT CIENTÍFICO + FEW-SHOT)
# ==========================================
def generar_con_memoria(pregunta_actual, historial_chat):
    # 1. Búsqueda técnica (RAG)
    contexto_nuevo, debug = buscar_respuesta_inteligente(pregunta_actual)
    
    # 2. Historial de Chat
    historial_texto = ""
    for msg in historial_chat[-5:]: # Recordamos los últimos 5 mensajes
        role = "Cliente" if msg["rol"] == "user" else "Operador"
        historial_texto += f"{role}: {msg['txt']}\n"
    
    # 3. PROMPT AVANZADO (Few-Shot)
    prompt = f"""
    ROL: Eres un OPERADOR TURÍSTICO SENIOR en Cusco. Tu trabajo es dar información técnica, precisa y logística.

    INSTRUCCIONES DE COMPORTAMIENTO (FEW-SHOT PROMPTING):
    
    Ejemplo 1 (Incorrecto - Muy vago):
    Usuario: "¿Cómo es la montaña de colores?"
    Bot: "Es muy bonita y alta, tienes que ir."
    
    Ejemplo 2 (CORRECTO - Técnico y Estructurado):
    Usuario: "¿Cómo es la montaña de colores?"
    Bot: "• Ubicación: Vinicunca, a 5,200 m.s.n.m.
    • Dificultad: Media-Alta (Requiere aclimatación previa).
    • Distancia: 5km de caminata aprox.
    • Recomendación: Llevar bastones y ropa térmica."

    Ejemplo 3 (CORRECTO - Precios):
    Usuario: "¿Cuánto cuesta?"
    Bot: "Según nuestra tarifa referencial 2025:
    • Extranjeros: 150 Soles aprox.
    • Nacionales: 100 Soles aprox.
    Incluye: Transporte y desayuno."

    ---
    AHORA TU TURNO:
    
    CONTEXTO TÉCNICO RECUPERADO (Base de Datos):
    {contexto_nuevo if contexto_nuevo else "No hay datos técnicos específicos en el manual para esta consulta."}
    
    HISTORIAL DE CONVERSACIÓN:
    {historial_texto}
    
    PREGUNTA ACTUAL: "{pregunta_actual}"
    
    REGLAS FINALES:
    1. Usa el contexto técnico para responder. Si hay números (precios, horas, altitudes), USALOS.
    2. Si el usuario pregunta algo y NO está en el contexto ni en el historial, di honestamente que no tienes ese dato técnico.
    3. Mantén un tono profesional pero amable.
    """
        
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        res = model.generate_content(prompt)
        return res.text, debug
    except Exception as e:
        return f"Error API: {e}", "Error"

# ==========================================
# 5. INTERFAZ GRÁFICA
# ==========================================
st.markdown(f"<h2 style='text-align: center; color: #FF5722;'>🏔️ TourBot Pro</h2>", unsafe_allow_html=True)
st.caption("Sistema Experto en Turismo - Cusco 2025")

# Panel lateral
with st.sidebar:
    st.header("⚙️ Estado")
    if not df_conocimiento.empty:
        st.success(f"✅ Sistema Online")
        st.info(f"🧠 {len(df_conocimiento)} datos vectorizados")
    else:
        st.error("⚠️ Base de datos desconectada")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = [{"rol": "assistant", "txt": "¡Hola! Soy tu Operador Turístico IA. ¿Necesitas itinerarios, precios o detalles técnicos?"}]

# Chat
for m in st.session_state.mensajes:
    avatar = "🦊" if m["rol"] == "assistant" else "👤"
    with st.chat_message(m["rol"], avatar=avatar):
        st.markdown(m["txt"])

# Input
if preg := st.chat_input("Consulta técnica..."):
    st.session_state.mensajes.append({"rol": "user", "txt": preg})
    with st.chat_message("user", avatar="👤"):
        st.markdown(preg)
        
    with st.spinner("Consultando manuales técnicos..."):
        resp, debug = generar_con_memoria(preg, st.session_state.mensajes)
    
    st.session_state.mensajes.append({"rol": "assistant", "txt": resp})
    with st.chat_message("assistant", avatar="🦊"):
        st.markdown(resp)
        st.caption(f"🔧 Fuente: {debug}")
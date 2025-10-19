import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# ==============================
# TÍTULO Y PRESENTACIÓN DE LA APP
# ==============================
st.set_page_config(page_title="RAG PDF Analyzer", page_icon="📑", layout="wide")

st.title('📑 Generación Aumentada por Recuperación (RAG)')
st.markdown("""
Bienvenido a tu **Asistente Inteligente de Documentos**.  
Aquí podrás cargar un PDF y hacerle preguntas directamente, 
utilizando un modelo de lenguaje de última generación que recupera información del documento.
""")

# Mostrar versión de Python
st.caption(f"⚙️ Versión de Python: {platform.python_version()}")

# ==============================
# IMAGEN DE PORTADA
# ==============================
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350, caption="Analiza documentos PDF con IA")
except Exception as e:
    st.warning(f"No se pudo cargar la imagen de portada: {e}")

# ==============================
# SIDEBAR CON INFORMACIÓN
# ==============================
with st.sidebar:
    st.header("ℹ️ Acerca de la aplicación")
    st.markdown("""
    Esta herramienta permite:
    - 📂 Subir un archivo PDF.  
    - 🧩 Dividirlo en fragmentos.  
    - 🧠 Crear una base de conocimiento con embeddings.  
    - 💬 Hacer preguntas sobre el documento con un **Agente RAG**.  
    """)
    st.info("Recuerda: Necesitas una **clave de OpenAI** para poder usar el modelo.")

# ==============================
# INGRESO DE CLAVE DE API
# ==============================
ke = st.text_input('🔑 Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# ==============================
# SUBIDA DE ARCHIVO PDF
# ==============================
st.subheader("📂 Carga tu documento PDF")
pdf = st.file_uploader("Selecciona el archivo", type="pdf")

# ==============================
# PROCESAMIENTO DEL PDF
# ==============================
if pdf is not None and ke:
    try:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.success(f"✅ Texto extraído del documento: {len(text)} caracteres")

        # División en fragmentos
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.info(f"📑 Documento dividido en **{len(chunks)} fragmentos**")

        # Creación de embeddings y base de conocimiento
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # ==============================
        # INTERFAZ DE PREGUNTAS DEL USUARIO
        # ==============================
        st.subheader("💬 Haz preguntas sobre tu documento")
        user_question = st.text_area("Escribe tu pregunta aquí...")

        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI(temperature=0, model_name="gpt-4o")

            # Cargar el chain de QA
            chain = load_qa_chain(llm, chain_type="stuff")

            # Generar la respuesta
            with st.spinner("Analizando el documento... ⏳"):
                response = chain.run(input_documents=docs, question=user_question)

            # Mostrar resultado
            st.markdown("### 📝 Respuesta:")
            st.markdown(response)

    except Exception as e:
        st.error(f"⚠️ Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("📥 Sube un archivo PDF para comenzar con el análisis")


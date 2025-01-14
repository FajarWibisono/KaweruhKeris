import streamlit as st
import os

# pip install streamlit langchain huggingface_hub sentence-transformers faiss-cpu

from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ─────────────────────────────────────────────────────────────────────────────
# 1. KONFIGURASI API & HALAMAN
# ─────────────────────────────────────────────────────────────────────────────

# Ganti GROQ_API_KEY dengan kunci Anda sendiri, misalnya di secrets.toml
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(
    page_title="KAWERUH KERIS PPBPN",
    page_icon="📓",
    layout="wide"
)

# CSS Styling
st.markdown(
    """
    <style>
        .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
        .user-message { background-color: #f0f2f6; }
        .bot-message { background-color: #e8f0fe; }
    </style>
    """,
    unsafe_allow_html=True
)

# Judul Aplikasi
st.title("📓 KAWERUH KERIS PPBPN")
st.markdown(
    """
    ### Selamat Datang di Asisten Pengetahuan Keris
    Chat Bot ini akan membantu Anda memahami lebih dalam tentang keris dan tosan aji.
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. STATE DAN INISIALISASI
# ─────────────────────────────────────────────────────────────────────────────
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ─────────────────────────────────────────────────────────────────────────────
# 3. PROMPT UNTUK MENJAMIN BAHASA INDONESIA
# ─────────────────────────────────────────────────────────────────────────────
# Prompt ini akan memaksa jawaban selalu dalam Bahasa Indonesia.
PROMPT_INDONESIA = """\
Gunakan informasi konteks berikut untuk menjawab pertanyaan pengguna dalam bahasa Indonesia yang baik dan terstruktur.
Jika Anda tidak menemukan jawaban yang sesuai di dalam konteks, tolong katakan:

"Mohon maaf, saya tidak menemukan informasi tersebut dalam dokumen."

Konteks: {context}
Riwayat Chat: {chat_history}
Pertanyaan: {question}

Jawaban:
"""

INDO_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=PROMPT_INDONESIA
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. FUNGSI INISIALISASI RAG
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def initialize_rag():
    """
    Memuat dokumen PDF dari folder 'documents', memecah menjadi chunk,
    membuat FAISS vector store, dan membentuk ConversationalRetrievalChain.
    """
    try:
        # 4.1 Load Dokumen PDF
        loader = DirectoryLoader("documents", glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        # 4.2 Split Dokumen
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=234)
        texts = text_splitter.split_documents(documents)

        # 4.3 Embedding Berbahasa Indonesia
        # Ganti sesuai preferensi, misal "indobenchmark/indobert-base-p1", dsb.
        embeddings = HuggingFaceEmbeddings(
            model_name="LazarusNLP/all-indo-e5-small-v4",
            model_kwargs={'device': 'cpu'}  
        )

        # 4.4 Membuat Vector Store FAISS
        vectorstore = FAISS.from_documents(texts, embeddings)

        # 4.5 Menginisialisasi LLM (ChatGroq)
        llm = ChatGroq(
            temperature=0.54,
            model_name="groq-merdeka-11B",
            max_tokens=1024
        )

        # 4.6 Membuat Memory untuk menyimpan riwayat percakapan
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )

        # 4.7 Membuat ConversationalRetrievalChain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                'prompt': INDO_PROMPT_TEMPLATE,  # Gunakan template Indonesia
                'output_key': 'answer'
            }
        )

        return chain

    except Exception as e:
        st.error(f"Error during initialization: {str(e)}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 5. INISIALISASI SISTEM
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.chain is None:
    with st.spinner("Memuat sistem..."):
        st.session_state.chain = initialize_rag()

# ─────────────────────────────────────────────────────────────────────────────
# 6. ANTARMUKA CHAT
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.chain:
    # 6.1 Tampilkan riwayat chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 6.2 Chat Input
    prompt = st.chat_input("✍️tuliskan pertanyaan Anda tentang tosan aji disini")
    if prompt:
        # Tambahkan pertanyaan user ke riwayat chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 6.3 Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                try:
                    # Panggil chain
                    result = st.session_state.chain({"question": prompt})
                    # Ambil jawaban
                    answer = result.get('answer', '')
                    st.write(answer)
                    # Tambahkan ke riwayat
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# ─────────────────────────────────────────────────────────────────────────────
# 7. FOOTER & DISCLAIMER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    ---
    **Disclaimer:**
    - Sistem ini menggunakan AI-LLM dan dapat menghasilkan jawaban yang tidak selalu akurat.
    - Ketik: LANJUTKAN JAWABANMU untuk kemungkinan mendapatkan jawaban yang lebih baik dan utuh.
    - Mohon verifikasi informasi penting dengan sumber terpercaya.
    """
)

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Konfigurasi API
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Konfigurasi halaman
st.set_page_config(
    page_title="KAWERUH KERIS PPBPN",
    page_icon="📓",
    layout="wide"
)

# CSS Styling
st.markdown("""
<style>
    .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .user-message { background-color: #f0f2f6; }
    .bot-message { background-color: #e8f0fe; }
</style>
""", unsafe_allow_html=True)

# Judul
st.title("📓 KAWERUH KERIS PPBN")
st.markdown("""
### Selamat Datang di Asisten Pengetahuan Keris
Chat Bot ini akan membantu Anda memahami lebih dalam tentang keris dan tosan aji.
""")

# Inisialisasi session state
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def initialize_rag():
    try:
        # Load documents
        loader = DirectoryLoader('documents', glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create vector store
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Initialize LLM
        llm = ChatGroq(
            temperature=0.54,
            model_name="mixtral-8x7b-32768",
            max_tokens=1024,
        )
      
        # Create memory
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'  # Menambahkan output_key
        )
        
        # Create chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={'output_key': 'answer'}  # Menambahkan output_key
        )
        
        return chain
    except Exception as e:
        st.error(f"Error during initialization: {str(e)}")
        return None

# Initialize system
if st.session_state.chain is None:
    with st.spinner("Memuat sistem..."):
        st.session_state.chain = initialize_rag()

# Chat interface
if st.session_state.chain:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Tuliskan pertanyaan Anda tentang tosan aji:"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                try:
                    result = st.session_state.chain({"question": prompt})
                    answer = result.get('answer', '')  # Menggunakan .get() untuk menghindari KeyError
                    st.write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

# Footer
st.markdown("""
---
**Disclaimer:**
- Sistem ini menggunakan AI dan dapat menghasilkan jawaban yang tidak selalu akurat
- ketik : LANJUTKAN JAWABANMU untuk kemungkinan mendapatkan jawaban yang lebih baik dan utuh.
- Mohon verifikasi informasi penting dengan sumber terpercaya
""")

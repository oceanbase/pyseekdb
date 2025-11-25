import os
import traceback
import streamlit as st
from dotenv import load_dotenv
from embedding_function_factory import create_embedding_function
from seekdb_utils import (
    get_seekdb_client, 
    get_database_stats,
    seekdb_query
)
from llm import get_llm_answer, get_llm_client

load_dotenv()
# Page config
st.set_page_config(
    page_title="seekdb RAG Demo",
    page_icon="https://avatars.githubusercontent.com/u/82347605?s=48&v=4",
    layout="wide"
)

# Configuration
DB_DIR = os.getenv("SEEKDB_DIR", "./seekdb_rag")
DB_NAME = os.getenv("SEEKDB_NAME", "test")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

@st.cache_resource
def init_clients():
    """Initialize and cache all clients."""
    seekdb_client = get_seekdb_client(db_dir=DB_DIR, db_name=DB_NAME)
    
    collection = seekdb_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=create_embedding_function()
    )
    
    llm = get_llm_client()
    
    return llm, collection


# Initialize clients
try:
    llm_client, collection = init_clients()
except Exception as e:
    st.error(f"❌ Failed to initialize: {e}")
    st.stop()

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []

# Header
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 16px;
    }
    .description {
        text-align: center;
        font-size: 15px;
        color: gray;
        margin-bottom: 36px;
    }
    </style>
    <div class="title"><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAD90lEQVR4nOyX3U8cVRiHf+fMzLIzLMvC8FFaCJSPig3UtFisKZW29MM0MSWtvdFYb/0HvDHemXhhYowx6oXpjVdqIMZoqmmzadDElJJgSkEsH6XIhyywX7DL7rI75zVL0VAy6LbsMiGZ52aTM3PeeZ9z3jn7DscuxxawGlvAamwBq7EFrMYWsJqcCWgOyLmKvZGsPqTYxdTLR+ULrx3DVacsnC++Z5zPZnwztiXgVpmjpQbNxxtYW8dB3nG8gZ1WJOQ/uspT1bpRMunHYpZyNcVU4O2Lnqt71KX64ApCsVXE0mMSh1JaALdeIJVWFVNlbQkaasvYAQB5W8V+tVW6+OGPxrUdFzhzSDtzri76xtbTWEbBO4/wy6YCTrcDZU2NKD34LDzV9ay4vhruir1Q9VKougdcdkHTHeltZIAhIr44h4iIa6deQWB0/H8FtksihfAPd6n7i1vryRftL2O1HadQc+IE9rUeg36gGYAjk1iUXi5X+dovFFXdfD1rArMh+sM7BO/1AXH9pyHujZS3NaWeuXSJn7vwKRXuP5SrE89UYC6YnA5Gaboon7nTG75eM8loAsvBFQrMBDA7HcLk7zN0f3CG7vU9oL6JFT2GmvY2Vn/+ZXa68xNy6rVsfQVziWkxM9WTT0LEkVgyNo5zxQmRV+RGQUU5CqtqWGljA8qbmlnFkRYqaXwu07J4Wuizw83wDQxuHDPdAe3Nrz5fqTj7OoAogFUAIl2BBLiYyZxcr/J/YSrAHu1LumYLdjqhJ8XuhazGFrAaW8BqTI9REjuXADMSQfKPDsM/9gDLM1MU8S2wVDzMVhZiwhDEnAUKyZrKVE8Rogvzm+eb/w/kcl8ivlH+0HuTJnp+FtN3esk/PMn2OvaxSq0OZXlVcr2zTGi8hyncyd2yhGAySYJiCCcDhry4mpFAtmGBsX555Luu5L2ubgr3T9FR/SV+sqhdavK8xeraDzOFF268f6v1E78u3KJQMrRxLDcC8fAs/vylh8Zu3uAT3huUGAsbHXs6lXcrPkDTybNM5lq2HrVdgQSWph8iMD5CvsEhaX7gN2PqTh/mByf+eZHo+eJWx0ftw1C4OysZb8JUIP79O+9T8ZdfA2srpax3rSkwxFg8uEyxoB8Rnw9LM/NIxf/t5QyTWHw8Ogj2RKddhMLJEFIiCgYBgsLyJA0uucTs5sy+DbeJ8nHLN/wF/cpjg4LiNBa5Le4Ge8XIcj9NRu/jr9gE+RNLMDsFFZbuiRkMeqz53ZGX2PDOdacFKGb4cdvfZXjnvjV6F3uwnIpnHCS5lrdFnbtbcUmdlVeYJuX0g2dXsutbCVvAamwBq7EFrMYWsJpdL/B3AAAA//9GlWcPF8A4TQAAAABJRU5ErkJggg==" width="40" height="40" style="vertical-align: middle; margin-right: 10px;"> seekdb RAG Demo</div>
    <div class="description">
        This chatbot is built with seekdb database, supported by OpenAI-compatible models.<br>
        Ask questions based on your document collection.
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# Main content
question = st.text_area(
    "Enter your question:",
    placeholder="e.g., What is seekdb?",
    height=100,
    key="question_input"
)

col1, col2 = st.columns([3, 2])
with col1:
    enable_hybrid_search = st.checkbox("Enable hybrid search", value=False, help="Combine vector search with keyword search for better results")
with col2:
    n_results = st.selectbox(
        "Number of relevant results to retrieve:",
        options=[3, 4, 5, 6, 7, 8, 9, 10],
        index=2,
    )

if st.button("Submit", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        try:
            # Search for relevant documents using seekdb_query()
            with st.spinner("🔍 Searching relevant documents..."):
                results = seekdb_query(
                    collection=collection, 
                    query_context=question, 
                    n_results=n_results, 
                    enable_hybrid_search=enable_hybrid_search
                )
            if not results or not results.get("ids") or not results["ids"][0]:
                st.warning("No relevant documents found. Try a different question.")
                st.session_state.results = []
            else:
                # Store results
                st.session_state.results = [
                    {
                        'text': results["documents"][0][i],
                        'similarity': 1.0 / (1.0 + float(results["distances"][0][i])),
                        'source': results["metadatas"][0][i].get('source_file', '') if results["metadatas"][0][i] else '',
                        'distance': float(results["distances"][0][i])
                    }
                    for i in range(len(results["ids"][0]))
                ]
                
                # Generate answer
                context = "\n\n".join([r['text'] for r in st.session_state.results])
                
                with st.spinner("🤖 Generating answer..."):
                    answer = get_llm_answer(llm_client, context, question)

                # Display conversation
                st.divider()
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    st.write(answer)
                
        except Exception as e:
            st.error(f"❌ Error: {e}")
            with st.expander("Detailed error information"):
                st.code(traceback.format_exc(), language='python')
            st.session_state.results = []

# Sidebar
with st.sidebar:
    st.subheader("📊 Database Statistics")
    
    try:
        stats = get_database_stats(collection)
        col1, col2 = st.columns(2)
        col1.metric("Embeddings", stats['total_embeddings'])
        col2.metric("Files", stats['unique_source_files'])
    except Exception as e:
        st.error(f"Unable to load stats: {e}")
    
    st.divider()
    st.subheader("📄 Retrieved Documents")
    
    if st.session_state.results:
        for idx, result in enumerate(st.session_state.results, 1):
            filename = os.path.basename(result['source']) if result['source'] else "Unknown"
            
            with st.expander(f"{idx}. {filename}", expanded=False):
                st.caption(f"L2 Distance: {result['distance']:.4f}")
                preview = result['text'][:200]
                if len(result['text']) > 200:
                    preview += "..."
                st.text(preview)
    else:
        st.info("Results will appear here after submitting a question")

# Footer
st.divider()
st.caption("Built with seekdb • OpenAI-compatible models")

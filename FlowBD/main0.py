from glob import glob
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index_client import SentenceSplitter
import streamlit as st
from hashlib import md5
import json
import time
import os
# os.environ["HF_ENDPOINT"]= "https://hf-mirror.com"

from llama_index.core import VectorStoreIndex, Document, SimpleDirectoryReader, Settings,StorageContext
from typing import List, Any

from llama_index.vector_stores.duckdb import DuckDBVectorStore

from llama_index.core.text_splitter import SentenceSplitter

import logging
from logging.handlers import TimedRotatingFileHandler
st.set_page_config(page_title="液流电池大模型",
                   page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

embed_dim = 1024



if 'need_init_db' not in st.session_state:
    st.session_state.need_init_db = False

def connect_index(recreate=False):

    if recreate:
        os.remove(os.path.join("persist_duckdb", "flowbattery"))
    vector_store = DuckDBVectorStore(
        embed_dim=embed_dim, database_name='flowbattery', persist_dir="persist_duckdb")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)        
        # index = VectorStoreIndex.from_documents([], storage_context=storage_context)

    index = VectorStoreIndex.from_documents([],storage_context=storage_context)
    return index


def insert_nodes_index(index: VectorStoreIndex, nodes):
    index.insert_nodes(nodes)

@st.experimental_dialog(" ")
def msgbox(item):
    st.write(item)
    # reason = st.text_input("Because...")
    if st.button("Ok"):
        st.session_state.need_init_db = False
        # st.session_state.vote = {"item": item, "reason": reason}
        st.rerun()

 
@st.cache_resource()
def init():

    # 创建日志目录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('history', exist_ok=True)

    database_name = os.getenv('DATABASE_NAME', 'flowbattery')
    persist_dir = os.getenv('PERSIST_DIR', 'persist_duckdb')
    ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')
    ollama_model = os.getenv('OLLAMA_MODEL', 'wizardlm2:7b-q5_K_M')
    ollama_embedding_model = os.getenv('OLLAMA_EMBEDDING_MODEL', 'snowflake-arctic-embed:latest')



    # 日志文件路径
    log_file = os.path.join(log_dir, "app.log")

    # 创建一个日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置日志记录级别

    # 创建一个按日期命名的文件处理器，并将其添加到记录器
    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=5)
    file_handler.suffix = "%Y-%m-%d.log"  # 设置分割后日志文件的后缀
    file_handler.setLevel(logging.INFO)   # 设置处理器的日志级别
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将文件处理器添加到日志记录器
    logger.addHandler(file_handler)

    llm = Ollama(model=ollama_model,
                base_url=ollama_base_url, request_timeout=300.0)
    Settings.llm = llm
    # from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.embed_model = OllamaEmbedding(
        model_name=ollama_embedding_model, base_url=ollama_base_url)

    # Settings.embed_model = HuggingFaceEmbedding(
    #     model_name="sentence-transformers/all-mpnet-base-v2"
    # )
    embed_dim = len(Settings.embed_model.get_query_embedding('hello'))
    print('embed_dim: %s', embed_dim)
    files = glob("data/*.pdf",recursive=True)
    if len(files)==0:
        st.session_state.need_init_db = True
    index = connect_index()
    return logger,embed_dim,index



logger, embed_dim,index = init()
if st.session_state.need_init_db:
    msgbox('请将PDF文件copy到data目录下，\n再点【初始化数据】按钮')
# files = glob("data/*.pdf",recursive=True)
# if len(files)==0:
#     msgbox('请将PDF文件copy到data目录下，再点【初始化数据】按钮')
# openai.api_key = st.secrets.openai_key
# openai.base_url = st.secrets.openai_base_url


def has_subdirectories(directory):
    file_list = os.listdir(directory)
    for file in file_list:
        file_path = os.path.join(directory, file)
        if os.path.isdir(file_path):
            return True
    return False


def load_documents(file_path, num_pages=None):
    if num_pages:
        documents = SimpleDirectoryReader(
            input_files=[file_path]).load_data()[:num_pages]
    else:
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    return documents


def create_nodes(documents, chunk_size=2048, chunk_overlap=50):
    node_parser = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = node_parser.get_nodes_from_documents(documents)
    return nodes


table_exists = True

# @st.cache_resource(show_spinner=False)





st.title("液流电池大模型")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about 液流电池!"}
    ]

# @st.cache_resource(show_spinner=False)


def load_data():
    progress_text = "索引中..."
    my_bar = st.progress(0, text=progress_text)
    
    input_dir = "data"
    files = glob(os.path.join('data', '**', '*.pdf'), recursive=True)
    print('files: %s', files)
    index = connect_index(True)
    for percent_complete in range(len(files)):
        document = load_documents(files[percent_complete])
        nodes = create_nodes(document)
        insert_nodes_index(index, nodes)
        # documents.extend(document)
        my_bar.progress(percent_complete + 1, text=progress_text)
    my_bar.empty()


@st.experimental_dialog("Confirmation")
def confirm(item):
    st.write(item)
    # reason = st.text_input("Because...")
    col1,col2 = st.columns(2)
    with col1:
        if st.button("Submit"):
            # st.session_state.vote = {"item": item, "reason": reason}
            load_data()
            st.rerun()
    with col2:
        if st.button("Cancel"):
            st.rerun()
    
def main():
    with st.sidebar:
        
        if st.button('初始化数据'):
            # st.warning("你确定重建索引吗?", icon="⚠️")
            # st.session_state.rebuild = True
            # print('你确定重建索引吗?')
            # print(st.button('hello'))
            confirm("你确定重建索引吗?")
            # if st.button("确定"):
                
            #         # print('重建索引中，请稍等')
            #     load_data()
            #     st.session_state.rebuild = False
            # else:
            #     print('give up.')

        uploaded_file = st.sidebar.file_uploader(
            "上传PDF",
            type="pdf",
            label_visibility="hidden",
            # accept_multiple_files=True
        )
        if uploaded_file is not None:
            # print(uploaded_file)
            data = uploaded_file.read()
            md5_hash = md5(data).hexdigest()
            if not os.path.exists(os.path.join("data", md5_hash)):
                os.mkdir(os.path.join("data", md5_hash))
                pdf_path = os.path.join("data", md5_hash, uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(data)

                # from tempfile import tempdir
                # file = NamedTemporaryFile(suffix=".pdf")
                try:
                    with st.spinner(text="索引中，请稍等"):
                        document = load_documents(pdf_path)
                        nodes = create_nodes(document)
                        insert_nodes_index(index, nodes)
                        st.success("索引成功!")
                        # time.sleep(2)

                except:
                    st.error("索引失败")
                    os.remove(pdf_path)
                    os.remove(os.path.join("data", md5_hash))
                uploaded_file = None
            # else:
            #     st.info("该PDF已经上传过了!")
        if st.button('导出聊天记录'):
            # index.save_to_disk("persist_duckdb")
            # json.dumps(st.session_state.messages)
            json_data = json.dumps(st.session_state.messages,
                                ensure_ascii=False, indent=4)
            # md5_hash = md5(json_data).hexdigest()
            # 将JSON格式的字符串写入文件
            import time
            name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).replace(':', '-')
            with open(os.path.join("history", f"{name}.json"), 'w', encoding='utf-8') as json_file:
                json_file.write(json_data)
            st.session_state.messages = [
                {"role": "assistant", "content": "Ask me a question about 液流电池!"}
            ]
            st.success("导出成功!")
        if st.button('新建聊天记录'):
            st.session_state.messages = [
                {"role": "assistant", "content": "Ask me a question about 液流电池!"}
            ]
    # if st.session_state.rebuild:
    #     with st.spinner(text="重建索引中，请稍等"):
    #         load_data(True)
    #         time.sleep(10)

    if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="condense_question", verbose=True)

    # Prompt for user input and save to chat history
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:  # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                # Add response to message history
                st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
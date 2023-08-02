# =========================
#  Module: Vector DB Build
# =========================
import box
import yaml
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings


# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Load docs
def load_docs(directory: str):
    """
    Load documents from the given directory.
    """
    loader = DirectoryLoader(directory)
    documents = loader.load()

    return documents

#Split docs
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    """
    Split the documents into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP)
    docs = text_splitter.split_documents(documents)

    return docs


# Build vector database

def run_db_build_domain(dataPath):
    documents =load_docs(dataPath)
    texts = split_docs(documents)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    

    vectorstore_domain = FAISS.from_documents(texts, embeddings)
    vectorstore_domain.save_local(cfg.DB_FAISS_PATH_DOMAIN)
    

def run_db_build_agent(file):
    loader=PyPDFLoader(file)
    documents = loader.load()
    texts = split_docs(documents)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    

    vectorstore_agent = FAISS.from_documents(texts, embeddings)
    vectorstore_agent.save_local(cfg.DB_FAISS_PATH_AGENT)
    

# if __name__ == "__main__":
#     run_db_build()

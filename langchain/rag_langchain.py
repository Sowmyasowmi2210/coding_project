
import openai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# 1. Azure OpenAI configuration
openai.api_type = "azure"
openai.api_base = "https://wusazeoai02.openai.azure.com/"
openai.api_key = "29a42536131c4e288710723d3201d737"
openai.api_version = "2023-09-15-preview"
DEPLOYMENT_NAME = "gpt-35-turbo"  # your deployment/model

# 2. Load the document
path = "Amazon.PDF"
print("Exists:", os.path.exists(path))
loader = PyMuPDFLoader(path)
documents = loader.load()

# 3. Split into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
print(f"Loaded {len(docs)} chunks")

# 4. Embed and build FAISS vector store
embeddings = OpenAIEmbeddings(deployment=DEPLOYMENT_NAME)
vectorstore = FAISS.from_documents(docs, embeddings)

# 5. Initialize retrieval-based QA model
llm = ChatOpenAI(deployment=DEPLOYMENT_NAME)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 6. Use the RAG model
query = "Explain the main topic of this PDF"
result = qa_chain({"query": query})

print("\nAnswer:\n", result["result"])
print("\nSource Chunks Used:\n")
for doc in result["source_documents"]:
    snippet = doc.page_content[:200].replace("\n", " ")
    print(f"- {snippet} ...")
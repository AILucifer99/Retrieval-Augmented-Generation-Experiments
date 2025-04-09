import os
from authorization import auth

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = auth.OPENAI_API_KEY

fileName = "Jungle-Book-PG.pdf"
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "Data", "{}.pdf".format(fileName.split(".")[0]))
print("\nWorking with File :- {}\n".format(file_path))

vector_db_dir = os.path.join(current_dir, "Vector-DB", "chroma_db_jungle_book")

if not os.path.exists(vector_db_dir) :
    print("Vector DB Doesn't Exists. Creating the vector database for the file, please wait.....\n")

    if not os.path.exists(file_path) :
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the name or the location."
        )
    
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048, 
        chunk_overlap=256,
    )
    docs = text_splitter.split_documents(documents)

    print("---------- Document Chunks Information ----------")
    print(f"\nNumber of splitted documents :- {len(docs)}\n")
    print(f"Sample Chunk :- \n{docs[0].page_content}\n")

    print("---------- Creating the embeddings and Vectorstore ----------")
    print("Generating Embeddings.......")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    print("Embeddings completed.......")

    print("Creating Vectorstore in ChromaDB.....")
    db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=vector_db_dir,
    )
    print("Vectorstore added successfully......\n")
    
else :
    print(f"VectorDB already created and located at :- {vector_db_dir}")



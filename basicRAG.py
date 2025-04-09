import os
from authorization import auth

from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
# import warnings
# warnings.filterwarnings("ignore")


os.environ["OPENAI_API_KEY"] = auth.OPENAI_API_KEY

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_jungle_book")


# Define the embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Load the existing vector store with the embedding function
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)


# Define the user's question
query = "Who is Mowgli's brother ?"
retrieverCategory = "sst"


# Retrieve relevant documents based on the query
if retrieverCategory == "sst" :
    print("\nUsing Retriever :- {}".format(retrieverCategory))
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5, 
            "score_threshold": 0.3,
        },
    )

    relevant_docs = retriever.invoke(query)

    # Display the relevant results with metadata
    print("\n---------- Relevant Documents ----------")
    print("Question :- {}\n".format(query))
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


elif retrieverCategory == "mmr" :
    print("\nUsing Retriever :- {}".format(retrieverCategory))
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k" : 10,
            "fetch_k" : 15,
            "lambda_mult" : 0.75,
        },
    )

    relevant_docs = retriever.invoke(query)

    # Display the relevant results with metadata
    print("\n---------- Relevant Documents ----------")
    print("Question :- {}\n".format(query))
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
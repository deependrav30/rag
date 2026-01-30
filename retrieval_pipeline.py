from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persist_directory = 'db/chroma_db'

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
    )
    
#query = "Which island does SpaceX lease for its launches in the Pacific?"
query= "What was NVIDIA'S frst graphcs acelrtar called?"

retriver = db.as_retriever(
    search_kwargs={
            "k": 5,
        }
    )

# retriver = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#             "k": 5,
#             "score_threshold": 0.3
#         }
#     )

relevant_docs = retriver.invoke(query)

print(f"User Query: {query}\n")
#Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}: \n{doc.page_content}")
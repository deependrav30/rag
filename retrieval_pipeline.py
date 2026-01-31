from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

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

combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from the documents above. If you cannot find the answer in the documents, please respond with "I don't know"."""

model = ChatOpenAI(model="gpt-4o", temperature=0)

messages = [
    SystemMessage(content="You are a helpful AI assistant that provides accurate information based on the provided documents."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)

print("\n --- Generated Response ---")
print("Content Only:")
print(result.content)
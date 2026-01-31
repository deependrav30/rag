from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

persist_directory = 'db/chroma_db'
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embedding_model
    )

model = ChatOpenAI(model="gpt-4o", temperature=0)

chat_history = []

def ask_question(user_question):
    print(f"User Question: {user_question}\n")
    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone amnd searchable. Just return rewriten the question."),
        ] + chat_history + [
            HumanMessage(content=f"New Question: {user_question}")
        ]

        result = model.invoke(messages)
        seach_question = result.content.strip()
        print(f"Rewritten Search Question: {seach_question}\n")
    else:
        seach_question = user_question


    retriver = db.as_retriever(
        search_kwargs={
                "k": 3,
            }
        )
    docs = retriver.invoke(seach_question)    

    print("--- Retrieving Relevant Documents ---")
    print(f"Fund {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f" Doc {i} Preview: {preview}...")

    combined_input = f"""Based on the following documents, please answer this question: {seach_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from the documents above. If you cannot find the answer in the documents, please respond with "I don't know".
    """

    messages = [
        SystemMessage(content="You are a helpful AI assistant that provides accurate information based on the provided documents."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer

def start_chat():
    print("Welcome to the History-Aware QA System! Type 'exit' to quit.")
    while True:
        user_question = input("\nPlease enter your question: ")
        if user_question.lower() == 'exit':
            print("Exiting the chat. Goodbye!")
            break
        ask_question(user_question)


if __name__ == "__main__":
    start_chat()
from langchain_chroma import Chroma
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM
#from langchain_huggingface import HuggingFacePipeline
#from transformers import pipeline
#from langchain_google_genai import ChatGoogleGenerativeAI
"""from langchain_openai import ChatOpenAI"""
#from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

persistent_directory="db/chroma_db"

#Load embeddings and vector store
#embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

db=Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

#search for relevant documents
retriever=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k":4,
        "score_threshold":0.5
    }
)

query="how does one aquire tesla shares?"

relevant_docs=retriever.invoke(query)

print(f"User query: {query}")

#Display results
print("---Context---")

for i,doc in enumerate (relevant_docs,1):
    print(f"Document {i}:\n{doc.page_content}\n")

#Combine the query and the relevant documents content
combined_input=f"""Based on the following documents, pls answer this question: {query}

Documents:
{chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If u can't find the answer in the document say, "I dont have enough information to answer that question based on the document u gave."

"""
# Ollama LLM
model = OllamaLLM(model="llama3.2:3b")

"""#Create a ChatOpenAI model
pipe = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200
)
model=HuggingFacePipeline(pipeline=pipe)


#define the messages for the model
messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]"""

#Invoke the model with the combined input
result=model.invoke(combined_input)

#Display the full result and content only
print("\n---Generated response---")

print("Content only:")
print(result)
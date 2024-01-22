
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from redundant_filter_retriever import RedundantFilterRetriever
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()

db = Chroma(
  persist_directory="emb",
  embedding_function=embeddings,
)

retriever = RedundantFilterRetriever(
  embeddings=embeddings,
  chroma=db
)

chain = RetrievalQA.from_chain_type(
  llm=chat,
  retriever=retriever,
  chain_type="stuff"
)

result = chain.invoke("What is an interesting fact about the english language?")

print(result["result"])
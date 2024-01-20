from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory, ConversationSummaryMemory

from dotenv import load_dotenv
load_dotenv()

chat = ChatOpenAI(verbose=True)

# memory = ConversationBufferMemory(
#   memory_key="messages", 
#   return_messages=True,
#   chat_memory=FileChatMessageHistory("messages.json")
# )

memory = ConversationSummaryMemory(
  memory_key="messages", 
  return_messages=True,
  llm=chat
)

prompt = ChatPromptTemplate(
  input_variables=["content", "messages"],
  messages=[
    MessagesPlaceholder(variable_name="messages"),
    HumanMessagePromptTemplate.from_template("{content}")
  ]
)

chain = LLMChain(
  llm=chat,
  prompt=prompt,
  memory=memory,
  verbose=True
)

while True:
  content = input(">> ")
  result = chain.invoke({"content": content})
  print(result["text"])

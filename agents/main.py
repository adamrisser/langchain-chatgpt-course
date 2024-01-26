from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from tools.sql import run_query_tool, describe_tables_tool, list_tables
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler

from dotenv import load_dotenv
load_dotenv()

tables = list_tables()

handler = ChatModelStartHandler()
chat = ChatOpenAI(
  callbacks=[handler]
)
prompt = ChatPromptTemplate(
  messages=[
    SystemMessage(
      content=(
        "You are an AI that has access to a SQLite database."
        f"The database has tables if: {tables}\n"
        "Do not make any assumptions about what tables exist "
        "or what columns exists. Instead, use the 'describe_tables' function"
      )
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
  ]
)
   

memory = ConversationBufferMemory(
  memory_key="chat_history",
  return_messages=True
)

tools = [run_query_tool, describe_tables_tool, write_report_tool]

agent = create_openai_functions_agent(
  llm=chat,
  prompt=prompt,
  tools=tools
)

agent_executor = AgentExecutor(
  agent=agent,
  # verbose=True,
  tools=tools,
  memory=memory
)

# agent_executor.invoke({ "input": "How many users have a shipping address?"})
# agent_executor.invoke({ "input": "Summarize the top 5 more popular products. Write the results to a report file."})
agent_executor.invoke({ "input": "How many orders are there? Write the result to an html report."})
agent_executor.invoke({ "input": "Repeat the exact same process for users."})
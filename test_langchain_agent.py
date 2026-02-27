from app.agents.langchain_agent import create_langchain_agent

agent = create_langchain_agent()

res = agent.invoke(
    {"messages": [{"role": "user", "content": "What is 2+2?"}]}
)

print(res)
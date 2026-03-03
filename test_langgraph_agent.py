from app.agents.langgraph_agent import create_langgraph_agent

def main():
    agent = create_langgraph_agent()
    
    result = agent.invoke(
        {
            "question": "2+2"
        }
    )
    print(result)
if __name__ == "__main__":
    main()
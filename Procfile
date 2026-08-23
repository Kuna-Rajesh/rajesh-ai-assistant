web: sh -c "python agent/agent.py start & uvicorn agent.token_server:app --host 0.0.0.0 --port $PORT"
worker: python agent/agent.py start

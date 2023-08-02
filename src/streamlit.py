'''
=================================================
      Module: Creating a StreamLit application
=================================================
'''
import box
import yaml
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
from src.llm import build_llm
from dotenv import find_dotenv, load_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def set_streamlit_agent(llm):
    
    tools = load_tools(["ddg-search"])
    agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                response = agent.run(prompt, callbacks=[st_callback])
                st.write(response)

    

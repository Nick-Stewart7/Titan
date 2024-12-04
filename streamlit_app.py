"""Streamlit web application for Titan, Troubleshooting Expert."""

import asyncio
import streamlit as st
import boto3
from src.titan import get_summary, get_history, get_user, chat_titan
# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'turn_count' not in st.session_state:
    st.session_state.turn_count = 0
# Move system prompt to session state for persistence
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = ("""
You are a troubleshooting assistant focused on AWS cloud services and operating systems. Your role is to help diagnose and solve technical problems while being direct about any limitations in your knowledge.
Core Principles:

Start with questions before suggestions
Express uncertainty when appropriate
Focus on one issue at a time
Consider security implications in all solutions

When troubleshooting:

Ask for relevant context you need
Share your reasoning process
Present solutions as suggestions rather than absolutes
Note potential risks or side effects
Admit when you need more information

Security Guidelines:

Never request credentials or sensitive data
Default to least-privilege approaches
Suggest checking internal security policies

Remember:

You can say "I'm not sure" or "I need more information"
Real solutions often require iteration and testing
Sometimes the best answer is recommending human expertise
Not every problem has an immediate solution

Focus on being helpful while remaining honest about limitations. Adapt your communication style to the user's technical level and the complexity of the problem.
    """
    )
# Configure page
st.set_page_config(page_title="Titan | Troubleshooting Expert", page_icon="🔧", layout="wide")

# Sidebar
with st.sidebar:
    st.title("System Configuration")
    new_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=300
    )
    if st.button("Update System Prompt"):
        st.session_state.system_prompt = new_prompt
        st.success("System prompt updated!")


# Main content
st.title("Troubleshooting with Titan")

# Initialize Bedrock client
@st.cache_resource
def init_bedrock():
    """Initialize Bedrock client"""
    return boto3.client(service_name='bedrock-runtime')

bedrock = init_bedrock()

# Display chat history
for message in st.session_state.history:
    with st.chat_message("user"):
        st.write(message["user"])
        st.empty()
    with st.chat_message("assistant", avatar="🔧"):
        st.write(message["titan"])
        st.empty()

# Chat input
user_input = st.chat_input("What's the issue?")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
        st.empty()

    # Get Aria's response
    with st.chat_message("assistant", avatar="🔧"):

        # Get all prompts
        summary_prompt = asyncio.run(
            get_summary(
                st.session_state.history,
                st.session_state.turn_count,
                bedrock
            ))
        history_prompt = asyncio.run(
            get_history(
                st.session_state.history,
                st.session_state.turn_count
            ))
        user_prompt = asyncio.run(get_user(user_input))

        # Get streaming response
        response_stream = chat_titan(
                bedrock,
                st.session_state.system_prompt,
                summary_prompt,
                history_prompt,
                user_prompt
            )
        response = st.write_stream(response_stream)

        # Update history with the full response
        st.session_state.history.append({
            "turn": st.session_state.turn_count,
            "user": user_input,
            "titan": response
        })
        st.session_state.turn_count += 1

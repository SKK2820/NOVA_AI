
import os
import streamlit as st
from dotenv import load_dotenv

# Load local .env when running on your computer
load_dotenv()

# Read Streamlit Cloud secrets when deployed
try:
    cloud_secrets = st.secrets
except Exception:
    cloud_secrets = {}

# Prefer local environment variables, then Streamlit secrets
API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or cloud_secrets.get("GEMINI_API_KEY")
)

MODEL = (
    os.getenv("GEMINI_MODEL")
    or cloud_secrets.get("GEMINI_MODEL")
)

# Validate configuration
if not API_KEY:
    raise ValueError("NOVA ERROR: Gemini API key is missing!")

if not MODEL or MODEL == "YOUR_FREE_MODEL_ID":
    raise ValueError("NOVA ERROR: Please configure a model ID!")

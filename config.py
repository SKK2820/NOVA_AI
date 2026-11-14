import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_setting(name):
    value = os.getenv(name)

    if value:
        return value

    try:
        return st.secrets.get(name)
    except Exception:
        return None


API_KEY = get_setting("GEMINI_API_KEY")
MODEL = get_setting("GEMINI_MODEL")

if not API_KEY:
    raise ValueError("NOVA ERROR: Gemini API key is missing!")

if not MODEL or MODEL == "YOUR_FREE_MODEL_ID":
    raise ValueError("NOVA ERROR: Please configure a model ID!")

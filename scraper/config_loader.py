"""Config loader that works both locally (config.json) and on Streamlit Cloud (st.secrets)."""
import json
import os
from typing import Optional


def load_config(filepath: str = "config.json") -> dict:
    """Load config from config.json if it exists, otherwise from st.secrets.

    Lets the app run locally with a config.json on disk, OR on Streamlit Cloud
    where secrets are injected via the Cloud UI as TOML.
    """
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            return json.load(file)

    # Fallback: Streamlit Cloud secrets
    try:
        import streamlit as st  # Lazy import so non-Streamlit callers don't need it
        # st.secrets behaves like a nested dict; convert to plain dict
        return _to_plain_dict(st.secrets)
    except Exception as e:
        raise FileNotFoundError(
            f"Config not found at {filepath} and no Streamlit secrets available: {e}"
        )


def _to_plain_dict(obj) -> dict:
    """Recursively convert a Streamlit secrets object (or AttrDict) to a plain dict."""
    if hasattr(obj, 'to_dict'):
        return {k: _to_plain_dict(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_dict(v) for v in obj]
    return obj

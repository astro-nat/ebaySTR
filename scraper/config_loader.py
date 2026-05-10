"""Config loader that works both locally (config.json) and on Streamlit Cloud (st.secrets)."""
import json
import os
from typing import Optional, Tuple

# Per-process cache for parsed config. Keyed by (filepath, mtime) so
# edits to config.json picked up automatically on the next call without
# requiring a restart. Without this, the app's ~10 load_config() call
# sites each re-read + re-parse config.json on every Streamlit rerun —
# adds up to noticeable disk + JSON-parse overhead on every interaction.
_CONFIG_CACHE: dict = {'key': None, 'value': None}


def _config_cache_key(filepath: str) -> Tuple[str, float]:
    """Build a (filepath, mtime) cache key. Returns mtime=-1 when the
    file is missing — Streamlit-Cloud secrets fallback path handles
    that branch and we don't try to mtime-cache it.
    """
    try:
        return (filepath, os.path.getmtime(filepath))
    except OSError:
        return (filepath, -1.0)


def load_config(filepath: str = "config.json") -> dict:
    """Load config from config.json if it exists, otherwise from st.secrets.

    Lets the app run locally with a config.json on disk, OR on Streamlit Cloud
    where secrets are injected via the Cloud UI as TOML.

    Cached per (filepath, mtime) — mid-session edits to config.json
    invalidate automatically on the next call.
    """
    key = _config_cache_key(filepath)
    if _CONFIG_CACHE['key'] == key and _CONFIG_CACHE['value'] is not None:
        return _CONFIG_CACHE['value']

    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            cfg = json.load(file)
        _CONFIG_CACHE['key'] = key
        _CONFIG_CACHE['value'] = cfg
        return cfg

    # Fallback: Streamlit Cloud secrets
    try:
        import streamlit as st  # Lazy import so non-Streamlit callers don't need it
        # st.secrets behaves like a nested dict; convert to plain dict
        cfg = _to_plain_dict(st.secrets)
        # Cache the secrets-fallback result too — secrets don't change
        # mid-session on Streamlit Cloud, so this is safe.
        _CONFIG_CACHE['key'] = key
        _CONFIG_CACHE['value'] = cfg
        return cfg
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

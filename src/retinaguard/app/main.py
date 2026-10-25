"""Streamlit entry point. Pages are added in Phase 6."""

import streamlit as st

from retinaguard import DISCLAIMER, __version__

st.set_page_config(page_title="RetinaGuard", layout="wide")
st.title("RetinaGuard")
st.warning(DISCLAIMER)
st.caption(f"version {__version__}")

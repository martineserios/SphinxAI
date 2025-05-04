import duckdb as ddb
import streamlit as st

from sphinx_ai.config import (DB_LOCATION, INPUT_DIR, OUTPUT_DIR,
                              OUTPUT_TRANSCRIPTIONS_BUCKET_NAME)

# Streamlit configs
st.set_page_config(page_title="SphinxAI", page_icon=":material/robot_2:")

# Pages configs
page_cv_models = st.Page("pages/cv_models.py", title="CV Models", icon=":material/ar_on_you:")
page_transcription = st.Page("pages/transcription.py", title="Transcription", icon=":material/speech_to_text:")
page_load_data = st.Page("pages/load_data.py", title="Load Pupils Data", icon=":material/upload:")
page_visualize = st.Page("pages/visualize.py", title="Visualize", icon=":material/dashboard:")

pg = st.navigation([page_cv_models, page_transcription, page_load_data, page_visualize])
pg.run()

# Connections
@st.cache_resource
def connect_to_database(db_path: str, read_only: bool):
    return ddb.connect(database=db_path, read_only=False)

db_conn = connect_to_database(DB_LOCATION, False)



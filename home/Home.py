import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import streamlit.components.v1 as components
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import plotly.express as px
from copyright import display_custom_license
import numpy as np
import plotly.express as px
import time
from sidebar_content import sidebar_content
import base64
import json
from urllib.parse import quote, unquote
import os
import uuid

st.set_page_config(layout="wide", 
                   page_title='OpenAlex DOI Search Tool',
                   page_icon="https://openalex.org/img/openalex-logo-icon-black-and-white.ea51cede.png")
pd.set_option('display.max_colwidth', None)
sidebar_content()

# Create directory to store session data if it doesn't exist
os.makedirs("sessions", exist_ok=True)

st.title('OpenAlex DOI Search Tool', anchor=False)

# Handle session from query param
session_id = st.query_params.get("session_id", None)
df_dois = None

if session_id:
    filepath = f"sessions/{session_id}.json"
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            dois = json.load(f)
        df_dois = pd.DataFrame(dois, columns=["doi_submitted"])
        st.success("Loaded session data from shared link.")
    else:
        st.error("Session not found. It may have expired or been deleted.")
else:
    radio = st.radio('Select an option', ['Insert DOIs', 'Upload a file with DOIs'])
    if radio == 'Insert DOIs':
        st.write('Insert DOIs (max 700). One DOI per line:')
        dois = st.text_area('DOIs', placeholder="10.1234/example\n10.5678/another")
        doi_list = [doi.strip() for doi in dois.split('\n') if doi.strip()]
        df_dois = pd.DataFrame(doi_list, columns=["doi_submitted"])
    else:
        file = st.file_uploader("Upload a CSV with DOIs", type="csv")
        if file:
            try:
                df = pd.read_csv(file)
                valid_cols = ['doi', 'DOI', 'dois', 'DOIs', 'Hyperlinked DOI']
                col = next((c for c in valid_cols if c in df.columns), None)
                if col:
                    df_dois = df[[col]].dropna()
                    df_dois.columns = ['doi_submitted']
                else:
                    st.error("CSV must contain a valid DOI column.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

if df_dois is not None:
    df_dois['doi_submitted'] = df_dois['doi_submitted'].str.replace('https://doi.org/', '', regex=False)
    df_dois = df_dois.drop_duplicates().reset_index(drop=True)

    if len(df_dois) > 700:
        st.error("Maximum 700 DOIs allowed.")
    else:
        st.info(f"{len(df_dois)} DOIs ready for search.")
        st.dataframe(df_dois)

        if st.button("🔗 Generate Shareable Link"):
            # Generate and save session
            new_id = str(uuid.uuid4())
            with open(f"sessions/{new_id}.json", 'w') as f:
                json.dump(df_dois['doi_submitted'].tolist(), f)

            # Update query param in URL
            st.query_params.session_id = new_id

            st.success("Link generated! You can copy and share this:")
            st.code(f"{st.request.url}?session_id={new_id}", language="text")

        if st.button("🔍 Search DOIs"):
            with st.status("Searching DOIs...", expanded=True) as status:
                def batch_dois(dois, size=20):
                    for i in range(0, len(dois), size):
                        yield dois[i:i+size]

                results = []
                for batch in batch_dois(df_dois['doi_submitted'].tolist()):
                    q = '|'.join(batch)
                    url = f"https://api.openalex.org/works?filter=doi:{q}&mailto=you@example.com"
                    r = requests.get(url)
                    if r.status_code == 200:
                        results.extend(r.json().get("results", []))
                    time.sleep(1)

                if results:
                    df_results = pd.json_normalize(results)
                    st.success(f"{len(df_results)} results found.")
                    st.dataframe(df_results)
                else:
                    st.warning("No results found.")
                    
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

# ========== Helper functions ==========

def encode_dois(dois):
    json_str = json.dumps(dois)
    b64_bytes = base64.urlsafe_b64encode(json_str.encode('utf-8'))
    return b64_bytes.decode('utf-8')

def decode_dois(b64_str):
    try:
        json_str = base64.urlsafe_b64decode(b64_str.encode('utf-8')).decode('utf-8')
        return json.loads(json_str)
    except Exception:
        st.error("Failed to decode DOIs from URL.")
        return []

# ========== Streamlit App Config ==========

st.set_page_config(
    layout="wide",
    page_title="OpenAlex DOI Search Tool",
    page_icon="https://openalex.org/img/openalex-logo-icon-black-and-white.ea51cede.png",
)

pd.set_option('display.max_colwidth', None)
sidebar_content()

st.title('OpenAlex DOI Search Tool', anchor=False)

# ========== Load from Query Params ==========

query_param = st.query_params.get("dois", None)
df_dois = None

if query_param:
    decoded = decode_dois(query_param)
    if decoded:
        df_dois = pd.DataFrame(decoded, columns=["doi_submitted"])
        st.success("Loaded DOIs from shared link.")

# ========== UI: Input DOIs ==========

if df_dois is None:
    radio = st.radio('Select an option', ['Insert DOIs', 'Upload a file with DOIs'])

    if radio == 'Insert DOIs':
        st.write("Please enter DOIs (e.g. 10.1234/abc). One per line. Max 700.")
        text_input = st.text_area("Enter DOIs")
        doi_list = [d.strip() for d in text_input.splitlines() if d.strip()]
        if doi_list:
            df_dois = pd.DataFrame(doi_list, columns=["doi_submitted"])
    else:
        file = st.file_uploader("Upload CSV with DOIs", type="csv")
        if file:
            try:
                df = pd.read_csv(file)
                valid_cols = ['doi', 'DOI', 'dois', 'DOIs', 'Hyperlinked DOI']
                col = next((c for c in valid_cols if c in df.columns), None)
                if col:
                    df_dois = df[[col]].dropna()
                    df_dois.columns = ['doi_submitted']
                else:
                    st.error("No valid DOI column found.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

# ========== Clean and Display DOIs ==========

if df_dois is not None:
    df_dois['doi_submitted'] = df_dois['doi_submitted'].str.replace('https://doi.org/', '', regex=False)
    df_dois = df_dois.drop_duplicates().reset_index(drop=True)

    if len(df_dois) > 700:
        st.error("Max 700 DOIs allowed.")
    else:
        st.info(f"{len(df_dois)} unique DOIs entered.")
        with st.expander("See the DOIs you entered"):
            df_dois.index += 1
            st.dataframe(df_dois)

        # Generate shareable link
        if st.button("🔗 Generate Shareable Link"):
            encoded = encode_dois(df_dois['doi_submitted'].tolist())
            st.query_params.dois = encoded  # update URL
            st.success("Shareable link created:")
            st.code(f"{st.request.url}?dois={encoded}", language="text")

        # Search DOIs
        if st.button("🔍 Search DOIs", type="primary"):
            with st.status("Searching DOIs in OpenAlex...", expanded=True) as status:
                def batch_dois(dois, batch_size=20):
                    for i in range(0, len(dois), batch_size):
                        yield dois[i:i+batch_size]

                start_time = time.time()
                all_results = []

                for batch in batch_dois(df_dois['doi_submitted'].tolist()):
                    filter_str = '|'.join(batch)
                    url = f"https://api.openalex.org/works?filter=doi:{filter_str}&mailto=y.ozkan@imperial.ac.uk"
                    res = requests.get(url)
                    if res.status_code == 200:
                        all_results.extend(res.json().get("results", []))
                    time.sleep(1)

                if not all_results:
                    st.warning("No results found.")
                    return

                results_df = pd.json_normalize(all_results)
                results_df['doi_submitted'] = results_df['doi'].str.replace('https://doi.org/', '', regex=False)
                merged_df = df_dois.merge(results_df, on='doi_submitted', how='left')
                merged_df = merged_df.dropna(subset=['id'])
                num_results = len(merged_df)

                st.success(f"{num_results} result(s) found.")
                with st.expander("All Results Table"):
                    st.dataframe(merged_df)

                # OA summary
                oa_counts = merged_df['open_access.oa_status'].value_counts(dropna=False).reset_index()
                oa_counts.columns = ['OA Status', '# Outputs']
                st.subheader("Open Access Summary")
                st.dataframe(oa_counts)

                end_time = time.time()
                t = time.strftime("%M:%S", time.gmtime(end_time - start_time))
                status.update(label=f"Search complete in {t} min.", state="complete", expanded=False)

else:
    st.info("Please enter or upload DOIs to start.")
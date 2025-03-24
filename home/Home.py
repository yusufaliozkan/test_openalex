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

st.set_page_config(layout = "wide", 
                    page_title='OpenAlex DOI Search Tool',
                    page_icon="https://openalex.org/img/openalex-logo-icon-black-and-white.ea51cede.png",
                    initial_sidebar_state="auto") 
pd.set_option('display.max_colwidth', None)

sidebar_content() 

st.title('OpenAlex DOI Search Tool', anchor=False)

df_dois = None

radio = st.radio('Select an option', ['Insert DOIs', 'Upload a file with DOIs'])
if radio == 'Insert DOIs':
    st.write('Please insert [DOIs](https://www.doi.org/) (commencing "10.") in separarate rows.')
    dois = st.text_area(
        'Type or paste in one DOI per line in this box, then press Ctrl+Enter.', 
        help='DOIs will be without a hyperlink such as 10.1136/bmjgh-2023-013696',
        placeholder=''' e.g.
        10.1136/bmjgh-2023-013696
        10.1097/jac.0b013e31822cbdfd
        '''
        )
    # Split the input text into individual DOIs based on newline character
    doi_list = dois.split('\n')
    
    # Remove any empty strings that may result from extra newlines
    doi_list = [doi.strip() for doi in doi_list if doi.strip()]
    
    # Create a DataFrame
    df_dois = pd.DataFrame(doi_list, columns=["doi"])
else:
    st.write('Please upload and submit a .csv file of [DOIs](https://www.doi.org/) (commencing “10.") in separate rows.')
    st.warning('The title of the column containing DOIs should be one of the followings: doi, DOI, dois, DOIs, Hyperlinked DOI. Otherwise the tool will not identify DOIs.')
    dois = st.file_uploader("Choose a CSV file", type="csv")

    if dois is not None:
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(dois)
        
        # List of possible DOI column names
        doi_columns = ['doi', 'DOI', 'dois', 'DOIs', 'Hyperlinked DOI']
        
        # Find the first matching DOI column
        doi_column = None
        for col in doi_columns:
            if col in df.columns:
                doi_column = col
                break
        
        if doi_column:
            # Create a DataFrame with DOIs only
            df_dois = df[[doi_column]]
            df_dois.columns = ['doi']  # Standardize column name to 'DOI'
        
        else:
            st.error('''
            No DOI column in the file.
            
            Make sure that the column listing DOIs have one of the following alternative names:
            'doi', 'DOI', 'dois', 'DOIs', 'Hyperlinked DOI'
            ''')
            st.stop()
    else:
        st.write("Please upload a CSV file containing DOIs.")
if df_dois is not None and len(df_dois) > 500:
    st.error('Please enter 500 or fewer DOIs')
else:
    if dois:
        df_dois['doi'] = df_dois['doi'].str.replace('https://doi.org/', '')
        df_dois = df_dois.drop_duplicates().reset_index(drop=True)
        no_dois = len(df_dois)
        if len(df_dois) > 100:
            st.toast('You entered over 100 DOIs. It may take some time to retrieve results. Please wait.')
        if len(df_dois) >100:
            st.warning('You entered over 100 DOIs. It may take some time to retrieve results.')
        st.info(f'You entered {no_dois} unique DOIs')
        with st.expander(f'See the DOIs you entered'):
            df_dois

        submit = st.button('Search DOIs', icon=":material/search:")
        
        if submit or st.session_state.get('status_expanded', False):
            if submit:
                st.session_state['status_expanded'] = True
            with st.status("Searching DOIs in OpenAlex", expanded=st.session_state.get('status_expanded', True)) as status:

    else:
        st.warning("Enter DOIs in the text area or upload a file to calculate the Citation Source Index.")
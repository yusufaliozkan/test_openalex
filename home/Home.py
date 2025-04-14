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
import hashlib
import uuid

# --- Configuration ---
st.set_page_config(layout="wide",
                    page_title='OpenAlex DOI Search Tool',
                    page_icon="https://openalex.org/img/openalex-logo-icon-black-and-white.ea51cede.png",
                    initial_sidebar_state="auto")
pd.set_option('display.max_colwidth', None)

sidebar_content()

st.title('OpenAlex DOI Search Tool', anchor=False)

# --- Session State for Storing Results ---
if 'search_results' not in st.session_state:
    st.session_state['search_results'] = {}

# --- Function to Generate Unique ID ---
def generate_unique_id(data):
    """Generates a unique ID based on the input data."""
    return hashlib.sha256(str(data).encode()).hexdigest()[:10] # Using a short hash

# --- Function to Store Results ---
def store_results(results_df):
    """Stores the results in the session state with a unique ID."""
    unique_id = generate_unique_id(results_df.to_json(orient='records'))
    st.session_state['search_results'][unique_id] = results_df.to_dict(orient='records')
    return unique_id

# --- Function to Load Results ---
def load_results(unique_id):
    """Loads results from the session state based on the unique ID."""
    if unique_id in st.session_state['search_results']:
        return pd.DataFrame(st.session_state['search_results'][unique_id])
    return None

# --- Load Results from Query Parameter if Present ---
query_params = st.experimental_get_query_params()
if 'results_id' in query_params:
    results_id = query_params['results_id'][0]
    loaded_df = load_results(results_id)
    if loaded_df is not None:
        st.info("Results loaded from shared link.")
        # Display the loaded results here (you can reuse your existing display logic)
        st.subheader("Loaded Results", anchor=False)
        st.dataframe(loaded_df)
    else:
        st.error("Invalid or expired results link.")
    st.stop() # Stop further execution if loading from link

# --- Input Section ---
df_dois = None
radio = st.radio('Select an option', ['Insert DOIs', 'Upload a file with DOIs'])
if radio == 'Insert DOIs':
    st.write('Please insert [DOIs](https://www.doi.org/) (commencing "10.") in separate rows. Maximum **700 DOIs permitted**!')
    dois = st.text_area(
        'Type or paste in one DOI per line in this box, then press Ctrl+Enter.',
        help='DOIs will be without a hyperlink such as 10.1136/bmjgh-2023-013696',
        placeholder=''' e.g.
        10.1136/bmjgh-2023-013696
        10.1097/jac.0b013e31822cbdfd
        '''
    )
    doi_list = dois.split('\n')
    doi_list = [doi.strip() for doi in doi_list if doi.strip()]
    df_dois = pd.DataFrame(doi_list, columns=["doi_submitted"])
else:
    st.write('Please upload and submit a .csv file of [DOIs](https://www.doi.org/) (commencing “10.") in separate rows. Maximum **700 DOIs permitted**!')
    st.warning('The title of the column containing DOIs should be one of the followings: doi, DOI, dois, DOIs, Hyperlinked DOI. Otherwise the tool will not identify DOIs.')
    dois = st.file_uploader("Choose a CSV file", type="csv")

    if dois is not None:
        try:
            df = pd.read_csv(dois, engine='python')
        except pd.errors.ParserError as e:
            st.error("There was a problem parsing your CSV file. Check for stray commas or improperly quoted values.")
            st.exception(e)
            st.stop()

        doi_columns = ['doi', 'DOI', 'dois', 'DOIs', 'Hyperlinked DOI']
        doi_column = next((col for col in doi_columns if col in df.columns), None)

        if doi_column:
            df_dois = df[[doi_column]]
            df_dois.columns = ['doi_submitted']
        else:
            st.error('''
            No DOI column in the file.

            Make sure that the column listing DOIs have one of the following names:
            'doi', 'DOI', 'dois', 'DOIs', 'Hyperlinked DOI'
            ''')
            st.stop()
    else:
        st.write("Please upload a CSV file containing DOIs.")

# --- Search and Display Results ---
if df_dois is not None and len(df_dois) > 700:
    st.error('Please enter 700 or fewer DOIs')
else:
    if dois:
        df_dois = df_dois.dropna()
        df_dois['doi_submitted'] = df_dois['doi_submitted'].str.replace('https://doi.org/', '', regex=False)
        df_dois = df_dois.drop_duplicates().reset_index(drop=True)
        no_dois = len(df_dois)
        if len(df_dois) > 100:
            st.toast('You entered over 100 DOIs. It may take some time to retrieve results (upto 90 seconds). Please wait.')
        if len(df_dois) > 100:
            st.warning('You entered over 100 DOIs. It may take some time to retrieve results (upto 90 seconds).')
        st.info(f'You entered {no_dois} unique DOIs')
        with st.expander(f'See the DOIs you entered'):
            df_dois = df_dois.reset_index(drop=True)
            df_dois.index += 1
            st.dataframe(df_dois, use_container_width=False)

        submit = st.button('Search DOIs', icon=":material/search:")

        if submit or st.session_state.get('status_expanded', False):
            if submit:
                st.session_state['status_expanded'] = True
            with st.status("Searching DOIs in OpenAlex", expanded=st.session_state.get('status_expanded', True)) as status:
                df_dois['doi_submitted'] = df_dois['doi_submitted'].str.replace('https://doi.org/', '', regex=False)
                df_dois['doi_submitted'] = df_dois['doi_submitted'].str.strip().str.replace('https://doi.org/', '', regex=False)

                def batch_dois(dois, batch_size=20):
                    for i in range(0, len(dois), batch_size):
                        yield dois[i:i + batch_size]

                start_time = time.time()
                all_results = []

                for batch in batch_dois(df_dois['doi_submitted'].tolist(), batch_size=20):
                    filter_string = '|'.join(batch)
                    url = f"https://api.openalex.org/works?filter=doi:{filter_string}&mailto=y.ozkan@imperial.ac.uk"
                    response = requests.get(url)
                    if response.status_code == 200:
                        results = response.json().get('results', [])
                        all_results.extend(results)
                    else:
                        print(f"Request failed for batch starting with {batch[0]}")
                    time.sleep(1)

                results_df = pd.json_normalize(all_results, sep='.')

                if not results_df.empty and 'doi' in results_df.columns:
                    results_df['doi_submitted'] = results_df['doi'].str.replace('https://doi.org/', '', regex=False)
                    merged_df = df_dois.merge(results_df, on='doi_submitted', how='left')
                    merged_df = merged_df.loc[:, ~merged_df.columns.str.startswith('abstract_inverted_index.')]
                    all_results_df = merged_df.copy()
                    merged_df = merged_df.dropna(subset='id')

                    if merged_df['id'].isnull().all():
                        st.warning("No DOIs found in the OpenAlex database.")
                    else:
                        num_results = merged_df['id'].notnull().sum()
                        st.success(f"{num_results} result(s) found.")

                    oa_status_summary = merged_df['open_access.oa_status'].value_counts(dropna=False).reset_index()
                    oa_status_summary.columns = ['OA status', '# Outputs']
                    merged_df['open_access.is_oa'] = merged_df['open_access.is_oa'].map({True: 'Open Access', False: 'Closed Access'})
                    oa_summary = merged_df['open_access.is_oa'].value_counts(dropna=False).reset_index()
                    oa_summary.columns = ['Is OA?', '# Outputs']

                    @st.fragment
                    def results_display(merged_df, oa_summary, oa_status_summary):
                        st.subheader("Open Access Status Summary", anchor=False)
                        if len(oa_summary) >= 1:
                            items = [
                                f"**{row['# Outputs']}** *{row['Is OA?']}*"
                                for _, row in oa_summary.iterrows()
                            ]
                            st.write(f"{' and '.join(items)} papers found")
                        elif len(oa_summary) == 1:
                            st.write(f'''
                                **{oa_summary.iloc[0]['# Outputs']}** *{oa_summary.iloc[0]['Is OA?']}* papers found.
                            ''')
                        available_oa_statuses = oa_status_summary['OA status'].dropna().unique().tolist()
                        selected_statuses = st.multiselect(
                            'Filter by OA Status',
                            options=available_oa_statuses,
                            default=[]
                        )
                        if selected_statuses:
                            filtered_df = merged_df[merged_df['open_access.oa_status'].isin(selected_statuses)]
                            filtered_raw_df = filtered_df.copy()
                        else:
                            filtered_df = merged_df
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            if selected_statuses:
                                oa_status_summary = filtered_df['open_access.oa_status'].value_counts(dropna=False).reset_index()
                                oa_status_summary.columns = ['OA status', '# Outputs']
                                merged_df['open_access.is_oa'] = merged_df['open_access.is_oa'].map({True: 'Open Access', False: 'Closed Access'})
                                oa_summary = merged_df['open_access.is_oa'].value_counts(dropna=False).reset_index()
                                oa_summary.columns = ['Is OA?', '# Outputs']
                                st.dataframe(oa_status_summary, hide_index=True, use_container_width=False)
                            else:
                                st.dataframe(oa_status_summary, hide_index=True, use_container_width=False)
                        with col2:
                            filtered_df = filtered_df.reset_index(drop=True)
                            filtered_df.index += 1
                            filtered_df = filtered_df[['doi', 'type_crossref', 'primary_location.source.display_name', 'primary_location.source.host_organization_name', 'publication_year', 'publication_date', 'open_access.is_oa', 'open_access.oa_status', 'open_access.oa_url', 'primary_location.license']]
                            filtered_df.columns = ['DOI', 'Type', 'Journal', 'Publisher', 'Publication year', 'Publication date', 'Is OA?', 'OA Status', 'OA URL', 'Licence']
                            st.dataframe(filtered_df)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if selected_statuses:
                                top_journals = filtered_raw_df['primary_location.source.display_name'].value_counts(dropna=False).reset_index()
                            else:
                                top_journals = merged_df['primary_location.source.display_name'].value_counts(dropna=False).reset_index()
                            top_journals.columns = ['Journal name', '# Outputs']
                            top_journals = top_journals.dropna()
                            st.subheader("Journals", anchor=False)
                            st.dataframe(top_journals, hide_index=True, use_container_width=False)
                        with col2:
                            if selected_statuses:
                                authors_df = filtered_raw_df.explode('authorships').reset_index(drop=True)
                            else:
                                authors_df = merged_df.explode('authorships').reset_index(drop=True)

                            authors_df = pd.json_normalize(authors_df['authorships']).reset_index(drop=True)
                            institutions_df = authors_df.explode('institutions').reset_index(drop=True)
                            institution_details = pd.json_normalize(institutions_df['institutions']).reset_index(drop=True)
                            institutions_df = pd.concat([
                                institutions_df.drop(columns=['institutions']).reset_index(drop=True),
                                institution_details
                            ], axis=1)
                            expected_cols = ['author.display_name', 'display_name', 'country_code', 'type']
                            for col in expected_cols:
                                if col not in institutions_df.columns:
                                    institutions_df[col] = "No info"
                            existing_cols = [col for col in expected_cols if col in institutions_df.columns]
                            institutions_table = institutions_df[existing_cols].drop_duplicates().reset_index(drop=True)
                            institutions_table.columns = ['author', 'institution', 'country_code', 'type']
                            institution_freq = institutions_table['institution'].value_counts(dropna=True).reset_index()
                            institution_freq.columns = ['Institution', '# Count']
                            st.subheader("Institutional Affiliations", anchor=False)
                            st.dataframe(institution_freq, hide_index=True, use_container_width=False)
                        with col3:
                            country_freq = institutions_table['country_code'].value_counts(dropna=True).reset_index()
                            country_freq.columns = ['Country Code', '# Count']
                            st.subheader("Country Affiliations", anchor=False)
                            st.dataframe(country_freq, hide_index=True, use_container_width=False)
                    results_display(merged_df, oa_summary, oa_status_summary)

                    @st.fragment
                    def all_results_display(all_results_df):
                        display = st.toggle('Show all results')
                        if display:
                            st.subheader('All results', anchor=False)
                            all_results_df = all_results_df.loc[:, ~all_results_df.columns.str.startswith('abstract_inverted_index.')]
                            st.dataframe(all_results_df)
                    all_results_display(all_results_df)

                    # --- Generate and Display Shareable Link ---
                    if not merged_df.empty:
                        results_id = store_results(merged_df)
                        share_url = f"{st.experimental_get_query_params().get('streamlit_url', st.request_url)}?results_id={results_id}"
                        st.subheader("Share Results", anchor=False
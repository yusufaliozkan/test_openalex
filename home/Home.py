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
import pycountry
import matplotlib.pyplot as plt
import numpy as np
from geopy.geocoders import Nominatim
import pydeck as pdk

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
    st.write('Please insert [DOIs](https://www.doi.org/) (commencing "10.") in separarate rows. Maximum **700 DOIs permitted**!')
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
    df_dois = pd.DataFrame(doi_list, columns=["doi_submitted"])
else:
    st.write('Please upload and submit a .csv file of [DOIs](https://www.doi.org/) (commencing “10.") in separate rows. Maximum **700 DOIs permitted**!')
    st.warning('The title of the column containing DOIs should be one of the followings: doi, DOI, dois, DOIs, Hyperlinked DOI. Otherwise the tool will not identify DOIs.')
    dois = st.file_uploader("Choose a CSV file", type="csv")

    if dois is not None:
        try:
            df = pd.read_csv(dois, engine='python')  # more tolerant to inconsistent rows
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

if df_dois is not None and len(df_dois) > 700:
    st.error('Please enter 500 or fewer DOIs')
else:
    if dois:
        df_dois = df_dois.dropna()
        df_dois['doi_submitted'] = df_dois['doi_submitted'].str.replace('https://doi.org/', '')
        df_dois = df_dois.drop_duplicates().reset_index(drop=True)
        no_dois = len(df_dois)
        if len(df_dois) > 100:
            st.toast('You entered over 100 DOIs. It may take some time to retrieve results (upto 90 seconds). Please wait.')
        if len(df_dois) >100:
            st.warning('You entered over 100 DOIs. It may take some time to retrieve results (upto 90 seconds).')
        st.info(f'You entered {no_dois} unique DOIs')
        with st.expander(f'See the DOIs you entered'):
            df_dois = df_dois.reset_index(drop=True)
            df_dois.index +=1
            st.dataframe(df_dois,  use_container_width=False)

        submit = st.button('Search DOIs', icon=":material/search:")
        
        if submit or st.session_state.get('status_expanded', False):
            if submit:
                st.session_state['status_expanded'] = True
            with st.spinner("Searching DOIs in OpenAlex"):
            # with st.status("Searching DOIs in OpenAlex", expanded=st.session_state.get('status_expanded', True)) as status:
                df_dois['doi_submitted'] = df_dois['doi_submitted'].str.replace('https://doi.org/', '', regex=False)
                df_dois['doi_submitted'] = df_dois['doi_submitted'].str.strip().str.lower()

                # df = pd.read_csv('your_doi_file.csv') or use your existing df
                df_dois['doi_submitted'] = df_dois['doi_submitted'].str.strip().str.replace('https://doi.org/', '', regex=False)

                # Function to batch DOIs
                def batch_dois(dois, batch_size=20):
                    for i in range(0, len(dois), batch_size):
                        yield dois[i:i + batch_size]
                
                start_time = time.time()
                # Store results
                all_results = []

                # Process in batches
                for batch in batch_dois(df_dois['doi_submitted'].tolist(), batch_size=20):
                    filter_string = '|'.join(batch)
                    url = f"https://api.openalex.org/works?filter=doi:{filter_string}&mailto=y.ozkan@imperial.ac.uk"
                    response = requests.get(url)
                    if response.status_code == 200:
                        results = response.json().get('results', [])
                        all_results.extend(results)
                    else:
                        print(f"Request failed for batch starting with {batch[0]}")
                    time.sleep(1)  # Be polite to the API

                for record in all_results:
                    record.setdefault('primary_location', {})

                    # If 'source' is missing or None, replace with empty dict
                    if not isinstance(record['primary_location'].get('source'), dict):
                        record['primary_location']['source'] = {}

                    record['primary_location']['source'].setdefault('display_name', None)
                    record['primary_location']['source'].setdefault('host_organization', None)
                    record['primary_location']['source'].setdefault('host_organization_name', None)

                # Normalize and flatten nested fields
                results_df = pd.json_normalize(all_results, sep='.')
                results_df = results_df.drop_duplicates(subset='id')

                # Add cleaned DOI for merging
                if not results_df.empty and 'doi' in results_df.columns:
                    results_df['doi_submitted'] = results_df['doi'].str.replace('https://doi.org/', '', regex=False)
                    results_df['doi_submitted'] = results_df['doi_submitted'].str.strip().str.lower()

                    # Merge with original DOIs
                    merged_df = df_dois.merge(results_df, on='doi_submitted', how='left')
                    merged_df['primary_location.source.display_name'] = merged_df['primary_location.source.display_name'].fillna('No journal name')

                    duplicates_df = merged_df[merged_df.duplicated(subset='doi', keep=False)]
                    duplicates_df = duplicates_df.reset_index(drop=True)

                    merged_df = merged_df.loc[:, ~merged_df.columns.str.startswith('abstract_inverted_index.')]
                    all_results_df = merged_df.copy()
                    merged_df = merged_df.dropna(subset='id')
                    
                    container_status = st.container()
                    if merged_df['id'].isnull().all():
                        st.warning("No DOIs found in the OpenAlex database.")
                    else:
                        num_results = merged_df['id'].notnull().sum()
                        if duplicates_df['doi'].nunique()!=0:
                            duplicate_count = duplicates_df['doi'].nunique()
                            st.success(f"{num_results} result(s) found with {duplicate_count} duplicate(s).")
                        else:
                            st.success(f"{num_results} result(s) found.")

                    oa_status_summary = merged_df['open_access.oa_status'].value_counts(dropna=False).reset_index()
                    oa_status_summary.columns = ['OA status', '# Outputs']
                    merged_df['open_access.is_oa'] = merged_df['open_access.is_oa'].map({True: 'Open Access', False: 'Closed Access'})
                    oa_summary = merged_df['open_access.is_oa'].value_counts(dropna=False).reset_index()
                    oa_summary.columns = ['Is OA?', '# Outputs']

                    # OA Summary
                    @st.fragment
                    def results(merged_df, oa_summary, oa_status_summary, duplicates_df):
                        if duplicates_df['doi'].nunique() != 0:
                            duplicate_count = duplicates_df['doi'].nunique()
                            show_duplicates = st.toggle(f'{duplicate_count} duplicate(s) found. Display and edit duplicates.')
                            if show_duplicates:
                                st.info("To remove duplicate, click the one you wish to remove from the 'select_row_to_remove' column and press 'Remove selected duplicate(s)'")
                                duplicates_df['select_row_to_remove'] = False
                                duplicates_df = duplicates_df[['select_row_to_remove'] + [col for col in duplicates_df.columns if col != 'select_row_to_remove']]
                                editable = "select_row_to_remove"
                                disabled_columns = [col for col in duplicates_df.columns if col != editable]
                                duplicates_df = st.data_editor(
                                    duplicates_df,
                                    disabled=disabled_columns
                                )
                                selected_ids = duplicates_df[duplicates_df['select_row_to_remove']]['id'].tolist()
                                remove = st.button('Remove selected duplicate(s)')
                                if remove:
                                    merged_df = merged_df[~merged_df['id'].isin(selected_ids)]
                        if merged_df['id'].isnull().all():
                            st.warning("No DOIs found in the OpenAlex database.")
                        else:
                            num_results = merged_df['id'].notnull().sum()
                            # if not duplicates_df.empty:
                            #     duplicate_count = duplicates_df['doi'].nunique()
                            #     st.success(f"{num_results} result(s) found with {duplicate_count} duplicate(s).")
                            # else:
                            #     st.success(f"{num_results} result(s) found.")
                        oa_status_summary = merged_df['open_access.oa_status'].value_counts(dropna=False).reset_index()
                        oa_status_summary.columns = ['OA status', '# Outputs']
                        # merged_df['open_access.is_oa'] = merged_df['open_access.is_oa'].map({True: 'Open Access', False: 'Closed Access'})
                        oa_summary = merged_df['open_access.is_oa'].value_counts(dropna=False).reset_index()
                        oa_summary.columns = ['Is OA?', '# Outputs']

                        if merged_df.empty:
                            st.error('No item to display!')
                            st.stop()
                        
                        st.subheader("Open Access Status Summary", anchor=False)
                        with st.expander('Results',  expanded= True):
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
                                # default=available_oa_statuses  # All selected by default
                            )
                            
                            if selected_statuses:
                                filtered_df = merged_df[merged_df['open_access.oa_status'].isin(selected_statuses)]
                                filtered_raw_df = filtered_df.copy()
                                
                            else:
                                filtered_df = merged_df.copy()
                            df_unpaywall = filtered_df.copy()

                            # Display it in Streamlit
                            
                            col1, col2 = st.columns([1,4])
                            with col1:
                                custom_colors = {
                                    "closed": "#d62728",   # soft red
                                    "green": "#2ca02c",    # muted green
                                    "gold": "#e6b800",     # warm gold
                                    "hybrid": "#1f77b4",   # calm blue
                                    "bronze": "#b87333"    # bronze tone
                                }
                                if selected_statuses:
                                    oa_status_summary = filtered_df['open_access.oa_status'].value_counts(dropna=False).reset_index()
                                    oa_status_summary.columns = ['OA status', '# Outputs']
                                    # merged_df['open_access.is_oa'] = merged_df['open_access.is_oa'].map({True: 'Open Access', False: 'Closed Access'})
                                    oa_summary = merged_df['open_access.is_oa'].value_counts(dropna=False).reset_index()
                                    oa_summary.columns = ['Is OA?', '# Outputs']
                                    table_view = st.toggle('Display as a table', key='OAstatus1')
                                    if table_view:
                                        st.dataframe(oa_status_summary, hide_index =True,  use_container_width=False)
                                    else:
                                        fig = px.pie(oa_status_summary,
                                                    names="OA status",
                                                    values="# Outputs",
                                                    title="Open Access Status",
                                                    color="OA status",
                                                    color_discrete_map=custom_colors)

                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    table_view = st.toggle('Display as a table', key='OAstatus2')
                                    if table_view:
                                        st.dataframe(oa_status_summary, hide_index =True,  use_container_width=False)
                                    else:
                                        fig = px.pie(oa_status_summary,
                                                    names="OA status",
                                                    values="# Outputs",
                                                    title="Open Access Status",
                                                    color="OA status",
                                                    color_discrete_map=custom_colors)

                                        st.plotly_chart(fig, use_container_width=True)
                            with col2:
                                
                                def safe_get_nested(row, path):
                                    current = row
                                    for key in path:
                                        if isinstance(current, dict):
                                            current = current.get(key, None)
                                        else:
                                            return None
                                    return current

                                # filtered_df['primary_location.source.display_name'] = filtered_df.apply(
                                #     lambda row: safe_get_nested(row.get('primary_location', {}), ['source', 'display_name']),
                                #     axis=1
                                # )

                                # filtered_df['primary_location.source.host_organization_name'] = filtered_df.apply(
                                #     lambda row: safe_get_nested(row.get('primary_location', {}), ['source', 'host_organization_name']),
                                #     axis=1
                                # )         
                                filtered_df= filtered_df.reset_index(drop=True)
                                filtered_df.index +=1
                                filtered_df = filtered_df[['doi', 'type_crossref','primary_location.source.display_name', 'primary_location.source.host_organization_name', 'publication_year', 'publication_date', 'open_access.is_oa','open_access.oa_status', 'open_access.oa_url', 'primary_location.license']]
                                filtered_df.columns = ['DOI', 'Type','Journal', 'Publisher','Publication year', 'Publication date','Is OA?', 'OA Status', 'OA URL', 'Licence']
                                st.dataframe(
                                    filtered_df,
                                    column_config={
                                        'DOI':st.column_config.LinkColumn('DOI'),
                                        'OA URL':st.column_config.LinkColumn('OA URL')
                                    }
                                    )

                            display_unpaywall_option = st.checkbox('Check DOI(s) on Unpaywall')

                            if display_unpaywall_option:
                                def get_oa_info(doi):
                                    url = f"https://api.unpaywall.org/v2/{doi}?email=email@ic.ac.uk"
                                    try:
                                        response = requests.get(url)
                                        if response.status_code == 200:
                                            data = response.json()
                                            return pd.Series({
                                                "oa_status": data.get("oa_status", "not_found"),
                                                "publisher": data.get("publisher", "not_found")
                                            })
                                        else:
                                            return pd.Series({"oa_status": "error", "publisher": "error"})
                                    except:
                                        return pd.Series({"oa_status": "error", "publisher": "error"})
                                df_openalex_compare = filtered_df.copy()
                                df_openalex_compare = df_openalex_compare[['DOI', 'OA Status']]
                                df_openalex_compare = df_openalex_compare.rename(columns={'OA Status':'OA Status (OpenAlex)'})

                                df_unpaywall = df_unpaywall[['doi']]
                                df_unpaywall[["oa_status", "publisher"]]  = df_unpaywall['doi'].astype(str).apply(get_oa_info)
                                df_unpaywall = df_unpaywall.rename(columns={'oa_status':'OA Status (Unpaywall)'})
                                df_unpaywall = df_unpaywall.rename(columns={'doi':'DOI'})

                                oa_status_summary = df_unpaywall['OA Status (Unpaywall)'].value_counts(dropna=False).reset_index()['OA Status (Unpaywall)'].value_counts(dropna=False).reset_index()
                                oa_status_summary
                                st.dataframe(df_unpaywall, hide_index=True)
                                
                                @st.fragment
                                def results_unpaywall_compare(df_unpaywall, df_openalex_compare):
                                    compare = st.toggle('Compare OpenAlex results with Unpaywall')
                                    if compare:
                                        df_unpaywall = pd.merge(df_unpaywall[['DOI', 'OA Status (Unpaywall)']], df_openalex_compare[['DOI', 'OA Status (OpenAlex)']], on='DOI', how='inner')
                                        
                                        df_unpaywall = df_unpaywall[df_unpaywall['OA Status (Unpaywall)'] != df_unpaywall['OA Status (OpenAlex)']]
                                        row_count = len(df_unpaywall)
                                        if row_count == 0:
                                            st.info('Unpaywall and OpenAlex show the same OA status for all DOI(s)')
                                        else:
                                            st.info('Displays only DOIs with differing OA status results between Unpaywall and OpenAlex')
                                            st.dataframe(df_unpaywall, hide_index=True)
                                results_unpaywall_compare(df_unpaywall,df_openalex_compare)

                        st.subheader("Journals and Publishers", anchor=False)
                        with st.expander('Results', expanded= True):
                            col1, col2 = st.columns(2)
                            with col1:
                                # JOURNALS
                                if selected_statuses:
                                    top_journals = filtered_raw_df['primary_location.source.display_name'].value_counts(dropna=False).reset_index()
                                else:
                                    top_journals = merged_df['primary_location.source.display_name'].value_counts(dropna=False).reset_index()
                                top_journals.columns = ['Journal name', '# Outputs']
                                top_journals = top_journals.dropna()
                                st.dataframe(top_journals, hide_index=True,  use_container_width=False)

                            with col2:
                                # PUBLISHERS
                                if selected_statuses:
                                    top_publishers = filtered_raw_df['primary_location.source.host_organization_name'].value_counts(dropna=False).reset_index()
                                else:
                                    top_publishers = merged_df['primary_location.source.host_organization_name'].value_counts(dropna=False).reset_index()
                                top_publishers.columns = ['Publisher name', '# Outputs']
                                top_publishers = top_publishers.dropna()
                                st.dataframe(top_publishers, hide_index=True,  use_container_width=False)

                        st.subheader("Affiliations", anchor=False)
                        with st.expander('Results', expanded=True):
                            col1, col2 = st.columns(2)
                            with col1:
                                # AUTHORS
                                if selected_statuses:
                                    authors_df = filtered_raw_df.explode('authorships').reset_index(drop=True)
                                else:
                                    authors_df = merged_df.explode('authorships').reset_index(drop=True)
                                
                                authors_df = pd.json_normalize(authors_df['authorships']).reset_index(drop=True)
                                authors_table = authors_df[[
                                    'author.display_name',
                                    'author.orcid',
                                    'author_position',
                                    'is_corresponding',
                                    'raw_author_name'
                                ]].drop_duplicates().reset_index(drop=True)
                                
                                # st.subheader("Authors", anchor=False)
                                # st.dataframe(authors_table,  use_container_width=False)

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

                                # st.subheader("Author Institutions")
                                # st.dataframe(institutions_table,  use_container_width=False)

                                # Institution frequency table
                                institution_freq = institutions_table['institution'].value_counts(dropna=True).reset_index()
                                institution_freq.columns = ['Institution', '# Count']
                                st.subheader("Institutional Affiliations", anchor=False)
                                st.dataframe(institution_freq, hide_index=True,  use_container_width=False)
                            with col2:
                                # Country frequency table
                                def code_to_name(code):
                                    try:
                                        return pycountry.countries.get(alpha_2=code).name
                                    except:
                                        return code  # fallback to code if not found

                                # Compute frequency table
                                country_freq = institutions_table['country_code'].value_counts(dropna=True).reset_index()
                                country_freq.columns = ['Country Code', '# Count']

                                # Map codes to full names
                                country_freq['Country'] = country_freq['Country Code'].apply(code_to_name)
                                country_freq = country_freq[['Country', '# Count']]  # Reorder columns

                                # Show in Streamlit
                                st.subheader("Country Affiliations", anchor=False)
                                st.dataframe(country_freq, hide_index=True, use_container_width=False)

                        st.subheader("Topics and SDGs", anchor=False)
                        with st.expander('Results', expanded= True):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write('**Primary Topics**')
                                if selected_statuses:
                                    top_topics = filtered_raw_df['primary_topic.field.display_name'].value_counts(dropna=False).reset_index()
                                else:
                                    top_topics = merged_df['primary_topic.field.display_name'].value_counts(dropna=False).reset_index()
                                top_topics.columns = ['Primary topic', '# Outputs']
                                no_topics = top_topics['Primary topic'].nunique()
                                st.write(f'{no_topics} primary topics found')
                                top_topics = top_topics.dropna()
                                table_view = st.toggle('Display all topics as a table')
                                if table_view:
                                    col1.dataframe(top_topics, hide_index=True,  use_container_width=False)
                                else:
                                    top_topics = top_topics.sort_values(by="# Outputs", ascending=True)
                                    top_topics = top_topics.head(10)
                                    fig = px.bar(top_topics, 
                                                x="# Outputs", 
                                                y="Primary topic", 
                                                orientation='h',
                                                title="Outputs by Primary Topic (Top 10)",
                                                labels={"# Outputs": "Number of Outputs", "Primary topic": "Topic"},
                                                color_discrete_sequence=["#636EFA"])

                                    col1.plotly_chart(fig)
                            with col2:
                                st.write('**Sustainable Development Goals (SDGs)**')
                                if selected_statuses:
                                    sdg_df = filtered_raw_df.explode('sustainable_development_goals').reset_index(drop=True)
                                    outputs_associated_with_sdgs = filtered_raw_df[
                                        filtered_raw_df['sustainable_development_goals'].notna() & filtered_raw_df['sustainable_development_goals'].astype(bool)
                                    ]
                                    num_outputs_associated_with_sdgs = len(outputs_associated_with_sdgs)
                                else:
                                    sdg_df = merged_df.explode('sustainable_development_goals').reset_index(drop=True)
                                    outputs_associated_with_sdgs = sdg_df[
                                        sdg_df['sustainable_development_goals'].notna() & sdg_df['sustainable_development_goals'].astype(bool)
                                    ]
                                    num_outputs_associated_with_sdgs = len(outputs_associated_with_sdgs)
                                sdg_df = pd.json_normalize(sdg_df['sustainable_development_goals']).reset_index(drop=True)
                                if sdg_df.empty:
                                    st.warning('No SDG found')
                                else:
                                    sdg_df = sdg_df["display_name"].value_counts().reset_index()
                                    sdg_df.columns = ["SDG name", "# Outputs"]
                                    no_sdgs = sdg_df['SDG name'].nunique()
                                    st.write(f'{no_sdgs} SDGs found associated with {num_outputs_associated_with_sdgs} outputs')
                                    table_view = st.toggle('Display all SDG categories as a table', key='sdg')
                                    if table_view:
                                        col2.dataframe(sdg_df, hide_index=True,  use_container_width=False)
                                    else:
                                        sdg_df = sdg_df.head(10)
                                        fig = px.bar(sdg_df.sort_values("# Outputs", ascending=True),
                                                    x="# Outputs", y="SDG name",
                                                    orientation='h',
                                                    title="Number of Outputs by SDG (Top 10)",
                                                    labels={"# Outputs": "Number of Outputs", "SDG name": "Sustainable Development Goal"},
                                                    color_discrete_sequence=["#636EFA"])

                                        col2.plotly_chart(fig, use_container_width=True)

                        st.subheader("Funders", anchor=False)
                        with st.expander('Results', expanded= True):
                            st.write('**Funders**')
                            if selected_statuses:
                                funders_df = filtered_raw_df.explode('grants').reset_index(drop=True)
                                outputs_associated_with_funders = filtered_raw_df[
                                    filtered_raw_df['grants'].notna() & filtered_raw_df['grants'].astype(bool)
                                ]
                                num_outputs_associated_with_funders = len(outputs_associated_with_funders)
                            else:
                                funders_df = merged_df.explode('grants').reset_index(drop=True)
                                outputs_associated_with_funders = merged_df[
                                    merged_df['grants'].notna() & merged_df['grants'].astype(bool)
                                ]
                                num_outputs_associated_with_funders = len(outputs_associated_with_funders)
                            funders_df = pd.json_normalize(funders_df['grants']).reset_index(drop=True)
                            if funders_df.empty:
                                st.warning('No funder found')
                            else:
                                funders_df = funders_df["funder_display_name"].value_counts().reset_index()
                                funders_df.columns = ["Funder name", "Count"]
                                no_funders = funders_df['Funder name'].nunique()
                                st.write(f'{no_funders} funders found associated with {num_outputs_associated_with_funders} output(s)')                   
                                table_view = st.toggle('Display all funders as a table', key='funder')
                                if table_view:
                                    st.dataframe(funders_df, hide_index=True,  use_container_width=False)
                                else:
                                    funders_df = funders_df.head(10)
                                    fig = px.bar(funders_df.sort_values("Count", ascending=True),
                                                x="Count", y="Funder name",
                                                orientation='h',
                                                title="Top 10 Number of Funders",
                                                labels={"Count": "Number of Funders", "Funder name": "Funder name"},
                                                color_discrete_sequence=["#636EFA"])
                                    fig.update_layout(
                                        yaxis=dict(
                                            tickfont=dict(size=14)  # Adjust size as needed
                                        )
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                        # st.subheader("Datasets", anchor=False)
                        # with st.expander('Results', expanded= True):
                        #     st.write('**Datasets**')
                        #     if selected_statuses:
                        #         datasets_df = filtered_raw_df.explode('datasets').reset_index(drop=True)
                        #         outputs_with_datasets = filtered_raw_df[
                        #             filtered_raw_df['datasets'].notna() & filtered_raw_df['datasets'].astype(bool)
                        #         ]
                        #         num_outputs_with_datasets = len(outputs_with_datasets)
                        #     else:
                        #         datasets_df = merged_df.explode('datasets').reset_index(drop=True)
                        #         outputs_with_datasets = merged_df[
                        #             merged_df['datasets'].notna() & merged_df['datasets'].astype(bool)
                        #         ]
                        #         num_outputs_with_datasets = len(outputs_with_datasets)
                        #     datasets_df = pd.json_normalize(datasets_df['datasets']).reset_index(drop=True)
                        #     datasets_df
                        #     if datasets_df.empty:
                        #         st.warning('No outputs found with a dataset')
                        #     else:
                        #         funders_df = funders_df["funder_display_name"].value_counts().reset_index()
                        #         funders_df.columns = ["Funder name", "Count"]
                        #         no_funders = funders_df['Funder name'].nunique()
                        #         st.write(f'{no_funders} funders found associated with {num_outputs_associated_with_funders} output(s)')                   
                        #         table_view = st.toggle('Display all funders as a table', key='funder')
                        #         if table_view:
                        #             st.dataframe(funders_df, hide_index=True,  use_container_width=False)
                        #         else:
                        #             funders_df = funders_df.head(10)
                        #             fig = px.bar(funders_df.sort_values("Count", ascending=True),
                        #                         x="Count", y="Funder name",
                        #                         orientation='h',
                        #                         title="Top 10 Number of Funders",
                        #                         labels={"Count": "Number of Funders", "Funder name": "Funder name"},
                        #                         color_discrete_sequence=["#636EFA"])
                        #             fig.update_layout(
                        #                 yaxis=dict(
                        #                     tickfont=dict(size=14)  # Adjust size as needed
                        #                 )
                        #             )
                        #             st.plotly_chart(fig, use_container_width=True)

                        st.subheader('Metrics', anchor=False)
                        with st.expander('Results', expanded=True):
                            if selected_statuses:
                                filtered_df = merged_df[merged_df['open_access.oa_status'].isin(selected_statuses)]
                                filtered_raw_df = filtered_df.copy()
                            else:
                                filtered_raw_df = merged_df.copy()

                            filtered_raw_df['author_count'] = filtered_raw_df['authorships'].apply(lambda x: len(x) if isinstance(x, list) else 0)
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                num_papers = len(filtered_raw_df)
                                st.metric(label='Number of publications', value=num_papers, border=True)
                            with col2:
                                citation_count = filtered_raw_df['cited_by_count'].sum()
                                st.metric(label='Citation count', value=citation_count, border=True)   
                            with col3:
                                fwci = filtered_raw_df['fwci'].mean().round(2)
                                st.metric(label='Field Weighted Citation Impact mean', value=fwci, border=True)
                            with col4:
                                num_authors = filtered_raw_df['author_count'].sum()
                                st.metric(label='Total number of Authors', value=f'{num_authors}', border=True)

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(label='# Author per publication', value=(num_authors/num_papers).round(2), border=True)
                            with col2:
                                if 'apc_paid.value_usd' in filtered_raw_df.columns:
                                    apc_paid_usd = filtered_raw_df['apc_paid.value_usd'].sum()
                                    st.metric(label='APC paid (USD)', value=f'${apc_paid_usd:,.2f}', border=True)
                                else:
                                    st.metric(label='APC paid (USD)', value='Not available', border=True)        

                            year_counts = filtered_raw_df['publication_year'].value_counts().reset_index()
                            year_counts.columns = ['Publication Year', 'Count']
                            year_counts = year_counts.sort_values('Publication Year')
 
                            # Create Plotly chart
                            fig = px.bar(
                                year_counts,
                                x='Publication Year',
                                y='Count',
                                title='Number of Publications per Year',
                                labels={'Count': 'Number of Publications'},
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    results(merged_df, oa_summary, oa_status_summary, duplicates_df)
                    @st.fragment
                    def all_results(all_results_df):
                        display = st.toggle('Show all results')                        
                        if display:
                            st.subheader('All results', anchor=False)
                            all_results_df = all_results_df.loc[:, ~all_results_df.columns.str.startswith('abstract_inverted_index.')]
                            all_results_df
                    all_results(all_results_df)
                    end_time = time.time()
                    processing_time = end_time - start_time
                    formatted_time = time.strftime("%M:%S", time.gmtime(processing_time))
                    container_status.write(f"Search complete in {formatted_time} minutes.")
                    # status.update(
                    #     label=f"Search complete! Results found for {num_results} DOIs in {formatted_time} minutes.",
                    #     state="complete",
                    #     expanded=True
                    # )

                else:
                    st.error("No DOIs found in the OpenAlex database. Check the submitted DOIs and resubmit.")
                    df_dois
                    # status.update(label=f"Search complete without any results!", state="complete", expanded=True)



    else:
        st.warning("Enter DOIs in the text area or upload a file to start search.")
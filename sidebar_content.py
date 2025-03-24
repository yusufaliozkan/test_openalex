import streamlit as st
import streamlit.components.v1 as components
from copyright import display_custom_license, cc_by_licence_image

def sidebar_content():
    st.logo(
        image='https://upload.wikimedia.org/wikipedia/commons/e/e3/OpenAlex_logo_2021.svg',
        # icon_image='https://upload.wikimedia.org/wikipedia/commons/e/e3/OpenAlex_logo_2021.svg'
        )

    with st.sidebar:
        # st.markdown(
        #     """
        #     <a href="https://www.imperial.ac.uk">
        #         <img src="https://upload.wikimedia.org/wikipedia/commons/e/e3/OpenAlex_logo_2021.svg" width="150">
        #     </a>
        #     """,
        #     unsafe_allow_html=True
        # )
        st.header("OpenAlex DOI Search Tool",anchor=False)  
        with st.expander('Licence'):  
            display_custom_license()
        with st.expander('Source code'):
            st.write('Source code and datasets used for this tool are available here:')
            st.caption(
                "[![GitHub repo](https://img.shields.io/badge/GitHub-OpenAlex-0a507a?logo=github)](https://github.com/yusufaliozkan/openalex) "
            )
        with st.expander('Disclaimer'):
            st.warning('''
            This tool uses OpenAlex API to retrieve metadata of multiple DOIs. API queries and metadata fields may change over time. 
            It is always good to crosscheck results with the OpenAlex database.
            ''')
        with st.expander('Contact'):
            st.write('For your questions, you can contact [Yusuf Ozkan, Research Outputs Analyst](https://profiles.imperial.ac.uk/y.ozkan) at Imperial College London.')
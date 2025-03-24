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
            There are some limitations of this tool (check the Limitations section under 'About this tool'). 
            Therefore, this tool should not be used to compare articles or a set of publications and never be used for any research performance assessment purposes.

            Although every effort is made to ensure accuracy and the tool is operational, the support may not be guaranteed. 
            Bear in mind that there might be some technical issues caused by OpenAlex or Streamlit.

            The country information is sourced from OpenAlex and may be adjusted to align with the country names used by the World Bank to generate the CSI. 
            As a result, certain country names and disputed territories might be displayed differently or not be displayed in this tool. 
            The creator of this tool assumes no responsibility for any omissions or inaccuracies.
            ''')
        with st.expander('Contact'):
            st.write('For your questions, you can contact [Yusuf Ozkan, Research Outputs Analyst](https://profiles.imperial.ac.uk/y.ozkan) at Imperial College London.')
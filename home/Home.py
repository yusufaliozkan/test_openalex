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

st.title('OpenAlex DOI Search Tool')


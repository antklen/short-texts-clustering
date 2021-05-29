"""
Download results.
"""

import base64

import streamlit as st


def csv_download_link(df, sidebar=False):

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="clusters.csv">Download as csv</a>'

    if sidebar:
        st.sidebar.markdown(href, unsafe_allow_html=True)
    else:
        st.markdown(href, unsafe_allow_html=True)

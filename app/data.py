"""
Prepare data for clustering.
"""

import pandas as pd
import streamlit as st

from .timer import timer


def load_data(session_state):

    st.header('Load data')
    st.write("""Upload csv file with data.
             Text data for clustering should be in column with name "text" or "phrase".
             Also choose size of sample to use. If sample size is greater than size of all data,
             then all data will be used.""")

    data_file = st.file_uploader('File with text data', type=['csv'])
    sample_size = st.number_input('Sample size', min_value=2,
                                  max_value=1000, value=500, step=1)
    load_data_button = st.button('Load data from file')

    if data_file is not None and load_data_button:
        with timer('load data', disable=False):

            df = pd.read_csv(data_file)

            if hasattr(df, 'phrase'):
                phrases = df['phrase']
            elif hasattr(df, 'text'):
                phrases = df['text'].rename('phrase')
            else:
                st.error('Text data in file should be in column with name "text" or "phrase".')
                return

            if sample_size < len(phrases):
                phrases = phrases.sample(n=sample_size, replace=False)

            session_state.phrases = phrases.tolist()

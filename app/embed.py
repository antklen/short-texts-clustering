"""
Make embeddings.
"""

import numpy as np
import streamlit as st
import tensorflow_hub as hub
import tensorflow_text  # pylint: disable=unused-import
from sklearn.manifold import TSNE

from .timer import timer


USE_URL = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'


class USEEmbedder:
    """Embeddings with Unversal Sentence Encoder.

    Args:
        model_path (str): Path or url to model.
        max_bach_size (int): Maximum batch size of single model call.
    """

    def __init__(self, model_path=None, max_batch_size=1000):

        self.model_path = model_path or USE_URL
        self.model = hub.load(self.model_path)
        self.max_batch_size = max_batch_size

    def __call__(self, text):
        """Get embeddings for input phrase or list of phrases.

        Args:
            text (str or list of str): Phrase or list with phrases.
        Returns:
            numpy.array with shape (512,) for one phrase or (n, 512) for list of phrases.
        """

        if not isinstance(text, str) and len(text) > self.max_batch_size:
            return np.vstack([
                self.model(text[i: i + self.max_batch_size]).numpy()
                for i in range(0, len(text), self.max_batch_size)
            ])
        else:
            return self.model(text).numpy().squeeze()


def make_embeddings(session_state):

    with timer(name='load_embedder', disable=False):
        embedder = load_embedder()

    with timer(name='compute_embeddings', disable=False):
        session_state.embeddings = compute_embeddings(embedder, session_state.phrases)


@st.cache
def load_embedder():
    return USEEmbedder()


@st.cache(allow_output_mutation=True)
def compute_embeddings(embedder, phrases):
    return embedder(phrases)


@st.cache(allow_output_mutation=True)
def compute_2d(embeddings, metric='cosine'):
    model = TSNE(metric=metric)
    return model.fit_transform(embeddings)

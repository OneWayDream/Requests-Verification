import streamlit as st
import pandas as pd
from utils.PropertiesLoader import *
import joblib


@st.cache(show_spinner=False)
def load_dataset_from_file(file):
    return pd.read_csv(file, on_bad_lines='skip')


@st.cache(show_spinner=False)
def load_dataset_by_link(link):
    return pd.read_csv(link, on_bad_lines='skip')


def load_static_models():
    prop = read_properties()
    folder_path = 'mlruns/' + prop['experiment_id'] + '/' + prop['run_id'] + '/artifacts/'
    st.session_state.static_vectorize_model = joblib.load(folder_path + prop['vectorization_model'] + '/model.pkl')
    st.session_state.static_cluster_model = joblib.load(folder_path + prop['clusterization_model'] + '/model.pkl')
    st.session_state.static_classify_model = joblib.load(folder_path + prop['classification_model'] + '/model.pkl')
    st.session_state.static_model_marks = prop['model_marks']

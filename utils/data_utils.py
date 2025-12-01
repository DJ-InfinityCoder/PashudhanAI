import os
import pandas as pd
import streamlit as st

DATASET_PATH = "dataset.csv"

@st.cache_data
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return None
    try:
        return pd.read_csv(DATASET_PATH)
    except Exception:
        return None

def get_breed_info(dataset, breed_label):
    if dataset is None:
        return None
    breed_row = dataset[dataset["breed"].astype(str).str.lower() == breed_label.lower()]
    if not breed_row.empty:
        return breed_row.iloc[0].to_dict()
    return None

def get_breed_context_string(dataset, breed_label):
    info_row = get_breed_info(dataset, breed_label)
    if info_row:
        return "\n".join([f"{k}: {v}" for k, v in info_row.items() if str(v).strip() not in ["nan", "None", ""]])
    return ""

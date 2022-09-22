import pandas as pd

def get_initial_data():
    csv_url = (
        "https://drive.google.com/uc?export=download&id=1_TP7SD18XKoODOPstaeRkXBSbUe-je8Y"
    )
    data = pd.read_csv(csv_url, on_bad_lines='skip')
    data = data.dropna(subset=['vector'])
    return data['vector'].tolist()
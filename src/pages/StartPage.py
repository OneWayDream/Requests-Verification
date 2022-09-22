from urllib.error import HTTPError

from utils.LoadUtils import *
from utils.AppContentUtils import *


def show_data_load_side_bar():
    image = Image.open(MainPropertiesLoader.DATA_LOAD_IMAGE)
    st.sidebar.image(image)
    show_size_bar_title(30, 'Load your dataset')
    if st.session_state.dataset.empty:
        option = st.sidebar.selectbox(
            'How do you want to load your dataset',
            ('CSV file', 'Link'))
    else:
        option = st.sidebar.selectbox(
            'How do you want to update your dataset',
            ('CSV file', 'Link'))
    if option == 'CSV file':
        show_data_load_by_csv()
    else:
        show_data_load_by_link()
    draw_horizontal_line(st.sidebar)
    if st.sidebar.button('Use default model', key='default_model_button'):
        st.session_state.page = 7
        st_autorefresh(interval=1, limit=2)
    return


def show_data_load_by_csv():
    uploaded_file = st.sidebar.file_uploader("Choose a file in CSV format", type=['csv'])
    flag = False
    if not st.session_state.dataset.empty:
        if st.sidebar.button("Next step"):
            st.session_state.page += 1
            st_autorefresh(interval=1, limit=2)
        flag = True
    if uploaded_file is not None:
        st.session_state.dataset = load_dataset_from_file(uploaded_file)
        if not flag:
            if st.sidebar.button("Next step"):
                st.session_state.page += 1
                st_autorefresh(interval=1, limit=2)
    return


def show_data_load_by_link():
    data_link = st.sidebar.text_input('Dataset link')
    if not data_link == '':
        try:
            with st.spinner('Wait for it...'):
                st.session_state.dataset = load_dataset_by_link(data_link)
                if st.sidebar.button("Next step"):
                    st.session_state.page += 1
                    st_autorefresh(interval=1, limit=2)
        except FileNotFoundError:
            show_exception(15, 'It doesn\'t seem like a link -_-', st.sidebar)
        except HTTPError:
            show_exception(15, 'We can\'t get the dataset from your link, st.sidebar', st.sidebar)
    return


def show_start_info():
    show_title(50, 'Request analysis tool')

    preparation_column, clusterization_column, classification_column = st.columns(3)

    with preparation_column:
        image = Image.open(MainPropertiesLoader.PREPARATION_IMAGE)
        st.image(image, width=200)
        show_title(40, 'Prepare')
        show_text(30, 'your dataset before use. Drop useless columns and fill skips.')

    with clusterization_column:
        image = Image.open(MainPropertiesLoader.CLUSTERIZATION_IMAGE)
        st.image(image, width=200)
        show_title(40, 'Clusterizate')
        show_text(30, 'your data and choose dangerous clusters.')

    with classification_column:
        image = Image.open(MainPropertiesLoader.CLASSIFICATION_IMAGE)
        st.image(image, width=200)
        show_title(40, 'Build')
        show_text(30, 'classification model and use it to handle new requests.')

from utils.LoadUtils import *
from utils.AppContentUtils import *
from urllib.error import HTTPError

from db.DbService import add_new_requests
from db.Requests import NewRequest


def show_static_model_content():
    if 'static_vectorize_model' not in st.session_state:
        load_static_models()
    image = Image.open(MainPropertiesLoader.STATIC_MODEL_IMAGE)
    st.sidebar.image(image)
    show_size_bar_title(30, 'Default model')
    show_size_bar_text(20, 'Use our static model to classify your data')
    data_mode = st.sidebar.selectbox('What do you want to do:?', ['Definition of a single request', 'Dataset marking'],
                                     index=0, key='data_mode_key')
    if data_mode == 'Definition of a single request':
        show_static_content_for_single_request()
    elif data_mode == 'Dataset marking':
        if 'upload_entity' not in st.session_state:
            st.session_state.upload_entity = None
        if 'marked_dataset' not in st.session_state:
            st.session_state.marked_dataset = False
        if 'after_button' not in st.session_state:
            st.session_state.after_button = False
        show_static_content_for_data_marking()
    if st.sidebar.button('<-- Move to the main page'):
        st.session_state.page = 0
        st_autorefresh(interval=1, limit=2)
    return


def show_static_content_for_single_request():
    show_title(50, 'Classify your request')
    request = st.text_input('Your request', key='request_key')
    if st.button('Classify', key='classify_button'):
        if len(request) == 0:
            show_exception(20, 'First enter your request')
        else:
            vector = st.session_state.static_vectorize_model.transform([request.lower()])
            prediction = st.session_state.static_classify_model.predict(vector.toarray())[0]
            prediction_probability = st.session_state.static_classify_model.predict_proba(vector.toarray())

            r_type = mark_interpretation(prediction)

            show_result_text(30, 'Request type: ' + str(prediction) + ' (' + r_type + ')')
            show_result_text(30, 'Prediction probability: ' + str(prediction_probability[0, st.session_state
                                                                  .static_model_marks
                                                                  .index(prediction)]))
            add_new_requests([NewRequest(vector=request.lower())])
    return


def show_static_content_for_data_marking():
    show_size_bar_title(25, 'Load your dataset')
    option = st.sidebar.selectbox(
        'How do you want to load your dataset',
        ('CSV file', 'Link'))
    if option == 'CSV file':
        show_data_load_by_csv()
    else:
        show_data_load_by_link()
    return


def show_data_load_by_csv():
    uploaded_file = st.sidebar.file_uploader("Choose a file in CSV format", type=['csv'])
    if not st.session_state.dataset.empty:
        show_dataset_marking_content()
    if uploaded_file is not None:
        if uploaded_file != st.session_state.upload_entity:
            st.session_state.dataset = load_dataset_from_file(uploaded_file)
            st.session_state.upload_entity = uploaded_file
            st.session_state.marked_dataset = False
            st.session_state.after_button = False
            st_autorefresh(interval=1, limit=2)
    return


def show_data_load_by_link():
    data_link = st.sidebar.text_input('Dataset link')
    if not st.session_state.dataset.empty:
        show_dataset_marking_content()
    if not data_link == '':
        try:
            with st.spinner('Wait for it...'):
                if data_link != st.session_state.upload_entity:
                    st.session_state.dataset = load_dataset_by_link(data_link)
                    st.session_state.upload_entity = data_link
                    st.session_state.marked_dataset = False
                    st.session_state.after_button = False
                    st_autorefresh(interval=1, limit=2)
        except FileNotFoundError:
            show_exception(15, 'It doesn\'t seem like a link -_-', st.sidebar)
        except HTTPError:
            show_exception(15, 'We can\'t get the dataset from your link, st.sidebar', st.sidebar)
    return


def show_dataset_marking_content():
    show_dataset()
    draw_horizontal_line()
    show_dataset_info(40, "Dataset info")
    show_title(40, 'Choose and adjust vectors')
    vector_column = st.selectbox('Vector\'s column', options=st.session_state.dataset.columns, key='vector_column_key')
    delete_null_vectors = st.checkbox('Delete null vectors', value=True, key='delete_null_flag')
    proba_column = st.checkbox('Add probabilities column', value=True, key='add_proba_flag')
    if st.button('Classify', key='dataset_classify_button'):

        if delete_null_vectors:
            st.session_state.dataset = st.session_state.dataset.copy().dropna(subset=[vector_column])

        vectors = st.session_state.static_vectorize_model.transform(st.session_state.dataset[vector_column]
                                                                    .apply(lambda x: x.lower()))
        prediction = st.session_state.static_classify_model.predict(vectors.toarray())

        st.session_state.dataset['marks'] = prediction

        if proba_column:
            prediction_probability = st.session_state.static_classify_model.predict_proba(vectors)
            prediction_probability = [
                prediction_probability[i, st.session_state.static_model_marks.index(
                    st.session_state.dataset.iloc[i]['marks']
                )] for i in range(len(prediction_probability))]
            st.session_state.dataset['proba'] = prediction_probability
            st.session_state.marked_dataset = True
            st.session_state.after_button = True
            requests_to_save = list()
            for vector in st.session_state.dataset[vector_column].apply(lambda x: x.lower()).tolist():
                requests_to_save.append(NewRequest(vector=vector))
            add_new_requests(requests_to_save)
    if st.session_state.after_button:
        st.balloons()
        st.session_state.after_button = False

    if st.session_state.marked_dataset:
        show_value_counts_block()
        if st.download_button('Download CSV', convert_df(st.session_state.dataset), "dataset.csv", key='download-csv'):
            show_success_text(30, 'Thanks for downloading!')

    return


def mark_interpretation(prediction):
    for r_type, r_info in st.session_state.request_types.items():
        if r_info.mark == prediction:
            return r_type


def show_value_counts_block():
    show_title(40, 'Marks values')
    st.text(st.session_state.dataset['marks'].value_counts())
    return


@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')


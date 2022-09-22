from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from utils.AppContentUtils import *


def show_vectorize_content():
    show_dataset()
    draw_horizontal_line()
    show_vectorize_adjusting_block()
    return


def show_vectorize_adjusting_block():

    show_title(50, 'Customize the vectorization')

    show_title(25, 'The column for vectorization:')
    column_option = st.selectbox(
        '',
        st.session_state.dataset.columns,
        key='column_select_box')

    show_title(25, 'The method of vectorization:')
    method_option = st.selectbox(
        '',
        ['TfidfVectorizer', 'CountVectorizer'],
        key='method_select_box')

    if method_option == 'TfidfVectorizer' or method_option == 'CountVectorizer':
        show_title(25, 'Ngram range:')
        ngram_range = st.slider(
            '',
            1, 5, (1, 2))

        show_title(25, 'Max features amount:')
        max_features = st.number_input('', min_value=100, max_value=10000, value=3000, step=1)

    if st.button("Confirm", key='confirm'):
        with st.spinner('Wait for it...'):
            vectors = st.session_state.dataset[column_option].apply(lambda s: str(s).lower())
            if method_option == 'TfidfVectorizer':
                vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range, max_features=max_features)
                tfid_X = vectorizer.fit_transform(vectors)
                st.session_state.vectorize_model = vectorizer
                st.session_state.vectorize_tech = 'Vectorizer'
                st.session_state.requests_vectors = tfid_X
            elif method_option == 'CountVectorizer':
                vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range, max_features=max_features)
                count_X = vectorizer.fit_transform(vectors)
                st.session_state.vectorize_model = vectorizer
                st.session_state.vectorize_tech = 'Vectorizer'
                st.session_state.requests_vectors = count_X

        st.session_state.vectorize_column = column_option
        show_success_text(25, 'Done! You can head to the next page ^u^')
        st.balloons()

    return

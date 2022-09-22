import sys

from utils import StyleUtils
from pages.StartPage import *
from pages.PreparationPage import *
from pages.VectorizePage import *
from pages.ClusterizationPage import *
from pages.MarkingPage import *
from pages.ModelTrainPage import *
from pages.ClassificationPage import *
from pages.StaticClassificationPage import *
from utils.AppContentUtils import *
from utils.PropertiesLoader import MainPropertiesLoader

from db.DbService import init, get_count_of_requests
from db.Config import *
from db.Requests import Request

from utils.DataUtils import get_initial_data


# ------------------------------------------------------

# st.session_state.page - current page number
# st.session_state.dataset - user's dataset
# st.session_state.vectorize_model - vectorization model
# st.session_state.requests_vectors - vectorized requests
# st.session_state.clusterization_model - clusterization model
# st.session_state.best_clusters_amount - the best cluster's amount for clusterize model
# st.session_state.request_types - map of request types, request codes and select words
# st.session_state.vectorize_column - the column for vectorization (need to create extra features)
# st.session_state.classification_model - classification model

# ------------------------------------------------------


def main():
    init_db()
    StyleUtils.set_png_as_page_bg(MainPropertiesLoader.BACKGROUND_IMAGE)
    StyleUtils.local_css(MainPropertiesLoader.CSS_FILE)
    if st.session_state.page == 0:
        show_data_load_page()
    elif st.session_state.page == 1:
        show_data_prepare_page()
    elif st.session_state.page == 2:
        show_vectorize_page()
    elif st.session_state.page == 3:
        show_clusterization_page()
    elif st.session_state.page == 4:
        show_marking_page()
    elif st.session_state.page == 5:
        show_model_learning_page()
    elif st.session_state.page == 6:
        show_classification_page()
    elif st.session_state.page == 7:
        show_static_model_page()
    return


def init_db():
    init(DRIVER + "://" + USER + ":" + PASSWORD + "@" + HOST + ":" + PORT + "/" + DB_NAME,
         MAX_REQUESTS,
         MAX_NEW_REQUESTS
         )
    # if get_count_of_requests() == 0:
    #     requests = list()
    #     for request in get_initial_data():
    #         requests.append(Request(vector=request))
    #     add_new_requests(requests)


def show_data_load_page():
    show_data_load_side_bar()
    if st.session_state.dataset.empty:
        show_start_info()
    else:
        show_dataset()
        draw_horizontal_line()
        show_dataset_info(40, "Dataset info")


def show_data_prepare_page():
    show_side_bar(MainPropertiesLoader.PREPARATION_IMAGE, MainPropertiesLoader.SIDE_BAR_TITLE_COLOR,
                  MainPropertiesLoader.SIDE_BAR_TEXT_COLOR, 30, 20,
                  "Prepare your dataset",
                  "Look over your dataset, delete some columns and handle skip before the next steps",
                  True, True)
    show_preparing_dataset_content()
    return


def show_vectorize_page():
    show_side_bar(MainPropertiesLoader.VECTORIZATION_IMAGE, MainPropertiesLoader.SIDE_BAR_TITLE_COLOR,
                  MainPropertiesLoader.SIDE_BAR_TEXT_COLOR, 30, 20,
                  "Vectorize your requests",
                  "Choose the column you wanna vectorize and customize the method of vectorization",
                  True, True)
    show_vectorize_content()
    return


def show_clusterization_page():
    show_side_bar(MainPropertiesLoader.CLUSTERIZATION_IMAGE, MainPropertiesLoader.SIDE_BAR_TITLE_COLOR,
                  MainPropertiesLoader.SIDE_BAR_TEXT_COLOR, 30, 20,
                  "Clusterize your requests",
                  "Group your requests by vectors that were obtained in the previous step",
                  True, True)
    show_clusterize_content()
    return


def show_marking_page():
    show_side_bar(MainPropertiesLoader.MARKING_IMAGE, MainPropertiesLoader.SIDE_BAR_TITLE_COLOR,
                  MainPropertiesLoader.SIDE_BAR_TEXT_COLOR, 30, 20,
                  "Mark the clusters",
                  "Define clusters as a group of harmful requests based on the specific words they contained. We\'ve "
                  "already added some typical dangers",
                  True, True)
    show_marking_content()
    return


def show_model_learning_page():
    show_side_bar(MainPropertiesLoader.LEARNING_IMAGE, MainPropertiesLoader.SIDE_BAR_TITLE_COLOR,
                  MainPropertiesLoader.SIDE_BAR_TEXT_COLOR, 30, 20,
                  "Train the model",
                  "Summarize all of the previous steps to the requests classification model!",
                  True, True)
    show_model_training_content()
    return


def show_classification_page():
    show_side_bar(MainPropertiesLoader.CLASSIFICATION_IMAGE, MainPropertiesLoader.SIDE_BAR_TITLE_COLOR,
                  MainPropertiesLoader.SIDE_BAR_TEXT_COLOR, 30, 20,
                  "Classify your requests",
                  "Define the danger of your requests using the model from the previous step",
                  False, True)
    show_prediction_content()
    return


def show_static_model_page():
    show_static_model_content()
    return


if __name__ == '__main__':
    if 'page' not in st.session_state:
        st.session_state.page = 0
    if 'dataset' not in st.session_state:
        st.session_state.dataset = pd.DataFrame()
    if 'vectorize_model' not in st.session_state:
        st.session_state.vectorize_model = None
    if 'requests_vectors' not in st.session_state:
        st.session_state.requests_vectors = None
    if 'clusterization_model' not in st.session_state:
        st.session_state.clusterization_model = None
    if 'best_clusters_amount' not in st.session_state:
        st.session_state.best_clusters_amount = None
    if 'request_types' not in st.session_state:
        st.session_state.request_types = MainPropertiesLoader.MARKS_DIST
    if 'vectorize_column' not in st.session_state:
        st.session_state.vectorize_column = 'vector'
    if 'classification_model' not in st.session_state:
        st.session_state.classification_model = None
    if 'columns_features' not in st.session_state:
        st.session_state.columns_features = None
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None
    if 'model_words_features' not in st.session_state:
        st.session_state.model_words_features = None
    if 'model_vectors' not in st.session_state:
        st.session_state.model_vectors = None
    main()

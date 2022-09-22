import json


class MarksInfo:
    def __init__(self, mark, select_words):
        self.mark = mark
        self.select_words = select_words


class MainPropertiesLoader:

    # Images

    BACKGROUND_IMAGE = 'images/background.png'
    PREPARATION_IMAGE = 'images/preparation.png'
    VECTORIZATION_IMAGE = 'images/vectorization.png'
    CLUSTERIZATION_IMAGE = 'images/clusterization.png'
    MARKING_IMAGE = 'images/marking.png'
    LEARNING_IMAGE = 'images/learning.png'
    CLASSIFICATION_IMAGE = 'images/data-classification.png'
    PREDICTION_IMAGE = 'images/prediction.png'
    DATA_LOAD_IMAGE = 'images/dataset_icon.png'
    STATIC_MODEL_IMAGE = 'images/static_model.png'

    # File paths

    CSS_FILE = 'styles/style.css'

    # Colors

    SIDE_BAR_TITLE_COLOR = "#6A5ACD"
    SIDE_BAR_TEXT_COLOR = "#6A5ACD"
    TEXT_COLOR = "#293133"
    TITLE_COLOR = "#293133"
    EXCEPTION_COLOR = "#E32636"
    SUCCESS_COLOR = '#FFCF48'
    RESULT_COLOR = '#34C924'

    # Machine learning features

    MARKS_DIST = {
        'Harmless request': MarksInfo(mark=0, select_words=[]),
        'SQL injection request': MarksInfo(mark=1, select_words=['select']),
        'Reflected XSS attacks': MarksInfo(mark=2, select_words=['xscript', 'xss']),
        'Stored XSS attacks': MarksInfo(mark=3, select_words=['<script>']),
        'X-path attacks': MarksInfo(mark=4, select_words=["' or"]),
        'OS Command Injection attacks': MarksInfo(mark=5, select_words=['cmd', 'execute', 'echo', 'function',
                                                                        'func', 'query']),
        'Path Traversal attacks': MarksInfo(mark=6, select_words=['../../../../../', '/etc/passwd']),
    }


class AdminPropertiesLoader:

    # Images

    BACKGROUND_IMAGE = 'images/admin_background.jpg'
    TEXT_COLOR = "#FFE4C4"
    TITLE_COLOR = "#293133"
    SUB_TITLE_COLOR = "#FAE7B5"
    EXCEPTION_COLOR = "#E32636"
    SUCCESS_COLOR = '#FFCF48'
    RESULT_COLOR = '#34C924'

    # Properties

    PROPERTIES_PATH = 'utils/properties.json'
    MLFLOW_PATH = 'mlflow_script.py'
    MIN_UPDATE_TIME = 60*60*6


class MlflowPropertiesLoader:

    VECTORIZER_MAX_FEATURES = 2000
    MLFLOW_PATH = AdminPropertiesLoader.PROPERTIES_PATH
    MARKS_DIST = MainPropertiesLoader.MARKS_DIST
    MIN_ENTRIES_AMOUNT = 5
    MAX_CLUSTERS_AMOUNT = 30
    MAX_CLUSTER_SIZE = 600


def read_properties():
    return json.load(open(AdminPropertiesLoader.PROPERTIES_PATH))


def write_properties(dist):
    json.dump(dist, open(AdminPropertiesLoader.PROPERTIES_PATH, 'w'))
    return


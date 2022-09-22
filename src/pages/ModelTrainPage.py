import pandas as pd
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from lightgbm import LGBMClassifier

from utils.AppContentUtils import *


def show_model_training_content():
    show_title(50, 'Model learning')

    if 'columns_types' not in st.session_state:
        st.session_state.columns_types = {}
        for column in st.session_state.dataset.columns:
            st.session_state.columns_types[column] = 'Categorical'

    show_title(30, 'Choose the type of learning')
    learning_type = st.selectbox(
        '',
        ['Сlusterization interpretation', 'Learn based on different features'],
        key='learning_type_select_box')

    draw_horizontal_line()

    show_title(30, 'Choose the model for prediction')
    model_type = st.selectbox(
        '',
        ['Catboost', 'LGBMClassifier'],
        key='model_select_box')

    draw_horizontal_line()

    if learning_type == 'Сlusterization interpretation':
        show_interpretation_block(model_type)
    elif learning_type == 'Learn based on different features':
        show_training_block(model_type)
    return


def show_interpretation_block(model_type):
    show_title(30, 'Learning settings')
    df = pd.DataFrame([['Clusterization Labels', 'Categorical']], columns=['feature', 'type'])
    st.write(df)
    if st.button('Learn the model', key='learn_button'):
        if model_type == 'Catboost':
            with st.spinner('Wait for it'):
                result_df = learn_catboost_model(
                    pd.DataFrame(st.session_state.dataset['clusterization_label'].astype(str)),
                    st.session_state.dataset.loc[:, "mark"],
                    st.session_state.columns_types)
                show_title(30, 'Model\'s metrics')
                st.write(result_df)
                st.session_state.is_learned = True
                st.session_state.columns_features = []
                st.session_state.model_type = 'INTERPRETATION'
                st.session_state.model_tech = 'Catboost'
        elif model_type == 'LGBMClassifier':
            with st.spinner('Wait for it'):
                result_df = learn_lgbm_classifier_model(
                    pd.DataFrame(st.session_state.dataset['clusterization_label']),
                    st.session_state.dataset.loc[:, "mark"])
                show_title(30, 'Model\'s metrics')
                st.write(result_df)
                st.session_state.is_learned = True
                st.session_state.columns_features = []
                st.session_state.model_type = 'INTERPRETATION'
                st.session_state.model_tech = 'LGBMClassifier'
    if 'is_learned' in st.session_state:
        show_success_text(25, 'Done! You can head to the next page ^u^')
    return


def show_training_block(model_type):
    if model_type == 'Catboost':
        show_title(30, 'Choose features for prediction')
        features_settings = st.multiselect(
            '',
            ['Use vectorized features', 'Use features based on select words'],
            key='columns_settings')

        draw_horizontal_line()

        show_title(30, 'Choose columns for prediction')
        columns_features = st.multiselect(
            '',
            st.session_state.dataset.columns[:-2],
            key='columns_multiselect')
        draw_horizontal_line()

        show_title(30, 'Adjust column\'s settings in the learning process')
        column = st.selectbox(
            '',
            columns_features,
            key='column_select_box')
        if column is not None:
            show_text(25, 'Choose the type of the variable')

            column_type = st.selectbox(
                '',
                ['Categorical', 'Interval'],
                key='type_select_box')
            if st.button('Choose the type', key='type_button'):
                st.session_state.columns_types[column] = column_type

        draw_horizontal_line()

        show_title(30, 'Learning settings')
        df = pd.DataFrame([[feature, st.session_state.columns_types.get(feature)] for feature in columns_features],
                          columns=['feature', 'type'])
        if 'Use vectorized features' in features_settings:
            f_df = pd.DataFrame([['*Vectorized features', 'Categorical']], columns=['feature', 'type'])
            df = df.append(f_df)
        if 'Use features based on select words' in features_settings:
            f_df = pd.DataFrame([['*Select words features', 'Categorical']], columns=['feature', 'type'])
            df = df.append(f_df)
        st.write(df)
        draw_horizontal_line()

    show_title(30, 'Choose row\'s limit for the learning')
    rows_limit = st.number_input('', min_value=1, max_value=999999, value=100, step=1)

    draw_horizontal_line()

    if st.button('Learn the model', key='learn_button'):
        if model_type == 'Catboost':
            with st.spinner('Wait for it'):
                learning_df = prepare_learning_dataset(features_settings, columns_features)
                y_vector = st.session_state.dataset.head(rows_limit).loc[:, "mark"]
                result_df = learn_catboost_model(learning_df.head(rows_limit), y_vector, st.session_state.columns_types)
                show_title(30, 'Model\'s metrics')
                st.write(result_df)
                st.session_state.is_learned = True
                st.session_state.columns_features = columns_features
                st.session_state.model_type = 'TRAINING'
                st.session_state.model_words_features = 'Select words features' in features_settings
                st.session_state.model_vectors = 'Use vectorized features' in features_settings
                st.session_state.model_tech = 'Catboost'
        elif model_type == 'LGBMClassifier':
            with st.spinner('Wait for it'):
                learning_df = pd.DataFrame(st.session_state.requests_vectors.toarray(),
                                           columns=[('v_' + str(i)) for i in range(
                                               0, st.session_state.requests_vectors.shape[1])])
                y_vector = st.session_state.dataset.head(rows_limit).loc[:, "mark"]
                result_df = learn_lgbm_classifier_model(learning_df.head(rows_limit), y_vector)
                show_title(30, 'Model\'s metrics')
                st.write(result_df)
                st.session_state.is_learned = True
                st.session_state.columns_features = []
                st.session_state.model_type = 'TRAINING'
                st.session_state.model_words_features = False
                st.session_state.model_vectors = True
                st.session_state.model_tech = 'LGBMClassifier'
    if 'is_learned' in st.session_state:
        show_success_text(25, 'Done! You can head to the next page ^u^')
    return


def learn_catboost_model(x_dataset, y_vector, columns_types):
    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_vector, test_size=0.20, random_state=1)
    catboost = CatBoostClassifier(n_estimators=200,
                                  loss_function='MultiClass',
                                  learning_rate=0.4,
                                  task_type='CPU',
                                  random_state=1,
                                  verbose=False)
    columns = list(x_dataset.columns)
    for column, column_type in columns_types.items():
        if column_type == 'Interval' and column in columns:
            columns = columns.remove(column)
    pool_train = Pool(x_train, y_train, cat_features=columns)
    pool_test = Pool(x_test, cat_features=columns)
    catboost.fit(pool_train)
    y_pred = catboost.predict(pool_test)
    result_df = pd.DataFrame([
        ['Recall', recall_score(y_test, y_pred, average='macro')],
        ['Precision', precision_score(y_test, y_pred, average='macro')],
        ['Accuracy', accuracy_score(y_test, y_pred)],
    ], columns=['metric', 'value'])
    st.session_state.classification_model = catboost
    return result_df


def learn_lgbm_classifier_model(x_dataset, y_vector):
    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_vector, test_size=0.20, random_state=1)
    lgbm_classifier = LGBMClassifier().fit(x_train, y_train)
    y_pred = lgbm_classifier.predict(x_test)
    result_df = pd.DataFrame([
        ['Recall', recall_score(y_test, y_pred, average='macro')],
        ['Precision', precision_score(y_test, y_pred, average='macro')],
        ['Accuracy', accuracy_score(y_test, y_pred)],
    ], columns=['metric', 'value'])
    st.session_state.classification_model = lgbm_classifier
    return result_df


def prepare_learning_dataset(features_settings, columns_features):
    x_dataset = pd.DataFrame()
    if 'Use vectorized features' in features_settings:
        f_df = pd.DataFrame(st.session_state.requests_vectors.todense(),
                            columns=[('f' + str(i)) for i in range(0, st.session_state.requests_vectors.shape[1])])
        x_dataset = pd.concat([x_dataset, f_df], axis=1)
    if 'Select words features' in features_settings:
        for r_type, r_info in st.session_state.request_types.items():
            for word in r_info.select_words:
                x_dataset[r_type + '&' + word] = st.session_state.dataset[st.session_state.vectorize_column] \
                    .apply(lambda x: 1 if (x.lower().find(word) != -1) else 0)
    for column in columns_features:
        x_dataset[column] = st.session_state.dataset[column]
    for column in x_dataset.columns:
        x_dataset[column] = x_dataset[column].astype(str)
    return x_dataset

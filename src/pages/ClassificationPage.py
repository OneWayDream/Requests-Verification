from utils.AppContentUtils import *
from db.DbService import add_new_requests
from db.Requests import NewRequest


def show_prediction_content():
    show_title(50, 'Requests classification')

    show_title(30, 'Vectors')

    vector = st.text_input('', placeholder='Your vector', key='vector_input')

    features = []
    for feature in st.session_state.columns_features:
        show_title(30, feature)
        features.append(st.text_input('', key=(feature + '_input')))

    if st.button('Classify', key='classify_button'):
        v = []
        if st.session_state.vectorize_tech == 'Vectorizer':
            v = st.session_state.vectorize_model.transform([vector.lower()]).toarray()
        if st.session_state.model_type == 'INTERPRETATION':
            label = st.session_state.clusterization_model.predict(v)
            prediction = st.session_state.classification_model.predict([label])
            if st.session_state.model_tech == 'Catboost':
                prediction = prediction[0][0]
            elif st.session_state.model_tech == 'LGBMClassifier':
                prediction = prediction[0]
            for r_type, r_info in st.session_state.request_types.items():
                if r_info.mark == prediction:
                    show_text(30, 'Result: ' + str(prediction) + ' (' + r_type + ')')
                    break
        elif st.session_state.model_type == 'TRAINING':
            req_l = []
            if st.session_state.model_vectors:
                req_l = v[0]
            if st.session_state.model_words_features:
                for r_type, r_info in st.session_state.request_types.items():
                    for word in r_info.select_words:
                        req_l.append(lambda x: 1 if (vector.lower().find(word) != -1) else 0)
            if len(features) > 0:
                req_l = req_l + features
            prediction = st.session_state.classification_model.predict([req_l])
            for r_type, r_info in st.session_state.request_types.items():
                if r_info.mark == prediction:
                    if st.session_state.model_tech == 'Catboost':
                        show_text(30, 'Result: ' + str(prediction[0][0]) + ' (' + r_type + ')')
                    elif st.session_state.model_tech == 'LGBMClassifier':
                        show_text(30, 'Result: ' + str(prediction[0]) + ' (' + r_type + ')')
                    break
        add_new_requests([NewRequest(vector=vector.lower())])
    return

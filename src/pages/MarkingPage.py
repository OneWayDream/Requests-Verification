import pandas as pd
from collections import Counter

from utils.AppContentUtils import *


def show_marking_content():
    show_dataset_marks()
    draw_horizontal_line()
    show_marks_settings()
    return


def show_dataset_marks():
    show_title(50, 'Dataset Marks')
    if 'mark' not in st.session_state.dataset.columns:
        st.session_state.dataset['mark'] = [0] * st.session_state.dataset.shape[0]
    df = pd.DataFrame(st.session_state.request_types.keys(), columns=['request_type'])
    df['mark'] = [value.mark for value in st.session_state.request_types.values()]
    count_dist = Counter(st.session_state.dataset['mark'].tolist())
    df['amount'] = [count_dist.get(entry.mark) if count_dist.get(entry.mark) is not None
                    else 0 for entry in st.session_state.request_types.values()]
    st.write(df)
    return


def show_marks_settings():
    show_title(50, 'Marks settings')
    show_marks_add_block()
    draw_horizontal_line()
    show_marks_delete_block()
    draw_horizontal_line()
    show_words_block()
    draw_horizontal_line()
    show_marking_block()
    return


def show_marks_add_block():
    show_title(30, 'Add new mark')
    mark_name = st.text_input('', key='add_mark', placeholder='Your new mark')

    if st.button('Add', key='mark_add'):
        if mark_name not in st.session_state.request_types.keys():
            st.session_state.request_types[mark_name] = MarksInfo(len(st.session_state.request_types.keys()), [])
            st_autorefresh(interval=1, limit=2)
        else:
            show_exception(15, 'The mark with this name already exists')
    return


def show_marks_delete_block():
    show_title(30, 'Delete existing mark')

    delete_marks = st.multiselect(
        '',
        list(st.session_state.request_types.keys())[1:],
        key='delete_multiselect')

    if st.button('Delete', key='mark_delete'):
        for mark in delete_marks:
            st.session_state.request_types.pop(mark)
        st_autorefresh(interval=1, limit=2)
    return


def show_words_block():
    show_title(50, 'Marks words')

    show_title(20, 'Choose the mark to manage select words')
    mark_key = st.selectbox(
        '',
        st.session_state.request_types.keys(),
        key='mark_select_box')

    st.write(pd.DataFrame(st.session_state.request_types.get(mark_key).select_words, columns=['word']))

    show_title(30, 'Add a new word')
    select_word = st.text_input('', key='add_word', placeholder='Your new word')

    if st.button('Add', key='word_add'):
        if select_word not in st.session_state.request_types.get(mark_key).select_words:
            st.session_state.request_types.get(mark_key).select_words.append(select_word)
            st_autorefresh(interval=1, limit=2)
        else:
            show_exception(15, 'This word already exists')

    show_title(30, 'Delete existing word')
    delete_words = st.multiselect(
        label='',
        options=list(st.session_state.request_types.get(mark_key).select_words),
        key='delete_words')

    if st.button('Delete', key='words_delete'):
        for word in delete_words:
            st.session_state.request_types.get(mark_key).select_words.remove(word)
        st_autorefresh(interval=1, limit=2)
    return


def show_marking_block():
    show_title(50, 'Marking')

    show_title(30, 'Choose necessary elements amount to mark cluster')
    entries_num = st.number_input('', min_value=1, max_value=999999, value=2, step=1)

    if st.button(label='Mark dataset', key='mark_button'):
        with st.spinner('Wait for it...'):
            mark_dataset(entries_num)
        st.session_state.is_marked = True
        st_autorefresh(interval=1, limit=2)

    if 'is_marked' in st.session_state:
        show_success_text(25, 'Done! You can head to the next page ^u^')
        st.snow()
    return


def mark_dataset(entries_num):
    for r_type, r_info in st.session_state.request_types.items():
        filtered_dataset = st.session_state.dataset[
            st.session_state.dataset['vector'].apply(lambda x: contains_check(x.lower(), r_info.select_words))]
        counters = filtered_dataset['clusterization_label'].value_counts().to_numpy()
        for i, label in enumerate(filtered_dataset['clusterization_label'].value_counts().index):
            if label != 0:
                if counters[i] >= entries_num:
                    st.session_state.dataset.loc[st.session_state.dataset.clusterization_label == label,
                                                 'mark'] = r_info.mark
    return


def contains_check(s, arr):
    result = False
    for elem in arr:
        result = result or (s.find(elem) != -1)
        if result:
            break
    return result

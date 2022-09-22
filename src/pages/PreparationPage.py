import pandas as pd

from utils.AppContentUtils import *


def show_preparing_dataset_content():
    show_dataset()
    draw_horizontal_line()
    show_dataset_info(40, "Dataset info")
    draw_horizontal_line()
    show_columns_delete_block()
    draw_horizontal_line()
    show_skips_handle_block()
    draw_horizontal_line()
    show_value_counts_block()
    return


def show_columns_delete_block():
    show_title(50, 'Delete columns')
    show_text(20, 'Choose the rows that you want delete')
    delete_option = st.multiselect(
        '',
        st.session_state.dataset.columns,
        key='delete_multiselect')
    if st.button("Confirm", key='confirm_delete'):
        delete_columns(delete_option)
    return


def delete_columns(columns):
    st.session_state.dataset = st.session_state.dataset.drop(columns, axis=1)
    st_autorefresh(interval=1, limit=2)
    return


def show_skips_handle_block():
    show_title(50, 'Handle skips')
    col_1, col_2 = st.columns([2, 3])

    with col_1:
        df = pd.DataFrame(
            [{col: int(st.session_state.dataset[col].isna().sum())
              for col in st.session_state.dataset.columns}]
        ).T
        df.columns = ['skips']
        st.write(df)

    with col_2:
        show_text(20, 'Choose the columns you want to handle')
        columns_option = st.multiselect(
            '',
            st.session_state.dataset.columns,
            key='fill_columns_multiselect')
        fill_option = st.selectbox(
            '',
            ('Delete', 'Fill in the mean value', 'Fill in the median value', 'Fill in the mode value',
             'Fill with in the custom value'), key='fill_option')
        if fill_option == 'Fill with in the custom value':
            show_text(20, 'Your replacing value')
            replacing_value = st.text_input('')
        if st.button("Confirm", key='confirm_fill_skips'):
            if fill_option == 'Delete':
                delete_skips(columns_option)
            elif fill_option == 'Fill in the mean value':
                replace_skips_by_mean(columns_option)
            elif fill_option == 'Fill in the median value':
                replace_skips_by_median(columns_option)
            elif fill_option == 'Fill in the mode value':
                replace_skips_by_mode(columns_option)
            elif fill_option == 'Fill with in the custom value':
                if not replacing_value == '':
                    replace_skips(columns_option, replacing_value)
                else:
                    show_exception(20, 'First insert the value to replace')
    return


def delete_skips(columns):
    st.session_state.dataset = st.session_state.dataset.copy().dropna(subset=columns)
    st_autorefresh(interval=1, limit=2)
    return


def replace_skips_by_mean(columns):
    try:
        for column in columns:
            mean_value = st.session_state.dataset[column].mean()
            st.session_state.dataset[column] = st.session_state.dataset[column].fillna(mean_value)
            st_autorefresh(interval=1, limit=2)
    except TypeError:
        show_exception(15, 'It\'s possible to find the mean value only for numbers')
    return


def replace_skips_by_median(columns):
    try:
        for column in columns:
            median_value = st.session_state.dataset[column].median(skipna=True)
            st.session_state.dataset[column] = st.session_state.dataset[column].fillna(median_value)
            st_autorefresh(interval=1, limit=2)
    except TypeError:
        show_exception(15, 'It\'s possible to find the median value only for numbers')
    return


def replace_skips_by_mode(columns):
    for column in columns:
        mode_value = st.session_state.dataset[column].mode()[0]
        st.session_state.dataset[column] = st.session_state.dataset[column].fillna(mode_value)
    st_autorefresh(interval=1, limit=2)
    return


def replace_skips(columns, replacing_value):
    for column in columns:
        st.session_state.dataset[column] = st.session_state.dataset[column].fillna(replacing_value)
    st_autorefresh(interval=1, limit=2)
    return


def show_value_counts_block():
    show_title(50, 'Column values')

    col1, col2 = st.columns([4, 1])

    with col2:
        show_text(20, 'Choose the columns you want to examine')
        column_option = st.selectbox('', st.session_state.dataset.columns, key='column_option')

        with col1:
            st.text(st.session_state.dataset[column_option].value_counts())
    return

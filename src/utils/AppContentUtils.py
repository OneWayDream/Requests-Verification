import io
import streamlit as st
from PIL import Image
from streamlit_autorefresh import st_autorefresh
from utils.PropertiesLoader import MainPropertiesLoader


def show_dataset_info(title_size, title_text):
    show_title(title_size, title_text)

    with st.container():
        buffer = io.StringIO()
        st.session_state.dataset.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)


def show_side_bar(image, title_color, text_color, title_size, text_size, title, text, next_button, previous_button):
    image = Image.open(image)
    st.sidebar.image(image)
    show_markdown_text(title_color, title_size, title, st.sidebar)
    show_markdown_text(text_color, text_size, text, st.sidebar)
    show_step_buttons(next_button, previous_button)
    return


def show_step_buttons(next_button, previous_button):
    if next_button and previous_button:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Previous step"):
                st.session_state.page -= 1
                st_autorefresh(interval=1, limit=2)

        with col2:
            if st.button("Next step"):
                st.session_state.page += 1
                st_autorefresh(interval=1, limit=2)

    if next_button and not previous_button:
        if st.sidebar.button("Next step"):
            st.session_state.page += 1
            st_autorefresh(interval=1, limit=2)

    if not next_button and previous_button:
        if st.sidebar.button("Previous step"):
            st.session_state.page -= 1
            st_autorefresh(interval=1, limit=2)

    return


def show_dataset():
    show_title(50, 'Your dataset', st)
    st.write(st.session_state.dataset.head(10))
    return


def draw_horizontal_line(elem=st):
    line = '<hr style="height:2px;border-width:0;color:gray;background-color:gray">'
    elem.markdown(line, unsafe_allow_html=True)
    return


def show_title(size, content, elem=st):
    show_markdown_text(MainPropertiesLoader.TITLE_COLOR, size, content, elem)


def show_text(size, content, elem=st):
    show_markdown_text(MainPropertiesLoader.TEXT_COLOR, size, content, elem)


def show_exception(size, content, elem=st):
    show_markdown_text(MainPropertiesLoader.EXCEPTION_COLOR, size, content, elem)


def show_size_bar_title(size, content):
    show_markdown_text(MainPropertiesLoader.SIDE_BAR_TITLE_COLOR, size, content, st.sidebar)


def show_size_bar_text(size, content):
    show_markdown_text(MainPropertiesLoader.SIDE_BAR_TEXT_COLOR, size, content, st.sidebar)


def show_success_text(size, content, elem=st):
    show_markdown_text(MainPropertiesLoader.SUCCESS_COLOR, size, content, elem)


def show_result_text(size, content, elem=st):
    show_markdown_text(MainPropertiesLoader.RESULT_COLOR, size, content, elem)


def show_markdown_text(color, size, content, elem):
    text = '<p style="font-family:sans-serif; color:' + color + '; font-size: ' + str(size) + 'px; text-align: ' \
                                                                                       'center; margin: 7% auto;">' \
                                                                                            + content + '</p> '
    elem.markdown(text, unsafe_allow_html=True)

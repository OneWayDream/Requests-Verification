import streamlit as st
from streamlit_autorefresh import st_autorefresh
from utils.PropertiesLoader import AdminPropertiesLoader


def show_dataset():
    show_title(50, 'Your dataset', st)
    st.write(st.session_state.dataset.head(10))
    return


def show_dataset_info(title_size, title_text):
    show_title(title_size, title_text)

    with st.container():
        buffer = io.StringIO()
        st.session_state.dataset.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)


def draw_horizontal_line(elem=st):
    line = '<hr style="height:2px;border-width:0;color:gray;background-color:gray">'
    elem.markdown(line, unsafe_allow_html=True)
    return


def show_title(size, content, elem=st):
    show_markdown_text(AdminPropertiesLoader.TITLE_COLOR, size, content, elem)


def show_subtitle(size, content, elem=st):
    show_markdown_text(AdminPropertiesLoader.SUB_TITLE_COLOR, size, content, elem)


def show_text(size, content, elem=st):
    show_markdown_text(AdminPropertiesLoader.TEXT_COLOR, size, content, elem)


def show_exception(size, content, elem=st):
    show_markdown_text(AdminPropertiesLoader.EXCEPTION_COLOR, size, content, elem)


def show_success_text(size, content, elem=st):
    show_markdown_text(AdminPropertiesLoader.SUCCESS_COLOR, size, content, elem)


def show_markdown_text(color, size, content, elem):
    text = '<p style="font-family:sans-serif; color:' + color + '; font-size: ' + str(size) + 'px; text-align: ' \
                                                                                       'center; margin: 7% auto;">' \
                                                                                            + content + '</p> '
    elem.markdown(text, unsafe_allow_html=True)

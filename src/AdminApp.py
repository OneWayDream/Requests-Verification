from utils import StyleUtils
from pages.AdminPanelPage import *


def main():
    StyleUtils.set_png_as_page_bg(AdminPropertiesLoader.BACKGROUND_IMAGE)
    show_admin_panel_content()
    return


if __name__ == '__main__':
    if 'success_changes' not in st.session_state:
        st.session_state.success_changes = False
    main()

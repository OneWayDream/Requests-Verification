import os
import signal
import time
from multiprocessing import Process

from utils.AdminContentUtils import *
from utils.PropertiesLoader import *


def show_admin_panel_content():

    prop = read_properties()
    update_time = prop['update-time']
    current_days = int(update_time // (60*60*24))
    current_hours = int((update_time // (60*60)) - current_days * 24)
    current_minutes = int((update_time // 60) - current_hours*60 - current_days*60*24)
    current_seconds = int(update_time % 60)
    current_store_limit = prop['store-limit']
    current_select_mode = prop['select-mode']

    show_title(50, 'Static model settings')
    show_subtitle(40, 'Update time')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_text(30, 'Days')
        days = st.number_input(label='Days', min_value=0, max_value=999, step=1, value=current_days, key='days_input')
    with col2:
        show_text(30, 'Hours')
        hours = st.number_input(label='Hours', min_value=0, max_value=23, step=1, value=current_hours,
                                key='hours_input')
    with col3:
        show_text(30, 'Minutes')
        minutes = st.number_input(label='Minutes', min_value=0, max_value=59, value=current_minutes, step=1,
                                  key='minutes_input')
    with col4:
        show_text(30, 'Seconds')
        seconds = st.number_input(label='Seconds', min_value=0, max_value=59, value=current_seconds, step=1,
                                  key='seconds_input')
    draw_horizontal_line()

    show_subtitle(40, 'Limit of new stored data')
    store_limit = st.number_input(label='Store limit', min_value=0, max_value=9999999, value=current_store_limit,
                                  step=1, key='store_limit')
    draw_horizontal_line()

    show_subtitle(40, 'Data selection mode')
    select_modes = ['All records', 'Anomaly search (COPOD)']
    select_mode = st.selectbox(label='', options=select_modes, index=select_modes.index(current_select_mode),
                               key='select_mode')
    draw_horizontal_line()

    relearn_checkbox = st.checkbox('Relearn model with new settings', value=False, key='relearn_checkbox_key')

    if st.button(label='Save changes', key='submit_button'):
        new_update_time = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
        if new_update_time < AdminPropertiesLoader.MIN_UPDATE_TIME:
            show_exception(20, 'You need to wait at least 30 minutes before a new retraining!')
        else:
            # сохраняем новые настройки
            prop['update-time'] = new_update_time
            prop['store-limit'] = store_limit
            prop['select-mode'] = select_mode
            write_properties(prop)
            st.session_state.success_changes = True
            # запускаем новый процесс-таймер
            restart_update_thread(update_time, prop['process-id'], relearn_checkbox)
            st_autorefresh(interval=1, limit=1)

    if st.session_state.success_changes:
        show_success_text(20, 'Done! Your settings have been successfully saved ^u^')
        st.session_state.success_changes = False
        st.balloons()
        st.snow()
    return


def restart_update_thread(update_time, process_id, first_execute):
    if process_id != -1:
        # os.kill(process_id, signal.SIGKILL)
        os.system("taskkill /f /im  mlflow_script.py")
    p = Process(target=update_process, args=(update_time, first_execute))
    p.start()


def update_process(update_time, first_execute):
    prop = read_properties()
    prop['process-id'] = os.getpid()
    write_properties(dist=prop)
    while True:
        if first_execute:
            os.system("python " + AdminPropertiesLoader.MLFLOW_PATH)
        time.sleep(update_time)
        if not first_execute:
            os.system("python " + AdminPropertiesLoader.MLFLOW_PATH)

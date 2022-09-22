from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import plotly.express as px

from utils.AppContentUtils import *


def show_clusterize_content():
    show_clusterize_customization_block()
    draw_horizontal_line()
    if st.session_state.clusterization_model is not None:
        with st.spinner('Wait for it...'):
            show_clusterization_plot_block()
            draw_horizontal_line()
            show_labels_info_block()
    return


def show_clusterize_customization_block():
    show_title(50, 'Customize the clusterization')

    show_text(25, 'The method of clusterization: ')
    method_option = st.selectbox(
        '',
        ['DBScan', 'MiniBatchKMeans'],
        key='method_select_box')

    if method_option == 'MiniBatchKMeans':
        draw_horizontal_line()
        show_overshoot_block(method_option)
        draw_horizontal_line()
        show_text(25, 'Clusters amount:')
        clusters_amount = st.number_input('', min_value=2, max_value=1000,
                                          value=(st.session_state.best_clusters_amount
                                                 if st.session_state.best_clusters_amount is not None
                                                 else 10),
                                          step=1, key='clusters_amount')

    if st.button("Confirm", key='confirm'):
        try:
            with st.spinner('Wait for it...'):
                if method_option == 'DBScan':
                    dbscan_model = DBSCAN().fit(st.session_state.requests_vectors)
                    dbscan_cluster_labels = dbscan_model.labels_
                    st.session_state.dataset.loc[:, 'clusterization_label'] = dbscan_cluster_labels
                    st.session_state.clusterization_model = dbscan_model
                elif method_option == 'MiniBatchKMeans':
                    mini_batch_means_model = MiniBatchKMeans(n_clusters=clusters_amount, random_state=10)
                    mini_batch_means_model.fit(st.session_state.requests_vectors)
                    mini_batch_means_cluster_labels = mini_batch_means_model.predict(st.session_state.requests_vectors)
                    st.session_state.dataset.loc[:, 'clusterization_label'] = mini_batch_means_cluster_labels
                    st.session_state.clusterization_model = mini_batch_means_model

            show_success_text(25, 'Done! You can head to the next page ^u^')
        except AttributeError:
            show_exception(15, 'First vectorize your requests on the previous page!')
    return


def show_overshoot_block(method_option):
    show_title(35, 'Overshoot the best amount of clusters:')

    show_title(25, 'Max cluster size:')
    max_cluster_size = st.number_input('', min_value=1, max_value=999999, value=1000, step=1, key='max_cluster_size')

    show_title(25, 'Max clusters amount (it may takes a lot of time to find your query)')
    max_clusters_amount = st.number_input('', min_value=1, max_value=1000, value=30, step=1, key='max_clusters_amount')

    best_clusters_amount = None

    if st.button("Search", key='search'):
        st.session_state.progress_bar = st.progress(0)
        result = cluster_size_searcher(method_option=method_option,
                                       vectors=st.session_state.requests_vectors,
                                       max_cluster_size=max_cluster_size,
                                       max_clusters_amount=max_clusters_amount)
        st.session_state.progress_bar.progress(1.0)
        del st.session_state.progress_bar
        df = pd.DataFrame(result, columns=['clusters_amount', 'max_cluster_size'])
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(df)

        with col2:
            min_value = df['max_cluster_size'].min()
            best_clusters_amount = df[df['max_cluster_size'] == min_value]['clusters_amount'].iloc[0]
            show_text(25, 'The best clusters amount: ' + str(best_clusters_amount))

        st.session_state.best_clusters_amount = best_clusters_amount
        return


@st.cache(show_spinner=False)
def cluster_size_searcher(method_option, vectors, max_cluster_size, max_clusters_amount):
    result = []
    n_clusters = 2
    current_max_size = max_cluster_size + 1
    while (current_max_size > max_cluster_size) and (n_clusters <= max_clusters_amount):
        model = None
        if method_option == 'MiniBatchKMeans':
            model = MiniBatchKMeans(n_clusters=n_clusters, random_state=10)
        labels = model.fit_predict(vectors)
        counts = np.bincount(labels)
        current_max_size = max(counts)
        result.append([n_clusters, current_max_size])

        st.session_state.progress_bar.progress(n_clusters / max_clusters_amount)
        n_clusters = n_clusters + 1
    return result


def show_clusterization_plot_block():
    show_title(50, 'Clusterization Plot')
    fig = build_clusterization_plot(st.session_state.requests_vectors,
                                    st.session_state.dataset['clusterization_label'])
    st.plotly_chart(fig, use_container_width=True)
    return


@st.cache(show_spinner=False)
def build_clusterization_plot(vectors, labels):
    x_selected = TSNE(n_components=2).fit_transform(vectors)
    visualisation_df = DataFrame()
    visualisation_df.loc[:, "x"] = x_selected[:, 0]
    visualisation_df.loc[:, "y"] = x_selected[:, 1]
    visualisation_df.loc[:, "labels"] = labels
    fig = px.scatter(visualisation_df, x="x", y="y", color="labels", size_max=60)
    return fig


def show_labels_info_block():
    show_title(50, 'Clusterization Labels')

    col1, col2 = st.columns(2)

    with col1:
        st.text(st.session_state.dataset['clusterization_label'].value_counts())

    with col2:
        show_text(20, 'Tip: try to get the smallest clusters here - it will be useful for marking')
    return
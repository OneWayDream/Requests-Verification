import warnings

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
# from yellowbrick.cluster import KElbowVisualizer
import hyperopt.hp as hp
import mlflow
from numpy.random import RandomState
from hyperopt import fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import *
from lightgbm import LGBMClassifier
from pyod.models.copod import COPOD
from utils.PropertiesLoader import *

from db.DbService import *
from db.Config import *
from db.Requests import Request

from utils.DataUtils import get_initial_data

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    recall = recall_score(actual, pred, average='macro')
    precision = precision_score(actual, pred, average='macro')
    accuracy = accuracy_score(actual, pred)
    f_macro = f1_score(actual, pred, average='macro')
    f_micro = f1_score(actual, pred, average='micro')
    f_weighted = f1_score(actual, pred, average='weighted')
    return recall, precision, accuracy, f_macro, f_micro, f_weighted


def print_models_info(mv):
    for m in mv:
        print("name: {}".format(m.name))
        print("latest version: {}".format(m.version))
        print("run_id: {}".format(m.run_id))
        print("current_stage: {}".format(m.current_stage))


def contains_check(s, arr):
    result = False
    for elem in arr:
        result = result or (s.find(elem) != -1)
        if result:
            break
    return result


def vectorize(dataset):
    vectors = dataset['vector'].apply(lambda s: str(s).lower())
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2),
                                 max_features=MlflowPropertiesLoader.VECTORIZER_MAX_FEATURES)
    tfid_vectors = vectorizer.fit_transform(vectors)
    return vectorizer, tfid_vectors


def clusterize(dataset):
    # find the optimal clusters amount
    # model = MiniBatchKMeans(random_state=10)
    # k_elbow = KElbowVisualizer(model, k=(1, MlflowPropertiesLoader.MAX_CLUSTERS_AMOUNT)).fit(dataset)

    # clusterization
    model = MiniBatchKMeans(
        n_clusters=mini_match_mean_clusterization(MlflowPropertiesLoader.MAX_CLUSTER_SIZE, dataset),
        random_state=10)
    model.fit(dataset)
    return model.labels_, model


def mini_match_mean_clusterization(MAX_CLUSTER_SIZE, X, max_clusters_amount=1000):
    n_clusters = 2
    current_max_size = MAX_CLUSTER_SIZE + 1
    min_size = MAX_CLUSTER_SIZE * 10
    min_clusters_amount = 0
    df = pd.DataFrame()
    while (current_max_size > MAX_CLUSTER_SIZE) and (n_clusters <= max_clusters_amount):
        mini_batch_means_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=10)
        mini_batch_means_model.fit(X)
        mini_batch_means_cluster_labels = mini_batch_means_model.predict(X)
        df.loc[:, 'mini_batch_means_labels'] = mini_batch_means_cluster_labels
        current_max_size = max(df['mini_batch_means_labels'].value_counts().to_numpy())
        print("(" + str(current_max_size) + ", " + str(n_clusters) + ") -> ", end='')

        if n_clusters % 10 == 0:
            print("")

        if min_size > current_max_size:
            min_size = current_max_size
            min_clusters_amount = n_clusters

        n_clusters = n_clusters + 1
    return min_clusters_amount


def marking(dataset):
    marks_dist = MlflowPropertiesLoader.MARKS_DIST
    # Предварительно помечаем все запросы как безопасные
    dataset['mark'] = [0] * data.shape[0]
    # Проходимся по всем строкам и ищем select-слова для каждого типа опасностей
    for r_type, r_info in marks_dist.items():
        filtered_dataset = dataset[
            dataset['vector'].apply(lambda x: contains_check(x.lower(), r_info.select_words))]
        # Проходимся по всем кластерам, которые содержат select-слова
        counters = filtered_dataset['clusterization_label'].value_counts().to_numpy()
        for i, label in enumerate(filtered_dataset['clusterization_label'].value_counts().index):
            if dataset[dataset['clusterization_label'] == label].iloc[0]['mark'] == 0:
                if counters[i] >= MlflowPropertiesLoader.MIN_ENTRIES_AMOUNT:
                    dataset.loc[dataset.clusterization_label == label, 'mark'] = r_info.mark
    return dataset


def optimize_hyperparams(dataset):
    X_train, X_validation, y_train, y_validation = train_test_split(
        dataset.drop(['mark'], axis=1),
        dataset['mark'], train_size=0.8, random_state=0)

    obj = HPOpt(X_train, X_validation, y_train, y_validation)

    # LightGBM parameters
    lgb_reg_params = {
        'learning_rate': hp.choice('learning_rate', np.arange(0.05, 0.31, 0.05)),
        'max_depth': hp.choice('max_depth', np.arange(5, 16, 1, dtype=int)),
        'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
        'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'n_estimators': 100,
    }
    lgb_fit_params = {
        'early_stopping_rounds': 10,
        'verbose': False
    }
    lgb_para = dict()
    lgb_para['reg_params'] = lgb_reg_params
    lgb_para['fit_params'] = lgb_fit_params
    # lgb_para['loss_func'] = lambda y, pred: 1 - recall_score(y, pred, average='macro')
    lgb_para['loss_func'] = lambda y, pred: 1 - f1_score(y, pred, average='weighted')

    ctb_opt = obj.process(fn_name='lgb_reg', space=lgb_para, trials=Trials(), algo=tpe.suggest, max_evals=100)
    result, trials = ctb_opt
    return result


def handle_new_entries(old_d, new_d, mode):
    # проверяем, есть ли новые запросы
    if new_d.shape[0] > 0:
        new_d = pd.DataFrame(new_d['vector'].unique().tolist(), columns=['vector'])
        if mode == 'Anomaly search (PYOD)':
            # склеиваем датасеты
            dataset = pd.concat([old_d, new_d], ignore_index=True)
            m, vectors = vectorize(dataset)
            copod_model = COPOD()
            # ищем аномалии
            copod_model.fit(vectors.toarray())
            dataset['label'] = copod_model.labels_
            new_d = dataset.loc[old_d.shape[0]::]
            new_d = new_d.loc[new_d['label'] == 1].drop(['label'], axis=1)
        new_d['add_flag'] = new_d['vector'].map(lambda x: x not in old_d['vector'].tolist())
        new_d = new_d.loc[new_d['add_flag'] == 1].drop(['add_flag'], axis=1)
        old_d = pd.concat([old_d, new_d], ignore_index=True)

        requests = list()
        for request in new_d['vector'].tolist():
            requests.append(Request(vector=request))
        add_new_requests(requests)

        truncate_table(NewRequest)
    return old_d


def init_db(max_new_requests):
    init(DRIVER + "://" + USER + ":" + PASSWORD + "@" + HOST + ":" + PORT + "/" + DB_NAME,
         MAX_REQUESTS,
         max_new_requests
         )
    if get_count_of_requests() == 0:
        requests = list()
        for request in get_initial_data():
            requests.append(Request(vector=request))
        add_new_requests(requests)


class HPOpt(object):

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        # try:
        result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        # except Exception as e:
        #     return {'status': STATUS_FAIL,
        #             'exception': str(e)}
        return result, trials

    def lgb_reg(self, para):
        reg = LGBMClassifier(**para['reg_params'])
        return self.train_reg(reg, para)

    def train_reg(self, reg, para):
        reg.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])
        pred = reg.predict(self.x_test)
        loss = para['loss_func'](self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}


if __name__ == "__main__":

    prop = read_properties()

    warnings.filterwarnings("ignore")
    np.random.seed(40)
    init_db(prop['store-limit'])

    saved_dataset = pd.DataFrame(list(map(lambda x: x.vector, get_all_requests())), columns=['vector'])
    fresh_dataset = pd.DataFrame(list(map(lambda x: x.vector, get_all_new_requests())), columns=['vector'])

    data = handle_new_entries(saved_dataset, fresh_dataset, prop['select-mode'])

    data = data.dropna(subset=['vector'])
    print("Dataset is loaded")
    mlflow.set_experiment("Requests Vulnerabilities")
    with mlflow.start_run():

        # requests vectorization
        vectorizer, tfid_vectors = vectorize(data)
        prepared_dataset = pd.DataFrame(tfid_vectors.toarray(), columns=['v_' + str(i) for i in
                                                                         range(
                                                                             MlflowPropertiesLoader.VECTORIZER_MAX_FEATURES)])
        print("Vectorization stage is done")

        # clusterization
        data.loc[:, 'clusterization_label'], clusterization_model = clusterize(prepared_dataset)
        print("Clusterization stage is done")

        # marking
        marking(dataset=data)
        data.index = [i for i in range(data.shape[0])]
        prepared_dataset['mark'] = data['mark']
        print("Marking stage is done")
        print(prepared_dataset['mark'].value_counts())

        # Hyperparameter Optimization
        best_params = optimize_hyperparams(prepared_dataset)
        print(best_params)
        best_params['colsample_bytree'] = 0.3 + 0.1 * (best_params['colsample_bytree'])
        best_params['learning_rate'] = 0.05 + 0.05 * (best_params['learning_rate'])
        print("Hyperparameter Optimization stage is done")
        print(best_params)

        # Model learning and evaluation
        x_train, x_test, y_train, y_test = train_test_split(
            prepared_dataset.drop(['mark'], axis=1),
            prepared_dataset['mark'], train_size=0.8, random_state=0)
        model = LGBMClassifier(**best_params)
        model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=10)

        predicted_qualities = model.predict(x_test)

        (recall, precision, accuracy, f_macro, f_micro, f_weighted) = eval_metrics(y_test, predicted_qualities)

        print("LGBMClassifier model (learning_rate=%f, max_depth=%d, min_child_weight=%d, colsample_bytree=%f, "
              "subsample=%f):" % (best_params['learning_rate'], best_params['max_depth'],
                                  best_params['min_child_weight'], best_params['colsample_bytree'],
                                  best_params['subsample']))
        print(classification_report(y_test, predicted_qualities))

        mlflow.log_param("learning_rate", best_params['learning_rate'])
        mlflow.log_param("max_depth", best_params['max_depth'])
        mlflow.log_param("min_child_weight", best_params['min_child_weight'])
        mlflow.log_param("colsample_bytree", best_params['colsample_bytree'])
        mlflow.log_param("subsample", best_params['subsample'])
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F_score_macro", f_macro)
        mlflow.log_metric("F_score_micro", f_micro)
        mlflow.log_metric("F_score_weighed", f_weighted)

        # Final model learning
        lgbm_model = LGBMClassifier(**best_params)
        lgbm_model.fit(prepared_dataset.drop(['mark'], axis=1),
                       prepared_dataset['mark'])

        mlflow.sklearn.log_model(vectorizer, prop['vectorization_model'])
        mlflow.sklearn.log_model(clusterization_model, prop['clusterization_model'])
        mlflow.sklearn.log_model(lgbm_model, prop['classification_model'])

    print(mlflow.last_active_run())
    dist = mlflow.last_active_run().to_dictionary()
    prop['experiment_id'] = dist['info']['experiment_id']
    prop['run_id'] = dist['info']['run_id']
    marks = prepared_dataset['mark'].unique().tolist()
    marks.sort()
    prop['model_marks'] = marks
    write_properties(prop)

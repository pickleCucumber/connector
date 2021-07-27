import numpy as np
import getpass
#import cx_Oracle as c
import pandas as pd
import pandas.io.sql as psql
import matplotlib.pyplot as plt

import datetime
import dateutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import metrics, linear_model, tree
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.tree import _tree

pd.set_option("display.max_columns", 200)
import sql_query
from operator import itemgetter
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Здесь указываются параметры доступа к PL/SQL
password = "Avquge>45"
ip = 'adress'
port = port
SID = 'DB'
dsn_tns = c.makedsn(ip, port, SID)
account = 'name_account'


def data_loading(period_from, period_to, new=True, file=None):
    """
    Загрузка данных. Либо запускаются sql-запросы, либо читается ранее сохранённый файл.
    """
    if not new:
        print('Loading data from file')
        data = pd.read_excel(file)
        return data
    else:
        db = c.connect(account, password, dsn_tns)
        print('Quering socdem')
        query_socdem = sql_query.generate_query_socdem(period_from, period_to)
        data_query_socdem = pd.io.sql.read_sql(query_socdem, con=db,
                                               parse_dates=['SEND_MONTH', 'SEND_DT', 'ACTIVE_DT', 'BIRTH_DT',
                                                            'FIRST_ANY_TRN_DT',
                                                            'ACT_MONTH', 'OUTBOUND_DATE', 'REGISTRATION_DT'])

        print('Quering lastcards')
        query_lastcards = sql_query.generate_query_lastcards(period_from, period_to)
        data_query_lastcards = pd.io.sql.read_sql(query_lastcards, con=db,
                                                  parse_dates=['LAST_CARD_SEND_DT'])

        print('Quering credits')
        query_credits = sql_query.generate_query_credits(period_from, period_to)
        data_query_credits = pd.io.sql.read_sql(query_credits, con=db,
                                                parse_dates=['FIRST_AGR_DATE', 'LAST_NONCARD_OPEN_DT',
                                                             'LAST_AGR_CLOSE_DT_ALL_CLOSED',
                                                             'MIN_POS_PLAN_CLOSE_DT', 'MAX_POS_PLAN_CLOSE_DT'])

        print('Quering dlq')
        query_dlq = sql_query.generate_query_dlq(period_from, period_to)
        data_query_dlq = pd.io.sql.read_sql(query_dlq, con=db)

        print('Quering bki')
        query_bki = sql_query.generate_query_bki(period_from, period_to)
        data_query_bki = pd.io.sql.read_sql(query_bki, con=db)

        print('Merging')
        data = pd.merge(data_query_socdem, data_query_lastcards, on=['CUSTOMER_ID_SB8', 'SEND_DT'], how='left')
        data = pd.merge(data, data_query_dlq, on=['CUSTOMER_ID_SB8', 'SEND_DT'], how='left')
        data = pd.merge(data, data_query_credits, on=['CUSTOMER_ID_SB8', 'SEND_DT'], how='left')
        data = pd.merge(data, data_query_bki, on=['CUSTOMER_ID_SB8', 'SEND_DT'], how='left')
        db.close()

        return data


def feature_stat(data_, feature, target_name, only_graph=False):
    """
    Показывает статистику о признаке: количества, частоты, график и IV.
    """
    data = data_.copy()
    if data[feature].isnull().any():
        if str(data[feature].dtype) == 'category':
            data[feature].cat.add_categories(['None'], inplace=True)
        data[feature].fillna('None', inplace=True)

    if only_graph == False:
        print('Counts:')
        print(data.groupby(feature)[target_name].count())
        print('Frequencies:')
        print(data[feature].value_counts(normalize=True, dropna=False))
    else:
        pass
    x = [i for i in data.groupby(feature)[target_name].count().index]

    if data[feature].isnull().any():
        if str(data[feature].dtype) == 'category':
            data[feature].cat.add_categories(['None'], inplace=True)
        data[feature].fillna('None', inplace=True)

    y1 = [i for i in data.groupby(feature)[target_name].count().values]
    y2 = [i for i in data.groupby(feature)[target_name].mean().values]
    ind = np.arange(len(data[feature].unique()))
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.bar(ind, y1, align='center', width=0.4, alpha=0.7)
    ax1.set_xlabel(feature)
    ax1.set_ylabel('Counts', color='b')
    ax1.tick_params('y1', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(ind, y2, 'r')
    ax2.set_ylabel('Mean rate', color='r')
    ax2.tick_params('y2', colors='r')
    plt.xticks(ind, x, rotation=45)
    ax1.set_xticklabels(x, rotation=35)
    plt.grid(False)
    plt.show()
    if only_graph == False:
        _, iv = calc_iv(data, target_name, feature)
        print('IV: ', iv)
    else:
        pass


def cont_split(data, feature, target, leafs, bins_dict, only_draw=False, no_draw=False, auto_calc=False):
    """
    Разбиение признака на бины c помощью древа решений. Также здесь округляются значения. При желании можно вывести ещё и график или только показать график и не делать разбиение.
    """
    bins_dict = bins_dict
    x = data[feature]
    y = data[target]

    clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=0.05,
                                      min_samples_leaf=0.05, min_weight_fraction_leaf=0.1, max_features=1,
                                      random_state=None,
                                      max_leaf_nodes=leafs, class_weight='balanced', presort=False)

    clf = clf.fit(x.values.reshape(-1, 1), y)
    # col_to_cat2 = ['MAX_DLQ_LENGTH_300', 'DLQ_COUNT_1D', 'DLQ_COUNT_1D_300R',
    #              'QUANNON14_ACT', 'QUAN_ADVANCED_CLOSED', 'QUAN_CLOSED_NON14', 'ACT_OR_SELF_PREVIOUS_CARDS',
    #             'QUANNON14', 'BKI_AGR_COUNT', 'BKI_ACTIVE_AGR_COUNT', 'BKI_DELAY_MORE', 'BKI_DELAY_30', 'BKI_DELAY_60',
    #              'BKI_DELAY_90', 'BKI_COUNT_ACTIVE_DELAY_30', 'Quan_act_all']

    h = tree_to_code1(clf, 'a')
    h = list(h)
    h.append(max(x))
    if feature == 'age_at_send' or feature == 'Quan_all':
        d = {k: np.round(v, 0) for k, v in zip(range(len(h)), h)}
    elif data[feature].max() > 500:
        d = {k: np.round(v, -2) for k, v in zip(range(len(h)), h)}
    else:
        d = {k: v for k, v in zip(range(len(h)), h)}
    if len(set(d.values())) < len(list(d.values())):
        return None
    if auto_calc == True:
        temp_series = pd.DataFrame({feature: pd.cut(data[feature], bins=[v for v in d.values()]), target: data[target]})
        temp_series[feature].cat.add_categories(['No value'], inplace=True)
        return calc_iv(temp_series, target, feature)[1]

    if no_draw == False:
        data.groupby(pd.cut(x, bins=[v for v in d.values()]))[target].mean().plot(kind='line')
        temp_series = pd.DataFrame({feature: pd.cut(data[feature], bins=[v for v in d.values()]), target: data[target]})
        temp_series[feature].cat.add_categories(['No value'], inplace=True)

        print(pd.cut(x, bins=[v for v in d.values()]).value_counts(normalize=True, dropna=False))

        print('IV: ', calc_iv(temp_series, target, feature)[1])
    else:
        pass

    bins_dict[feature] = [v for v in d.values()]

    if only_draw == False:
        return pd.cut(data[feature], bins=[v for v in d.values()]), bins_dict
    else:
        pass


def tree_to_code1(tree, feature_names):
    '''
    Вытаскивает пороги из древа решений.
    '''

    tr = [0]
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            if threshold not in tr: tr.append(threshold)
            recurse(tree_.children_left[node], depth + 1)
            if threshold not in tr: tr.append(threshold)
            recurse(tree_.children_right[node], depth + 1)
        else:
            pass

    recurse(0, 1)
    return sorted(tr)


def calc_iv(data, target, feature):
    """
    Считает данные о переменной.
    """
    df = pd.DataFrame(index=data[feature].unique(),
                      data={'% responders': data.groupby(feature)[target].sum() / np.sum(data[target])})
    df['% non-responders'] = (data.groupby(feature)[target].count() - data.groupby(feature)[target].sum()) / (
                len(data[target]) - np.sum(data[target]))
    df['WOE'] = np.log(df['% responders'] / df['% non-responders'])
    df['DG-DB'] = df['% responders'] - df['% non-responders']
    df['IV'] = df['WOE'] * df['DG-DB']
    return df, np.sum(df['IV'])


def split_best_iv(data, feature, target_name):
    """
    Находит оптимальное разбиение переменной на бины по IV: меняет количество листьев в дереве от 2 до 15 и останавливается, когда IV перестаёт расти.
    """
    best_iv = 0
    for i in range(2, 15):
        iv_temp = cont_split(data, feature, target_name, i, {}, only_draw=False, no_draw=True, auto_calc=True)
        if iv_temp == None:
            if i > 2:
                return cont_split(data, feature, target_name, i - 1, {}, only_draw=False, no_draw=True)
            else:
                return 'Bad'

        if iv_temp > best_iv:
            best_iv = iv_temp
        else:
            if i == 2:
                return 'Bad'
            else:
                return cont_split(data, feature, target_name, i - 1, {}, only_draw=False, no_draw=True)


def initial_preprocess(data, verbose=0):
    """
    Первичная обработка данных. Отбрасываются 3 переменных с обозначением клиента: 2 образовались из-за merge, одна просто лишняя, но была нужна в запросе. Далее создаются 2 новые переменные и заполняются значения для ACTIVATION_TYPE. После этого отбрасываются строки с незаполненными значениями дохода и со значениями выше или ниже 99 или 1 процентиля.
    """
    # These fields appeared due to merging.
    data.drop(['CUSTOMER_RK_SB8', 'CUSTOMER_ID_x', 'CUSTOMER_ID_y'], axis=1, inplace=True)
    data.drop(['DLQ_COUNT_7D', 'DLQ_COUNT_30D', 'DLQ_COUNT_90D', 'DLQ_COUNT_7D_300R', 'DLQ_COUNT_30D_300R',
               'DLQ_COUNT_90D_300R'
               ], axis=1, inplace=True)
    data['Quan_all'] = data['QUANNON14'] + data['BKI_AGR_COUNT']
    data['Quan_act_all'] = data['BKI_ACTIVE_AGR_COUNT'] + data['QUANNON14_ACT']
    data['ACTIVATION_TYPE'].fillna('Not activated', inplace=True)

    data.dropna(subset=['PERSONAL_INCOME'], inplace=True)
    if verbose == 1:
        print(data.shape)
    data = data[(data.PERSONAL_INCOME < np.percentile(data.PERSONAL_INCOME, 99)) & (
                data.PERSONAL_INCOME > np.percentile(data.PERSONAL_INCOME, 1))]
    if verbose == 1:
        print('After initial preprocess data shape is {}.'.format(data.shape))

    return data


def generate_target_variable(data_, action_type='act', period1=0, period2=30, method='self', verbose=0):

    data = data_.copy()
    if action_type not in ['act', 'util', 'tr']:
        print("Action_type should be one of the following: 'act', 'util', 'tr'")
        return

    if method not in ['self', 'non_self']:
        print("method should be one of the following: 'self', 'non_self'")
        return

    target_name = str(method) + str(action_type) + '_' + str(period1) + '_' + str(period2)
    data[target_name] = 0

    if method == 'self':
        if action_type == 'act':
            data.loc[(data['ACTIVE_DT'].dt.day.isnull() == False)
                     & ((data["ACTIVE_DT"] - data["SEND_DT"]).dt.days <= period2)
                     & ((data["ACTIVE_DT"] - data["SEND_DT"]).dt.days > period1)
                     & (data['ACTIVATION_TYPE'].isin(['Inbound', 'SMS']))
                     & (pd.to_datetime(data["ACTIVE_DT"]).dt.date < pd.to_datetime(data["OUTBOUND_DATE"]).dt.date)
                     | (data['SELFACTIVATION_ATTEMPT'] == 1), target_name] = 1

        elif action_type == 'util':
            data.loc[((data['ACTIVE_DT'].dt.day.isnull() == False)
                      & ((data["ACTIVE_DT"] - data["SEND_DT"]).dt.days <= period2)
                      & ((data["ACTIVE_DT"] - data["SEND_DT"]).dt.days > period1)
                      & (data['ACTIVATION_TYPE'].isin(['Inbound', 'SMS']))
                      & (pd.to_datetime(data["ACTIVE_DT"]).dt.date < pd.to_datetime(data["OUTBOUND_DATE"]).dt.date)
                      | (data1['SELFACTIVATION_ATTEMPT'] == 1))
                     & (data['FIRST_ANY_TRN_DT'].dt.date.isnull() == False), target_name] = 1

        elif action_type == 'tr':
            data.loc[((data['ACTIVE_DT'].dt.day.isnull() == False)
                      & ((data["ACTIVE_DT"] - data["SEND_DT"]).dt.days <= period2)
                      & ((data["ACTIVE_DT"] - data["SEND_DT"]).dt.days > period1)
                      & (data['ACTIVATION_TYPE'].isin(['Inbound', 'SMS']))
                      & (pd.to_datetime(data["ACTIVE_DT"]).dt.date < pd.to_datetime(data["OUTBOUND_DATE"]).dt.date)
                      | (data1['SELFACTIVATION_ATTEMPT'] == 1))
                     & (data['FIRST_ANY_TRN_DT'].dt.date.isnull() == False)
                     & (data['TRANS_NUM'] >= 5), target_name] = 1

    if method == 'non_self':
        if action_type == 'act':
            data.loc[(data['ACTIVE_DT'].dt.day.isnull() == False)
                     & ((data["ACTIVE_DT"] - data["SEND_DT"]).dt.days <= period2)
                     & ((data["ACTIVE_DT"] - data["SEND_DT"]).dt.days > period1)
                     & (((data['ACTIVATION_TYPE'].isin(['Inbound', 'SMS']) & (
                        pd.to_datetime(data["ACTIVE_DT"]).dt.date >= pd.to_datetime(
                    data["OUTBOUND_DATE"]).dt.date))) | (data['ACTIVATION_TYPE'] == 'Outbound')), target_name] = 1

        elif action_type == 'util':
            data.loc[(data['ACTIVE_DT'].dt.day.isnull() == False)
                     & ((data["ACTIVE_DT"] - data["SEND_DT"]).dt.days <= period2)
                     & ((data["ACTIVE_DT"] - data["SEND_DT"]).dt.days > period1)
                     & (((data['ACTIVATION_TYPE'].isin(['Inbound', 'SMS']) & (
                        pd.to_datetime(data["ACTIVE_DT"]).dt.date >= pd.to_datetime(
                    data["OUTBOUND_DATE"]).dt.date))) | (data['ACTIVATION_TYPE'] == 'Outbound'))
                     & (data['FIRST_ANY_TRN_DT'].dt.date.isnull() == False), target_name] = 1

        elif action_type == 'tr':
            data.loc[(data['ACTIVE_DT'].dt.day.isnull() == False)
                     & ((data["ACTIVE_DT"] - data["SEND_DT"]).dt.days <= period2)
                     & ((data["ACTIVE_DT"] - data["SEND_DT"]).dt.days > period1)
                     & (((data['ACTIVATION_TYPE'].isin(['Inbound', 'SMS']) & (
                        pd.to_datetime(data["ACTIVE_DT"]).dt.date >= pd.to_datetime(
                    data["OUTBOUND_DATE"]).dt.date))) | (data['ACTIVATION_TYPE'] == 'Outbound'))
                     & (data['FIRST_ANY_TRN_DT'].dt.date.isnull() == False)
                     & (data['TRANS_NUM'] >= 5), target_name] = 1

    data.drop(['ACTIVATION_TYPE', 'FIRST_ANY_TRN_DT', 'TRANS_NUM'], axis=1, inplace=True)

    if verbose == 1:
        print('After generating target data shape is {}.'.format(data.shape))
    return data, target_name

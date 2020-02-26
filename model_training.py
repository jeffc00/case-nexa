import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

summary = pd.read_csv('data/summary.csv')

summary_df = summary.sort_values(['day', 'assessment_end_time']).reset_index(drop=True)

summary_df['temp_cat'] = pd.cut(summary_df['temperature'],
                                bins=[float('-inf'), 36.5, 37.5, float('inf')],
                                labels=['hypothermia', 'normal', 'fever'])

priority_map = {'normal': 1, 'urgent': 2}
summary_df['priority_enc'] = summary_df['priority'].map(priority_map)

pain_map = {'no pain': 1, 'moderate pain': 2, 'severe pain': 3}
summary_df['pain_enc'] = summary_df['pain'].map(pain_map)

temp_map = {'hypothermia': 2, 'normal': 1, 'fever': 2}
summary_df['temp_cat_enc'] = summary_df['temp_cat'].map(temp_map)

summary_df = pd.get_dummies(summary_df, columns=['pain', 'temp_cat'])

summary_df['pain_severe pain:temp_cat_hypothermia'] = \
summary_df['pain_severe pain'] * summary_df['temp_cat_hypothermia']
summary_df['pain_moderate pain:temp_cat_normal'] = \
summary_df['pain_moderate pain'] * summary_df['temp_cat_normal']
summary_df['pain_severe pain:temp_cat_fever'] = \
summary_df['pain_severe pain'] * summary_df['temp_cat_fever']


# def create_new_features(df):
#     '''
#     Given a dataframe, returns dataframe with new added features:
#         free_doctors: number of doctors available after patient has passed triage
#         queue_size: queue size after patient has passed triage
#         urgents_on_queue: number of urgent priority patients on queue after patient has passed triage
#     '''
    
#     free_docs = [6]
#     q_urg = [list()]
#     q_norm = [list()]
#     q_real = [list()]
#     urgs_on_q = [0]
#     q_size = 0
#     nonbusy_docs = 0
#     consult_endtime_ls = [df.loc[0, 'consultation_end_time']]
    
#     for i in range(1, len(df)):
#         nonbusy_docs = (df.loc[i, 'assessment_end_time'] >= consult_endtime_ls).sum()
#         consult_endtime_ls = consult_endtime_ls[nonbusy_docs:]

#         if free_docs[i - 1] > 1:
#             free_docs.append(free_docs[i - 1] - 1 + nonbusy_docs)
#             consult_endtime_ls.append(df.loc[i, 'consultation_end_time'])
#             consult_endtime_ls = sorted(consult_endtime_ls)
#             q_urg.append(q_urg[i - 1])
#             q_norm.append(q_norm[i - 1])
#             q_real.append(q_urg[i] + q_norm[i])
#             urgs_on_q.append(urgs_on_q[i - 1])
#         elif nonbusy_docs == 0:
#             free_docs.append(0)
#             if df.loc[i, 'priority'] == 'urgent':
#                 urgs_on_q.append(urgs_on_q[i - 1] + 1)
#                 q_urg.append(q_urg[i - 1] + [i])
#                 q_norm.append(q_norm[i - 1])
#             else:
#                 urgs_on_q.append(urgs_on_q[i - 1])
#                 q_urg.append(q_urg[i - 1])
#                 q_norm.append(q_norm[i - 1] + [i])
#             q_real.append(q_urg[i] + q_norm[i])
#             q_size = len(q_real[i])
#         elif q_size == 0:
#             free_docs.append(free_docs[i - 1] - 1 + nonbusy_docs)
#             q_urg.append(q_urg[i - 1])
#             q_norm.append(q_norm[i - 1])
#             q_real.append(q_urg[i] + q_norm[i])
#             urgs_on_q.append(urgs_on_q[i - 1])
#             consult_endtime_ls.append(df.loc[i, 'consultation_end_time'])
#             consult_endtime_ls = sorted(consult_endtime_ls)
#         else:
#             free_docs.append(np.max([free_docs[i - 1] + nonbusy_docs - q_size, 0]))
#             if df.loc[i, 'priority'] == 'urgent':
#                 urgs_on_q.append(urgs_on_q[i - 1] + 1)
#                 q_urg.append(q_urg[i - 1] + [i])
#                 q_norm.append(q_norm[i - 1])
#             else:
#                 urgs_on_q.append(urgs_on_q[i - 1])
#                 q_urg.append(q_urg[i - 1])
#                 q_norm.append(q_norm[i - 1] + [i])
#             q_real.append(q_urg[i] + q_norm[i])
#             q_size = len(q_real[i])
#             while q_size > 0 and nonbusy_docs > 0:
#                 patient = q_real[i].pop(0)
#                 if df.loc[patient, 'priority'] == 'urgent':
#                     urgs_on_q[i] -= 1
#                     q_urg[i].pop(0)
#                 else:
#                     q_norm[i].pop(0)
#                 consult_endtime_ls.append(df.loc[patient, 'consultation_end_time'])
#                 consult_endtime_ls = sorted(consult_endtime_ls)
#                 q_size -= 1
#                 nonbusy_docs -= 1
                
# #     new_features = pd.DataFrame([free_docs, pd.Series(q_real).apply(lambda x: len(x)), urgs_on_q],
# #                                 index=['free_doctors', 'queue_size', 'urgents_on_queue']).T
    
#     new_features = pd.DataFrame([pd.Series(q_real).apply(lambda x: len(x))],
#                             index=['queue_size']).T
    
#     return pd.concat([df, new_features], axis=1)


# summary_mod = pd.DataFrame(columns=summary_df.columns)
# for i in summary_df['day'].unique():
#     df = create_new_features(summary_df[summary_df['day'] == i].reset_index(drop=True))
#     summary_mod = pd.concat([summary_mod, df], axis=0)

# uncomment the line below if you uncomment the code block above
# summary_df = summary_mod

cols_to_drop = ['Unnamed: 0', 'arrival_time', 'assessment_start_time', 'patient',
                'priority','duration', 'temperature', 'consultation_start_time',
                'pain_no pain', 'pain_moderate pain', 'pain_severe pain',
                'temp_cat_hypothermia', 'temp_cat_normal', 'temp_cat_fever']

summary_df.drop(cols_to_drop, axis=1, inplace=True)

df_train = summary_df[summary_df['day'] <= 45]
df_test = summary_df[summary_df['day'] > 45]


best_params_ls = []
train_rmse_ls = []
val_rmse_ls = []
def get_best_params_and_rmse(df):
    '''
    Given df, returns:
        - a list of the parameters of the best model for each day
        - a list of the cv train rmse of the best model for each day
        - a list of the cv validation rmse of the best model for each day
    '''
    
    for i in tqdm(df['day'].unique()):
        X_train = df[df['day'] == i].drop(['day', 'consultation_end_time'], axis=1)
        y_train = df.loc[df['day'] == i, 'consultation_end_time']

        tscv = TimeSeriesSplit(n_splits=5)

        ridge = Ridge(normalize=True, random_state=42)
        ridge_params = {'alpha': ss.uniform(0, 6)}

        rscv = RandomizedSearchCV(ridge,
                                ridge_params,
                                scoring='neg_root_mean_squared_error',
                                n_iter=100,
                                n_jobs=-1,
                                cv=tscv,
                                verbose=1,
                                random_state=42,
                                return_train_score=True)

        rscv.fit(X_train, y_train)

        best_params_ls.append(rscv.best_params_)

        cv_results = pd.DataFrame(rscv.cv_results_)
        train_rmse_ls.append(-cv_results.loc[rscv.best_index_, 'mean_train_score'])
        val_rmse_ls.append(-cv_results.loc[rscv.best_index_, 'mean_test_score'])
        
    return best_params_ls, train_rmse_ls, val_rmse_ls

best_params_ls, train_rmse_ls, val_rmse_ls = get_best_params_and_rmse(df_train)

results_train_df = pd.DataFrame([best_params_ls, train_rmse_ls, val_rmse_ls],
                                index=['best_params', 'train_rmse', 'val_rmse']).T


class MetaEstimator():
    def __init__(self, best_params):
        self.best_params = best_params
            
    def fit(self, X, y):
        days = X['day'].unique()
        X_ls = [X[X['day'] == i] for i in days]
        y_ls = [y[y['day'] == i] for i in days]
        
        best_estimator_ls = []
        for i in range(len(self.best_params)):
            params = self.best_params[i]
            ridge = Ridge(normalize=False, random_state=42, **params)
            X = X_ls[i].drop('day', axis=1)
            y = y_ls[i].drop('day', axis=1)
            best_estimator_ls.append(ridge.fit(X, y))
            
        self.estimators_ls = best_estimator_ls
        
        intercept = 0
        coef = np.zeros_like(self.estimators_ls[0].coef_)
        for estimator in self.estimators_ls:
            intercept += estimator.intercept_
            coef += estimator.coef_
        n_estimators = len(self.estimators_ls)
        
        self.coefs = np.append(intercept, coef[0]) / n_estimators
    
    def predict(self, X):
        days = X['day'].unique()
        X_ls = [X[X['day'] == i] for i in days]

        preds = np.empty(0)
        for i in range(len(X_ls)):
            C = pd.Series(np.ones(len(X_ls[i])), name='C', index=X_ls[i].index)
            X_new = pd.concat([C, X_ls[i]], axis=1).drop('day', axis=1)
            preds = np.append(preds, X_new @ self.coefs)
        return preds


scaler = StandardScaler()

df_train_new = df_train.drop(['day', 'consultation_end_time'], axis=1)
df_test_new = df_test.drop(['day', 'consultation_end_time'], axis=1)

X_train_scaled = \
pd.concat([pd.DataFrame(scaler.fit_transform(df_train_new), columns=df_train_new.columns),
            df_train['day']], axis=1)
X_test_scaled = \
pd.concat([pd.DataFrame(scaler.transform(df_test_new), columns=df_test_new.columns, index=df_test_new.index),
            df_test['day']], axis=1)

y_train = df_train[['day', 'consultation_end_time']]
y_test = df_test[['day', 'consultation_end_time']]

meta = MetaEstimator(results_train_df['best_params'])
meta.fit(X_train_scaled, y_train)

joblib.dump(meta, 'consultation_end_time_estimator.pkl')
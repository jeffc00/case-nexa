import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def main():
    file_path = 'data/summary.csv'
    summary_df = load_data(file_path)
    summary_df = preprocess_data(summary_df)

    df_train = summary_df[summary_df['day'] <= 45]
    df_test = summary_df[summary_df['day'] > 45]

    results_train_df = get_best_params_and_error(df_train)
    train_error, val_error = get_avg_train_val_errors(results_train_df, df_train)
    meta, features = train_model(summary_df, df_train, df_test, results_train_df)

    train_error, test_error = get_model_results(meta, df_train, df_test)

    overall_error = get_overall_error(meta, summary_df, results_train_df)

    save_trained_model_coefs(meta)

    save_trained_model_features(features)

    
def load_data(file_path):
    '''
    Load and sort training data.
    '''
    summary = pd.read_csv(file_path)
    summary_df = summary.sort_values(['day', 'assessment_end_time']).reset_index(drop=True)

    return summary_df


def preprocess_data(summary_df):
    '''
    Encode categorical features and return a new dataframe with unnecessary columns removed.
    '''
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

    cols_to_drop = ['Unnamed: 0', 'arrival_time', 'assessment_start_time', 'patient',
                    'priority', 'temperature', 'consultation_start_time',
                    'pain_no pain', 'pain_moderate pain', 'pain_severe pain',
                    'temp_cat_hypothermia', 'temp_cat_normal', 'temp_cat_fever']

    summary_df.drop(cols_to_drop, axis=1, inplace=True)

    return summary_df


def get_best_params_and_error(df):
    '''
    Given df, returns a dataframe containing:
        - the parameters of the best model for each day
        - the cv train MSLE of the best model for each day
        - the cv validation MSLE of the best model for each day
    '''
    best_params_ls = []
    train_error_ls = []
    val_error_ls = []
    
    for i in tqdm(df['day'].unique()):
        X_train = df[df['day'] == i].drop(['day', 'duration'], axis=1)
        y_train = df.loc[df['day'] == i, 'duration']

        tscv = TimeSeriesSplit(n_splits=5)

        ridge = Ridge(normalize=True, random_state=42)
        ridge_params = {'alpha': ss.uniform(0, 6)}

        rscv = RandomizedSearchCV(ridge,
                                ridge_params,
                                scoring=make_scorer(mean_squared_log_error, greater_is_better=False),
                                n_iter=100,
                                n_jobs=-1,
                                cv=tscv,
                                verbose=1,
                                random_state=42,
                                return_train_score=True)

        rscv.fit(X_train, y_train)

        best_params_ls.append(rscv.best_params_)

        cv_results = pd.DataFrame(rscv.cv_results_)
        train_error_ls.append(-cv_results.loc[rscv.best_index_, 'mean_train_score'])
        val_error_ls.append(-cv_results.loc[rscv.best_index_, 'mean_test_score'])

        results_train_df = pd.DataFrame([best_params_ls, train_error_ls, val_error_ls],
                                        index=['best_params', 'train_msle', 'val_msle']).T
        
    return results_train_df


def get_avg_train_val_errors(results_train_df, df_train):
    '''
    Returns train and validation errors during training.
    '''
    n_obs_per_day = df_train['day'].value_counts().sort_index().values
    n_obs_total = n_obs_per_day.sum()

    train_error = np.sqrt((results_train_df['train_msle']*n_obs_per_day).sum() / n_obs_total)
    val_error = np.sqrt((results_train_df['val_msle']*n_obs_per_day).sum() / n_obs_total)

    print(f'train RMSLE: {train_error}')
    print(f'val RMSLE: {val_error}')

    return train_error, val_error


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


def train_model(summary_df, df_train, df_test, results_train_df):
    '''
    Train a Ridge Regression model for predicting patient's consultation end time after triage assessment,
    and return:
        - trained model
        - train model's features
        - Root Mean Squared Log Error (RMSLE) for training, validation and testing
    '''

    scaler = StandardScaler()

    df_train_new = df_train.drop(['day', 'consultation_end_time', 'duration'], axis=1)
    df_test_new = df_test.drop(['day', 'consultation_end_time', 'duration'], axis=1)

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

    features = df_train_new.columns

    return meta, features


def get_model_results(meta, df_train, df_test):
    '''
    Returns train and test errors after training is concluded.
    '''
    X_train = df_train.drop(['consultation_end_time', 'duration'], axis=1)
    y_train_true = df_train['consultation_end_time'] - df_train['assessment_end_time']
    y_train_pred = meta.predict(X_train) - df_train['assessment_end_time']

    X_test = df_test.drop(['consultation_end_time', 'duration'], axis=1)
    y_test_true = df_test['consultation_end_time'] - df_test['assessment_end_time']
    y_test_pred = meta.predict(X_test) - df_test['assessment_end_time']

    train_error = np.sqrt(mean_squared_log_error(y_train_true, y_train_pred))
    test_error = np.sqrt(mean_squared_log_error(y_test_true, y_test_pred))

    print(f'train RMSLE: {train_error}')
    print(f'test RMSLE: {test_error}')

    return train_error, test_error


def get_overall_error(meta, summary_df, results_train_df):
    '''
    Returns the overall error over the entire dataset after training is concluded.
    '''
    scaler = StandardScaler()

    df = summary_df.drop(['day', 'consultation_end_time', 'duration'], axis=1)
    X = pd.concat([pd.DataFrame(scaler.fit_transform(df), columns=df.columns), summary_df['day']], axis=1)
    y = summary_df[['day', 'consultation_end_time']]

    meta = MetaEstimator(results_train_df['best_params'])
    meta.fit(X, y)

    X = summary_df.drop(['consultation_end_time', 'duration'], axis=1)
    y_true = summary_df['consultation_end_time'] - summary_df['assessment_end_time']
    y_pred = meta.predict(X) - summary_df['assessment_end_time']

    overall_error = np.sqrt(mean_squared_log_error(y_true, y_pred))
    print(f'overall root mean squared log error: {overall_error}')

    return overall_error


def save_trained_model_coefs(meta):
    joblib.dump(meta.coefs, 'trained_model_coefs.pkl')


def save_trained_model_features(features):
    joblib.dump(features, 'trained_model_features.pkl')


if __name__ == '__main__':
    main()
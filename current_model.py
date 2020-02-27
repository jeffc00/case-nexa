'''
A linear model that was trained using Ridge Regression.
'''
import joblib
import pandas as pd


def coefs():
    current_model_coefs = joblib.load('trained_model_coefs.pkl')
    intercept = current_model_coefs[0]
    coefs = current_model_coefs[1:]
    return intercept, coefs


def get_model_features(features):
    priority = features['priority']
    temperature = float(features['temperature'])
    pain = features['pain']

    features['pain_severe pain:temp_cat_fever'] = 0
    features['pain_severe pain:temp_cat_hypothermia'] = 0
    features['pain_moderate pain:temp_cat_normal'] = 0

    if priority == 'normal':
        features['priority_enc'] = 1
    else:
        features['priority_enc'] = 2

    if temperature <= 36.5:
        features['temp_cat_enc'] = 2
        if pain == 'severe pain':
            features['pain_severe pain:temp_cat_hypothermia'] = 1
        else:
            features['pain_severe pain:temp_cat_hypothermia'] = 0
    elif temperature <= 37.5:
        features['temp_cat_enc'] = 1
        if pain == 'no pain':
            features['pain_no pain:temp_cat_normal'] = 1
        else:
            features['pain_moderate pain:temp_cat_normal'] = 0
    else:
        features['temp_cat_enc'] = 2
        if pain == 'severe pain':
            features['pain_severe pain:temp_cat_fever'] = 1
        else:
            features['pain_severe pain:temp_cat_fever'] = 0

    if pain == 'no pain':
        features['pain_enc'] = 1
    elif pain == 'moderate pain':
        features['pain_enc'] = 2
    else:
        features['pain_enc'] = 3

    cols = ['assessment_end_time', 'priority_enc', 'pain_enc',
            'temp_cat_enc', 'pain_severe pain:temp_cat_hypothermia',
            'pain_moderate pain:temp_cat_normal', 'pain_severe pain:temp_cat_fever']

    return pd.Series(features).drop(['priority', 'temperature', 'pain']).reindex(cols)


def scale_features(X):
    stats = joblib.load('scaler_stats.pkl')
    return (X-stats['mean']) / stats['sd']


def predict(X_scaled):
    return coefs()[0] + X_scaled @ coefs()[1]
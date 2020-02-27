# This is where the interfaces to your model should be. Please edit the functions below and add whatever is needed
# for your model to work


def get_model():
    import current_ridge_model as model
    return model


def get_state_machine():
    return {'time': 0}


def get_features(state_machine, patient_id):
    return state_machine[patient_id]


def get_estimate(model, features):
    import pandas as pd

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

    patient_features = pd.Series(features).drop(['priority', 'temperature', 'pain']).reindex(cols)

    return model.predict(patient_features)


def update_state(state_machine, event):
    state_machine['time'] = event.time
    state_machine[event.patient] = {}
    if event.event == 'assessment concluded':
        state_machine[event.patient]['assessment_end_time'] = event.time
        assessment = dict(zip(['priority', 'temperature', 'pain'], event.assessment.split('|')))
        state_machine[event.patient].update(assessment)
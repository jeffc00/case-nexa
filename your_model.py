# This is where the interfaces to your model should be. Please edit the functions below and add whatever is needed
# for your model to work


def get_model():
    import current_model as model
    return model


def get_state_machine():
    return {'time': 0}


def get_features(state_machine, patient_id):
    return state_machine[patient_id]


def get_estimate(model, features):
    X = model.get_model_features(features)
    X_scaled = model.scale_features(X)
    return model.predict(X_scaled)


def update_state(state_machine, event):
    state_machine['time'] = event.time
    state_machine[event.patient] = {}
    if event.event == 'assessment concluded':
        state_machine[event.patient]['assessment_end_time'] = event.time
        assessment = dict(zip(['priority', 'temperature', 'pain'], event.assessment.split('|')))
        state_machine[event.patient].update(assessment)
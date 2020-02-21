# This is where the interfaces to your model should be. Please edit the functions below and add whatever is needed
# for your model to work


def get_model():
    return None


def get_state_machine():
    return {'time': 0}


def get_features(state_machine, patient_id):
    return state_machine['time']


def get_estimate(model, features):
    return features + 500


def update_state(state_machine, event):
    state_machine['time'] = event.time

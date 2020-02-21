import logging
import math

import pandas as pd

from your_model import get_estimate, get_features, get_model, get_state_machine, update_state
#  Place your model in a module name called 'your_model' in the same folder as this file


ESTIMATION_EVENT = 'assessment concluded'
FINISHING_EVENT = 'consultation_finished'


logger = logging.getLogger(__name__)


def evaluate_file(file_path):
    logger.debug('Processing file %s', file_path)
    day_events = pd.read_csv(file_path)
    model = get_model()
    state_machine = get_state_machine()
    estimation_estimated_times_per_patient = {}
    errors = []
    for _, event in day_events.iterrows():
        update_state(state_machine, event)
        if event.event == ESTIMATION_EVENT:
            estimation_estimated_times = get_estimation_estimated_times(model, state_machine, event)
            estimation_estimated_times_per_patient[event.patient] = estimation_estimated_times
        elif event.event == FINISHING_EVENT:
            error = get_logarithmic_error(estimation_estimated_times_per_patient[event.patient], event.time)
            errors.append(error)
    return errors


def get_estimation_estimated_times(model, state_machine, event):
    features = get_features(state_machine, event.patient)
    estimated_time = get_estimate(model, features)
    estimation_time = event.time
    assert (estimated_time > estimation_time)
    return estimation_time, estimated_time


def get_logarithmic_error(estimation_estimated_times, finishing_time):
    """
    >>> round(get_logarithmic_error((752, 1321), 795), 3)
    -2.583
    """
    estimation_time, estimated_time = estimation_estimated_times
    log_error = math.log(finishing_time - estimation_time) - math.log(estimated_time - estimation_time)
    return log_error

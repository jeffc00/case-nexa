import joblib


def coefs():
    current_model_coefs = joblib.load('trained_model_coefs.pkl')
    intercept = current_model_coefs[0]
    coefs = current_model_coefs[1:]
    return intercept, coefs


def predict(X):
    return coefs()[0] + X @ coefs()[1]
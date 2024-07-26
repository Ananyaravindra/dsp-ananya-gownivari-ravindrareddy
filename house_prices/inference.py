import joblib
import numpy as np
import pandas as pd
from .preprocess import preprocess_test_data


def make_predictions(data: pd.DataFrame) -> np.ndarray:
    scaler = joblib.load(
        'C:/Users/Ananya/dsp-ananya-gownivari-ravindrareddy/models/'
        'scaler.joblib'
    )
    onehot = joblib.load(
        'C:/Users/Ananya/dsp-ananya-gownivari-ravindrareddy/models/'
        'Encoder.joblib'
    )
    X_test_processed = preprocess_test_data(data, scaler, onehot)
    loaded_model = joblib.load(
        'C:/Users/Ananya/dsp-ananya-gownivari-ravindrareddy/models/'
        'model.joblib'
    )
    predicted_prices = loaded_model.predict(X_test_processed)
    return predicted_prices

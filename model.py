import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def train_model():
    data = pd.read_csv("stress_data.csv")

    X = data.drop("stress_level", axis=1)
    y = data["stress_level"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X, y_encoded)

    return model, le

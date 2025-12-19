import pandas as pd
from model import train_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


model, le = train_model()
data = pd.read_csv("stress_data.csv")
X = data.drop("stress_level", axis=1)
y = le.transform(data["stress_level"])

preds = model.predict(X)

print("\n--- Accuracy ---")
print(accuracy_score(y, preds))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y, preds))

print("\n--- Classification Report ---")
print(
    classification_report(
        y,
        preds,
        target_names=le.classes_
    )
)

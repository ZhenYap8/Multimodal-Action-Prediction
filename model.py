import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler


# ----------------------------
# 1. Load and clean data
# ----------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

label_map = {"tennis": 0, "orange": 1}

for df in [train, test]:
    df["target"] = df["target"].astype(str).str.strip().str.lower()
    df["target"] = df["target"].map(label_map)

train = train.dropna(subset=["target"]).reset_index(drop=True)
test = test.dropna(subset=["target"]).reset_index(drop=True)

train["target"] = train["target"].astype(int)
test["target"] = test["target"].astype(int)


# ----------------------------
# 2. Feature engineering
# ----------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["gaze_tennis_dist"] = np.sqrt(
        (df["gaze_x"] - df["tennis_x"]) ** 2 +
        (df["gaze_y"] - df["tennis_y"]) ** 2
    )
    df["gaze_orange_dist"] = np.sqrt(
        (df["gaze_x"] - df["orange_x"]) ** 2 +
        (df["gaze_y"] - df["orange_y"]) ** 2
    )
    df["hand_tennis_dist"] = np.sqrt(
        (df["hand_x"] - df["tennis_x"]) ** 2 +
        (df["hand_y"] - df["tennis_y"]) ** 2
    )
    df["hand_orange_dist"] = np.sqrt(
        (df["hand_x"] - df["orange_x"]) ** 2 +
        (df["hand_y"] - df["orange_y"]) ** 2
    )

    return df


train = add_features(train)
test = add_features(test)

feature_columns = [
    "gaze_tennis_dist",
    "gaze_orange_dist",
    "hand_tennis_dist",
    "hand_orange_dist",
]

X_train = train[feature_columns]
y_train = train["target"]

X_test = test[feature_columns]
y_test = test["target"]


# ----------------------------
# 3. Scale features
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ----------------------------
# 4. Train model
# ----------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

y_test_pred = model.predict(X_test_scaled)
y_train_pred = model.predict(X_train_scaled)


# ----------------------------
# 5. Evaluation summary
# ----------------------------
print("Train class counts:\n", y_train.value_counts())
print("\nTest class counts:\n", y_test.value_counts())
print(f"\nAccuracy: {accuracy_score(y_test, y_test_pred):.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_test_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Coefficients:\n", model.coef_)


# ----------------------------
# 6. Confusion matrix plot
# ----------------------------
cm = confusion_matrix(y_test, y_test_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Tennis", "Orange"]
)

disp.plot(
    cmap="Oranges",
    colorbar=True,
    values_format="d"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()
plt.show()

# ----------------------------
# 7. Learning curve
# ----------------------------
train_sizes, train_scores, val_scores = learning_curve(
    LogisticRegression(max_iter=200),
    X_train_scaled,
    y_train,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, linewidth=2, label="Training Accuracy")
plt.plot(train_sizes, val_mean, linewidth=2, label="Validation Accuracy")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Model Learning Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ----------------------------
# 8. Accuracy across action progression
# ----------------------------
train_results = train.copy()
train_results["y_true"] = y_train.values
train_results["y_pred"] = y_train_pred

frame_accuracy = (
    train_results.groupby("frame")
    .apply(lambda group: accuracy_score(group["y_true"], group["y_pred"]))
    .reset_index(name="accuracy")
)

print("\nAccuracy by frame:\n", frame_accuracy)

plt.figure(figsize=(8, 5))
plt.plot(frame_accuracy["frame"], frame_accuracy["accuracy"], marker="x", linewidth=2)
plt.xlabel("Frame")
plt.ylabel("Accuracy")
plt.title("Prediction Accuracy Across Action Progression")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
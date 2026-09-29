from sklearn.dummy import DummyClassifier
from src.data import load_dataset
from src.metrics import evaluate_multiclass

(X_tr, y_tr, _), (X_te, y_te, _), _ = load_dataset()

# Baseline 1: Dummy Classifier (Dự đoán theo lớp phổ biến nhất)
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_tr, y_tr)
y_pred_dummy = dummy.predict(X_te)
metrics_dummy = evaluate_multiclass(y_te, y_pred_dummy)

print(f"Dummy Baseline Macro-F1: {metrics_dummy['macro_f1']:.4f}")
print(f"Dummy Baseline Accuracy: {metrics_dummy['accuracy']:.4f}")
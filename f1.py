import pickle
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Load data
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# -------------------------------
# Step 1: Create Ground Truth
# -------------------------------
# Assume: movies with similarity > threshold are relevant

threshold = 0.5   # you can adjust

y_true = []
y_scores = []

# Compare first 100 movies (for demo)
for i in range(100):
    for j in range(100):
        if i != j:
            score = similarity[i][j]
            y_scores.append(score)

            # Ground truth (assumption)
            if score > 0.6:
                y_true.append(1)
            else:
                y_true.append(0)

# -------------------------------
# Step 2: Convert to Prediction
# -------------------------------
y_pred = [1 if score > threshold else 0 for score in y_scores]

# -------------------------------
# Step 3: Metrics
# -------------------------------
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

# -------------------------------
# Step 4: ROC
# -------------------------------
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# -------------------------------
# Step 5: Print Output in Terminal
# -------------------------------
print("\n===== RESULTS ANALYSIS =====")

print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1 Score  : {f1:.2f}")
print(f"AUC Score : {roc_auc:.2f}")

print("\nConfusion Matrix:")
print(cm)
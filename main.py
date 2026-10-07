from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


# ==========================================
# 1. DATASET
# ==========================================

X, y = make_classification(
    n_samples=2000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    weights=[0.95, 0.05],
    random_state=42
)


# ==========================================
# 2. CROSS-VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 3. BASELINE
# ==========================================

baseline = LogisticRegression(
    max_iter=1000
)


# ==========================================
# 4. CLASS WEIGHT
# ==========================================

class_weight = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)


# ==========================================
# 5. UNDERSAMPLING
# ==========================================

undersampling = Pipeline([
    ("sampler", RandomUnderSampler(random_state=42)),
    ("model", LogisticRegression(max_iter=1000))
])


# ==========================================
# 6. SMOTE
# ==========================================

smote = Pipeline([
    ("sampler", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000))
])


# ==========================================
# 7. МОДЕЛДЕР
# ==========================================

models = {
    "Baseline": baseline,
    "Class Weight": class_weight,
    "Undersampling": undersampling,
    "SMOTE": smote
}


# ==========================================
# 8. CROSS-VALIDATION
# ==========================================

print("\n==========================================")
print("       5-FOLD CROSS-VALIDATION")
print("==========================================")

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1"
    )

    print(f"\n{name}")

    print("F1 scores:", [round(score, 4) for score in scores])

    print("Mean F1:", round(scores.mean(), 4))

    print("Std:", round(scores.std(), 4))
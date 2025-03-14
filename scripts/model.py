from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import logging

logger = logging.getLogger(__name__)

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, stratify=y)

def train_model(X_train, y_train, model_type='rf'):
    if model_type == 'rf':
        model = RandomForestClassifier()
        param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10]}
        clf = GridSearchCV(model, param_grid, cv=3)
        clf.fit(X_train, y_train)
        return clf.best_estimator_
    else:
        raise ValueError("Model type not supported.")

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
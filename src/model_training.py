from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def train_rf(X_train, y_train):
    """
    Huấn luyện mô hình Random Forest với việc tối ưu hóa các tham số qua GridSearchCV.
    """
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [50, 100, 150], 
        'max_depth': [10, 20, None], 
        'min_samples_split': [2, 5, 10]
    }
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_

    print(f"Best Random Forest Params: {rf_grid.best_params_}")
    return rf_best

def train_ab(X_train, y_train):
    """
    Huấn luyện mô hình AdaBoost với việc tối ưu hóa các tham số qua GridSearchCV.
    """
    ab = AdaBoostClassifier(random_state=42)
    ab_params = {
        'n_estimators': [50, 100, 200], 
        'learning_rate': [0.01, 0.1, 1.0]
    }
    ab_grid = GridSearchCV(ab, ab_params, cv=5, scoring='accuracy', n_jobs=-1)
    ab_grid.fit(X_train, y_train)
    ab_best = ab_grid.best_estimator_

    print(f"Best AdaBoost Params: {ab_grid.best_params_}")
    return ab_best

def train_mlp(X_train, y_train):
    """
    Huấn luyện mô hình MLP với việc tối ưu hóa các tham số qua GridSearchCV.
    """
    mlp = MLPClassifier(max_iter=500, random_state=42)
    mlp_params = {
        'hidden_layer_sizes': [(50,), (100,), (200,)], 
        'activation': ['relu', 'tanh'], 
        'solver': ['adam', 'sgd']
    }
    mlp_grid = GridSearchCV(mlp, mlp_params, cv=5, scoring='accuracy', n_jobs=-1)
    mlp_grid.fit(X_train, y_train)
    mlp_best = mlp_grid.best_estimator_

    print(f"Best MLP Params: {mlp_grid.best_params_}")
    return mlp_best

def train_voting_classifier(rf_best, ab_best, mlp_best):
    """
    Tạo và huấn luyện Voting Classifier với các mô hình đã tối ưu hóa.
    """
    voting_clf = VotingClassifier(estimators=[
        ('rf', rf_best),
        ('ab', ab_best),
        ('mlp', mlp_best)
    ], voting='soft')

    voting_clf.fit(X_train, y_train)
    return voting_clf

def train_models(X_train, y_train):
    """
    Huấn luyện các mô hình cơ sở và Voting Classifier với GridSearchCV.
    """
    # Huấn luyện từng mô hình cơ sở với GridSearchCV
    print("Training Random Forest...")
    rf_best = train_rf(X_train, y_train)

    print("Training AdaBoost...")
    ab_best = train_ab(X_train, y_train)

    print("Training MLP...")
    mlp_best = train_mlp(X_train, y_train)

    # Tạo Voting Classifier từ các mô hình đã tối ưu hóa
    print("Training Voting Classifier...")
    voting_clf = train_voting_classifier(rf_best, ab_best, mlp_best)
    
    return voting_clf

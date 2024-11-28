from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_models
from src.model_evaluation import evaluate_model

def main():
    # Bước 1: Xử lý dữ liệu
    data = load_data('data/Churn_Modelling.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Bước 2: Huấn luyện mô hình
    voting_clf = train_models(X_train, y_train)

    # Bước 3: Đánh giá mô hình
    accuracy, f1, auc = evaluate_model(voting_clf, X_test, y_test)
    print(f'Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, AUC: {auc:.4f}')

if __name__ == "__main__":
    main()

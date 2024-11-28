import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Display the first few rows of the dataset to understand its structure
    data.head()
    label_encoder_gender = LabelEncoder()
    data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])


    # Khởi tạo OneHotEncoder với tham số drop='first' để loại bỏ một cột
    encoder = OneHotEncoder(sparse_output=False)

    # Mã hóa cột 'Geography'
    geography_encoded = encoder.fit_transform(data[['Geography']])


    # Tạo DataFrame từ kết quả mã hóa
    geography_df = pd.DataFrame(geography_encoded, columns=encoder.get_feature_names_out(['Geography']))

    # Ghép các cột đã mã hóa vào DataFrame gốc và loại bỏ cột gốc 'Geography'
    data = pd.concat([data.drop('Geography', axis=1), geography_df], axis=1)
    
    x_var = data.columns[data.columns != 'Exited']
    y_var = ["Exited"]
    
    X = data[x_var]
    y = data[y_var]
    

    # Chia dữ liệu thành 80% huấn luyện và 20% kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

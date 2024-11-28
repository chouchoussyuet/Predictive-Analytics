# Predictive-Analytics
Học phần Phân tích dữ liệu dự báo - UET VNU 2024

# Customer Churn Prediction with Ensemble Learning

Dự án này thực hiện dự đoán khả năng rời bỏ của khách hàng (churn prediction) sử dụng các mô hình học máy cơ sở như Random Forest, AdaBoost và MLP (Multilayer Perceptron). Các mô hình này được kết hợp trong một **Voting Classifier** để nâng cao độ chính xác dự đoán. Sau đó, nhãn dự đoán được cải thiện thông qua thuật toán **k-NN** (k-Nearest Neighbors) nhằm phát hiện và sửa nhãn sai.

Mục tiêu của dự án là cải thiện khả năng dự đoán churn của khách hàng bằng cách kết hợp các mô hình mạnh và sửa chữa các nhãn sai thông qua các điểm dữ liệu gần nhau trong không gian đặc trưng.

## Giới thiệu

Dự án này sử dụng bộ dữ liệu về khách hàng của một ngân hàng để dự đoán khả năng khách hàng sẽ rời bỏ ngân hàng. Các mô hình học máy cơ sở được huấn luyện và kết hợp lại với nhau để cải thiện độ chính xác của dự đoán. Quá trình dự đoán bao gồm hai phần chính:

1. **Ensemble Learning**: Các mô hình học máy cơ sở (Random Forest, AdaBoost và MLP) được huấn luyện và kết hợp lại thông qua **Voting Classifier**. Voting Classifier có thể sử dụng hai phương pháp: **hard voting** (voting theo nhãn đa số) và **soft voting** (voting theo xác suất). Chúng ta sẽ lựa chọn soft voting để có được dự đoán chính xác hơn.

2. **Sửa nhãn bằng k-NN**: Sau khi thực hiện dự đoán, thuật toán **k-NN** sẽ được sử dụng để sửa nhãn của các điểm dữ liệu. Thuật toán này dựa trên nguyên lý rằng các điểm có đặc trưng tương tự nhau có thể có cùng nhãn. Mỗi điểm dữ liệu sẽ được kiểm tra các điểm gần nhất, và nếu nhãn của điểm đó không giống với nhãn đa số của các điểm xung quanh, nhãn sẽ được sửa lại.

## Cài đặt và Yêu cầu

Để chạy dự án này, bạn cần cài đặt các thư viện cần thiết. Bạn có thể cài đặt các thư viện này thông qua `pip` bằng cách sử dụng file `requirements.txt`:

1. **Clone repository**:
- git clone [ https://github.com/yourusername/customer-churn-prediction.git](https://github.com/chouchoussyuet/Predictive-Analytics)
- cd customer-churn-prediction

2. **Cài đặt các thư viện yêu cầu**:
  pip install -r requirements.txt

Các thư viện cần thiết: 
+ pandas: Thư viện xử lý dữ liệu bảng, giúp đọc và xử lý bộ dữ liệu CSV.
+ numpy: Thư viện tính toán số học, hỗ trợ xử lý mảng và các phép toán vector.
+ scikit-learn: Thư viện học máy với các mô hình như Random Forest, AdaBoost, MLP, Voting Classifier, và k-NN. Cung cấp các công cụ cho việc tối ưu hóa mô hình và đánh giá hiệu suất.
+ matplotlib và seaborn: Dùng để trực quan hóa dữ liệu và kết quả.

## Chạy chương trình 
1. Xử lý dữ liệu (data_preprocessing.py):

Tải và tiền xử lý dữ liệu.
Chia dữ liệu thành các tập huấn luyện và kiểm tra.
Chuẩn hóa dữ liệu nếu cần thiết.

2. Huấn luyện mô hình (model_training.py):

Tối ưu hóa các tham số của các mô hình cơ sở (Random Forest, AdaBoost, MLP) sử dụng GridSearchCV.
Kết hợp các mô hình đã huấn luyện lại thành một VotingClassifier với voting kiểu "soft".

3. Phát hiện và sửa dự đoán sai bằng k-NN: 
Đánh giá mô hình (model_evaluation.py):

4. Đánh giá mô hình với các chỉ số như accuracy, F1-score và AUC.

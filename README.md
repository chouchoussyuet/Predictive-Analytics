# Predictive-Analytics
Học phần Phân tích dữ liệu dự báo - UET VNU 2024

# Customer Churn Prediction with Ensemble Learning

Dự án này thực hiện dự đoán khả năng rời bỏ của khách hàng (churn prediction) sử dụng các mô hình học máy cơ sở như Random Forest, AdaBoost và MLP (Multilayer Perceptron). Các mô hình này được kết hợp trong một Voting Classifier để nâng cao độ chính xác dự đoán. Sau đó, nhãn dự đoán được cải thiện thông qua thuật toán k-NN (k-Nearest Neighbors) nhằm phát hiện và sửa nhãn sai.

## Mục lục
1. [Giới thiệu](#giới-thiệu)
2. [Cài đặt và Yêu cầu](#cài-đặt-và-yêu-cầu)
3. [Cấu trúc Thư mục](#cấu-trúc-thư-mục)
4. [Cách sử dụng](#cách-sử-dụng)
5. [Các bước thực hiện](#các-bước-thực-hiện)
6. [Giải thích các phần](#giải-thích-các-phần)
7. [Liên hệ](#liên-hệ)

## Giới thiệu

Dự án này sử dụng bộ dữ liệu về khách hàng của một ngân hàng để dự đoán khả năng rời bỏ ngân hàng của họ. Các mô hình học máy cơ sở như Random Forest, AdaBoost và MLP được huấn luyện và kết hợp lại với nhau bằng Voting Classifier. Một thuật toán k-NN được áp dụng để sửa các nhãn sai trong quá trình dự đoán.

## Cài đặt và Yêu cầu

Để chạy dự án này, bạn cần cài đặt một số thư viện cần thiết. Sử dụng `pip` để cài đặt các yêu cầu:

```bash
pip install -r requirements.txt


# Nhận dạng chữ số viết tay MNIST

Repository này tổ chức lại đồ án theo hướng notebook là nơi trình bày thí nghiệm,
còn phần triển khai thuật toán được đặt trong package `src/pattern_recognition/`.
Các notebook gốc vẫn được giữ trong `plan/`; không xóa phần diễn giải, biểu đồ,
thử nghiệm hay đối chiếu với bài Lab.

## Cấu trúc

```text
.
├── plan/
│   ├── phase0_preprocessing_and_EDA.ipynb
│   ├── phase1.ipynb
│   ├── phase2.ipynb
│   └── phase3.ipynb
├── docs/
│   ├── DoAn_ThucHanh_NhanDang.pdf
│   └── Lab*.ipynb
├── src/pattern_recognition/
│   ├── config.py
│   ├── data.py
│   ├── models/
│   │   ├── from_scratch/
│   │   │   ├── naive_bayes.py
│   │   │   ├── logistic_regression.py
│   │   │   ├── mlp.py
│   │   │   └── random_forest.py
│   │   ├── pytorch/
│   │   │   └── cnn.py
│   │   └── sklearn/
│   │       └── ensemble_benchmarks.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── __init__.py
├── data/
│   └── processed/
├── artifacts/
├── figures/
├── results/
├── tests/
├── pyproject.toml
└── requirements.txt
```

`evaluation/` chứa phần đánh giá được dùng chung giữa các mô hình. Các biểu đồ
và trình bày kết quả gắn với một thí nghiệm cụ thể vẫn nằm trong notebook để
mạch báo cáo dễ theo dõi. `docs/` giữ đề bài và các bài Lab làm tài liệu đối chiếu.

## Ranh giới thuật toán và thư viện

- `models/from_scratch/naive_bayes.py`: Gaussian/Multinomial Naive Bayes tự cài
  bằng NumPy theo Phase 1. Các lớp `GaussianNB`/`MultinomialNB` của scikit-learn
  chỉ được dùng làm đối chiếu trong notebook.
- `models/from_scratch/logistic_regression.py`: Logistic Regression đa lớp,
  softmax, mini-batch gradient descent và L2 tự cài bằng NumPy theo Phase 1.
  `sklearn.linear_model.LogisticRegression` vẫn chỉ là mô hình tham chiếu.
- `models/from_scratch/mlp.py`: MLP Phase 2 tự tính forward/backprop, softmax
  cross-entropy, SGD momentum và early stopping. Mã dùng tensor/lớp tuyến tính
  PyTorch làm cấu trúc dữ liệu và phép toán; **không dùng autograd hoặc
  `torch.optim` cho MLP**. Đây là cách triển khai thủ công đang có trong notebook,
  không phải thay bằng MLP dựng sẵn.
- `models/pytorch/cnn.py`: CNN được khai báo bằng `torch.nn`; quá trình huấn luyện
  CNN giữ cách dùng PyTorch hiện tại. CNN không bị gộp vào MLP.
- `models/from_scratch/random_forest.py`: Decision Tree CART và Random Forest
  thuần NumPy tự cài theo Phase 3.
- `models/sklearn/ensemble_benchmarks.py`: hàm benchmark dùng trong thí nghiệm
  thư viện của Phase 3. Bagging, Random Forest, Extra Trees và
  HistGradientBoosting của scikit-learn là benchmark riêng, không phải phần
  Random Forest thuần NumPy.

Việc chuyển code sang `src/` không nhằm đổi thuật toán. Các notebook import lại
các lớp/hàm đã chuyển để tránh hai bản cài đặt bị lệch nhau. Những thay đổi cấu
trúc cần thiết (chẳng hạn truyền rõ dữ liệu train/validation vào hàm huấn luyện
MLP thay vì đọc biến toàn cục của notebook) không thay đổi công thức hay quy
trình huấn luyện.

## Cài đặt và chạy

Từ thư mục gốc repository, tạo/kích hoạt môi trường Python rồi chạy:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Mở Jupyter và chạy lần lượt:

1. `plan/phase0_preprocessing_and_EDA.ipynb`
2. `plan/phase1.ipynb`
3. `plan/phase2.ipynb`
4. `plan/phase3.ipynb`

Các notebook thêm thư mục `src/` vào đường dẫn import để vẫn chạy được trong
repository mà không cần cài package editable; `pip install -e . --no-deps` là
tùy chọn thuận tiện khi dùng package ngoài notebook. Phase 1–3 cần file
`data/processed/mnist_processed.npz` do Phase 0 tạo ra. Các file lớn trong
`data/MNIST/` được Git bỏ qua theo `.gitignore`; tải/xử lý dữ liệu lại bằng
notebook Phase 0 khi clone repository mới.

Notebook Phase 2 dùng PyTorch cho các phép toán tensor. MLP tự gán gradient và
cập nhật tham số, trong khi CNN sử dụng cơ chế huấn luyện PyTorch của notebook.
Không thay hai cách huấn luyện này cho nhau.

## Kiểm tra nhanh

```powershell
python -m unittest discover -s tests -v
```

Các bài kiểm tra tập trung vào smoke-test trên dữ liệu nhỏ để phát hiện lỗi
import/API sau khi tách module; không chạy toàn bộ thí nghiệm MNIST tốn thời gian.

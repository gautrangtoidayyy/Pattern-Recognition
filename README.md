
# Nhận dạng chữ số viết tay MNIST — Đồ án Nhận dạng mẫu

> Đồ án môn học **Nhận dạng mẫu (Pattern Recognition)** — xây dựng, huấn luyện và
> so sánh nhiều mô hình phân loại trên bộ dữ liệu MNIST (70 000 ảnh chữ số viết
> tay 28x28 pixel, 10 lớp từ 0 đến 9).

---

## Mục tiêu dự án

1. **Tự cài đặt (from scratch)** các thuật toán cốt lõi bằng NumPy/PyTorch mà
   **không dùng các hàm huấn luyện có sẵn** của thư viện, nhằm hiểu sâu nguyên lý
   toán học phía sau mỗi mô hình.
2. **So sánh đối chiếu** kết quả tự cài với các mô hình tương ứng của scikit-learn
   và PyTorch để kiểm chứng độ chính xác.
3. **Trình bày thí nghiệm** theo dạng notebook Jupyter có biểu đồ trực quan,
   metrics chi tiết (Accuracy, Precision, Recall, Macro-F1) và phân tích kết quả.

---

## Cấu trúc thư mục

```text
Pattern-Recognition/
│
├── notebook/                             # Notebook thí nghiệm chính
│   ├── preprocessing.ipynb               #   Khám phá, tiền xử lý và chia dữ liệu
│   ├── 1_Bayes.ipynb                     #   Thí nghiệm Naive Bayes
│   ├── 2_Logistic.ipynb                  #   Thí nghiệm Logistic Regression
│   ├── 3_MLP_CNN.ipynb                   #   Thí nghiệm MLP và CNN
│   └── 4_Ensemble_Comparisons.ipynb      #   Thí nghiệm Ensemble & so sánh tổng thể
│
├── docs/                                 # Tài liệu đề bài và bài Lab tham khảo
│   ├── DoAn_ThucHanh_NhanDang.pdf        #   Đề bài đồ án thực hành
│   ├── CNN_MNIST_from_scratch_explained.ipynb  # Tài liệu giải thích CNN from scratch
│   ├── Lab1.ipynb                        #   Bài Lab 1 (tham khảo)
│   ├── Lab2.ipynb                        #   Bài Lab 2 — Naive Bayes
│   ├── Lab3.ipynb                        #   Bài Lab 3 — Logistic Regression
│   └── Lab4.ipynb                        #   Bài Lab 4 — Neural Networks
│
├── src/pattern_recognition/              # Package Python chứa triển khai thuật toán
│   ├── __init__.py                       #   Đánh dấu package gốc
│   ├── config.py                         #   Hằng số cấu hình (đường dẫn, seed)
│   ├── data.py                           #   Hàm tải dữ liệu đã xử lý
│   │
│   ├── models/                           #   Thư mục chứa tất cả mô hình
│   │   ├── __init__.py
│   │   │
│   │   ├── from_scratch/                 #   Mô hình tự cài bằng NumPy/PyTorch
│   │   │   ├── __init__.py
│   │   │   ├── naive_bayes.py            #     Gaussian & Multinomial Naive Bayes
│   │   │   ├── logistic_regression.py    #     Softmax Logistic Regression đa lớp
│   │   │   ├── mlp.py                    #     Multi-Layer Perceptron (forward/backprop thủ công)
│   │   │   └── random_forest.py          #     Decision Tree CART & Random Forest
│   │   │
│   │   ├── pytorch/                      #   Mô hình dùng PyTorch đầy đủ
│   │   │   ├── __init__.py
│   │   │   └── cnn.py                    #     Convolutional Neural Network (CNN)
│   │   │
│   │   └── sklearn/                      #   Mô hình benchmark dùng scikit-learn
│   │       ├── __init__.py
│   │       └── ensemble_benchmarks.py    #     Hàm benchmark Bagging, RF, ExtraTrees, HGBC
│   │
│   └── evaluation/                       #   Module đánh giá dùng chung
│       ├── __init__.py
│       └── metrics.py                    #     Hàm evaluate cho MLP và CNN
│
├── data/                                 # Dữ liệu
│   └── processed/                        #   Chứa mnist_processed.npz (sinh bởi preprocessing)
│   (data/MNIST/ bị .gitignore — chứa raw MNIST khi tải về)
│
├── artifacts/                            # Artifact mô hình đã huấn luyện
│   ├── split_indices.npz                 #   Chỉ số chia train/val/test (tái lập kết quả)
│   ├── phase1_gaussian_nb.npz            #   Tham số Gaussian NB đã fit
│   └── phase1_multinomial_nb.npz         #   Tham số Multinomial NB đã fit
│
├── figures/                              # Biểu đồ kết quả (sinh bởi notebook)
├── results/                              # File CSV metrics (sinh bởi notebook)
│   (Dự kiến: 1_Bayes_metrics.csv, 2_Logistic_metrics.csv, ...)
│
├── tests/                                # Kiểm thử tự động
│   └── test_package_smoke.py             #   Smoke test import/API trên dữ liệu nhỏ
│
├── plan/                                 # Thư mục kế hoạch phát triển
├── pyproject.toml                        # Cấu hình build (setuptools, metadata)
├── requirements.txt                      # Danh sách thư viện phụ thuộc
└── .gitignore                            # Loại trừ .venv, __pycache__, data/MNIST, ...
```

---

## Mô tả chi tiết các Notebook

### `notebook/preprocessing.ipynb` — Tiền xử lý dữ liệu

- Tải bộ dữ liệu MNIST gốc (70 000 ảnh 28x28 pixel) qua `torchvision`.
- Khám phá dữ liệu (EDA): phân bố nhãn, hiển thị ảnh mẫu, kiểm tra giá trị pixel.
- Chuẩn hóa pixel về khoảng [0, 1] bằng cách chia cho 255.
- Chia dữ liệu thành 3 tập: **Train / Validation / Test** với seed cố định.
- Lưu kết quả vào `data/processed/mnist_processed.npz` để các notebook sau dùng chung.
- Lưu chỉ số chia vào `artifacts/split_indices.npz` để đảm bảo tái lập.

### `notebook/1_Bayes.ipynb` — Naive Bayes (Phase 1)

- Huấn luyện và đánh giá **Gaussian Naive Bayes** tự cài (from scratch).
- Huấn luyện và đánh giá **Multinomial Naive Bayes** tự cài (from scratch).
- So sánh đối chiếu với `GaussianNB` và `MultinomialNB` của scikit-learn.
- Hiển thị confusion matrix, classification report, biểu đồ phân tích lỗi.
- Xuất metrics vào `results/1_Bayes_metrics.csv`.

### `notebook/2_Logistic.ipynb` — Logistic Regression (Phase 1)

- Huấn luyện **Softmax Logistic Regression đa lớp** tự cài bằng NumPy.
- Sử dụng Mini-batch Gradient Descent với điều chuẩn L2 (không phạt bias).
- Theo dõi loss history qua các epoch; trực quan hóa đường cong hội tụ.
- So sánh đối chiếu với `sklearn.linear_model.LogisticRegression`.
- Xuất metrics vào `results/2_Logistic_metrics.csv`.

### `notebook/3_MLP_CNN.ipynb` — MLP & CNN (Phase 2)

- **MLP tự cài (from scratch)**:
  - Lan truyền thuận (forward) và lan truyền ngược (backprop) hoàn toàn thủ công.
  - Hàm mất mát Softmax Cross-Entropy, bộ tối ưu SGD Momentum, Inverted Dropout.
  - Hỗ trợ Early Stopping dựa trên validation loss.
  - Dùng `torch.Tensor` và `nn.Linear` như **cấu trúc dữ liệu**; **không dùng**
    `torch.autograd` hay `torch.optim` — gradient được tính hoàn toàn bằng tay.
  - Phân tích ảnh hưởng của kiến trúc, hàm kích hoạt (ReLU, Sigmoid, Tanh),
    cách khởi tạo trọng số (He, Xavier, Constant), Dropout rate.
- **CNN dùng PyTorch**:
  - Kiến trúc: Conv2D (32 bộ lọc 4x4) -> ReLU -> MaxPool2D -> Flatten -> Dense(128)
    -> Dense(10).
  - Huấn luyện sử dụng cơ chế PyTorch tiêu chuẩn (`autograd`, `torch.optim`).
  - So sánh hiệu năng CNN với MLP from scratch.
- Xuất metrics vào `results/3_MLP_CNN_metrics.csv`.

### `notebook/4_Ensemble_Comparisons.ipynb` — Ensemble & So sánh tổng thể (Phase 3)

- Huấn luyện **Random Forest tự cài** (Decision Tree CART + Bagging, thuần NumPy).
- Benchmark các mô hình Ensemble của scikit-learn:
  Bagging, Random Forest, Extra Trees, HistGradientBoosting.
- Tổng hợp kết quả từ **tất cả** các notebook trước (đọc 4 file CSV metrics).
- Tạo `results/overall_comparison_metrics.csv` và các biểu đồ so sánh tổng thể.
- **Cần chạy sau notebook 1–3** để có đầy đủ dữ liệu tổng hợp.

---

## Mô tả chi tiết Package `src/pattern_recognition/`

### `config.py` — Hằng số cấu hình

| Hằng số        | Giá trị                                   | Mô tả                                                |
| ---------------- | ------------------------------------------- | ------------------------------------------------------ |
| `PROJECT_ROOT` | Thư mục gốc repository (tự phát hiện) | Đường dẫn tuyệt đối đến gốc dự án          |
| `DATA_PATH`    | `data/processed/mnist_processed.npz`      | Đường dẫn mặc định đến dữ liệu đã xử lý |
| `SEED`         | `42`                                      | Seed cố định đảm bảo tái lập kết quả         |

### `data.py` — Tải dữ liệu

- **`load_processed_mnist(path)`**: Tải file `.npz` chứa các mảng NumPy
  (X_train, y_train, X_val, y_val, X_test, y_test). Báo lỗi rõ ràng nếu chưa
  chạy `preprocessing.ipynb`.

### `models/from_scratch/` — Thuật toán tự cài

#### `naive_bayes.py`

| Lớp / Hàm                      | Mô tả                                                         |
| -------------------------------- | --------------------------------------------------------------- |
| `NaiveBayes`                   | Lớp tổng quát Naive Bayes; nhận hàm `log_pdf` tùy biến |
| `MultinomialNaiveBayes`        | Multinomial NB cho đặc trưng không âm (tần suất pixel)   |
| `gaussian_log_pdf(X, mu, var)` | Hàm log-mật độ phân phối chuẩn Gaussian                  |
| `multinomial_log_pdf(...)`     | Hàm log-xác suất với làm mịn Laplace (alpha smoothing)    |

- Hoàn toàn bằng NumPy. Tính log-posterior = log P(C) + sum log P(x_i | C) rồi
  chọn lớp có điểm cao nhất.

#### `logistic_regression.py`

| Lớp / Hàm              | Mô tả                                                 |
| ------------------------ | ------------------------------------------------------- |
| `LogisticRegressionNP` | Softmax Logistic Regression đa lớp, mini-batch GD, L2 |
| `add_bias(X)`          | Thêm cột 1 vào ma trận (bias trick)                 |
| `sigmoid(z)`           | Hàm sigmoid ổn định số học                        |
| `softmax(z)`           | Softmax đa lớp, trừ max chống tràn số             |

- Hoàn toàn bằng NumPy. One-hot encoding, gradient Cross-Entropy + L2
  regularization (không phạt bias), ghi loss history theo epoch.

#### `mlp.py`

| Lớp / Hàm                                                | Mô tả                                                               |
| ---------------------------------------------------------- | --------------------------------------------------------------------- |
| `build_mlp(sizes, ...)`                                  | Xây MLP theo danh sách kích thước (He/Xavier/Const init)         |
| `forward_with_cache(model, X)`                           | Lan truyền thuận, lưu cache (a_in, a_out) cho backprop             |
| `softmax_cross_entropy(logits, y)`                       | Tính loss và gradient dJ/dlogits                                    |
| `linear_backward(layer, a_in, d)`                        | Gradient cho lớp Linear: dW, db, grad_in                             |
| `relu_backward`, `sigmoid_backward`, `tanh_backward` | Gradient hàm kích hoạt                                             |
| `dropout_backward(mask, p, grad)`                        | Gradient Inverted Dropout                                             |
| `backward(cache, grad_out, wd)`                          | Lan truyền ngược toàn mạng, gán `.grad` cho mọi tham số     |
| `SGDMomentum`                                            | Bộ tối ưu SGD có momentum, tự cài                               |
| `train_one_epoch(...)`                                   | Huấn luyện 1 epoch theo mini-batch                                  |
| `evaluate(model, X, y)`                                  | Tính cross-entropy trung bình và accuracy (Dropout tắt)           |
| `train_model(...)`                                       | Vòng lặp huấn luyện nhiều epoch, ghi history                     |
| `train_with_early_stopping(...)`                         | Huấn luyện có Early Stopping theo val_loss, khôi phục best state |
| `layer_grad_norms(model, X, y)`                          | Đo norm dW từng lớp (phân tích vanishing/exploding gradient)     |
| `relu_output_stds(model, X)`                             | Đo std đầu ra ReLU (phân tích khởi tạo trọng số)             |

- Dùng `torch.Tensor` và `nn.Linear` làm cấu trúc dữ liệu; **tự tính gradient**
  bằng công thức BP1–BP4, **không dùng** `autograd` hay `torch.optim`.

#### `random_forest.py`

| Lớp                        | Mô tả                                                                 |
| --------------------------- | ----------------------------------------------------------------------- |
| `DecisionNode`            | Nút cây quyết định (feature_idx, threshold, left, right, value)    |
| `DecisionTreeFromScratch` | Cây CART phân loại: Gini impurity, Random Subspace, giới hạn depth |
| `RandomForestFromScratch` | Rừng ngẫu nhiên: Bootstrap Aggregating + Random Feature Selection    |

- Hoàn toàn bằng NumPy. Dùng phân vị (percentile) để chọn ngưỡng chia nhanh.
  Bầu chọn đa số (majority voting) khi dự đoán.

### `models/pytorch/cnn.py` — CNN dùng PyTorch

| Lớp / Hàm          | Mô tả                                                            |
| -------------------- | ------------------------------------------------------------------ |
| `MNISTCNN`         | CNN: Conv2D(32, 4x4) -> ReLU -> MaxPool(2x2) -> FC(128) -> FC(10)  |
| `predict_cnn(...)` | Dự đoán batch kèm xác suất softmax; đo thời gian inference |

- Sử dụng `torch.nn` và `autograd` đầy đủ. Kiến trúc theo tài liệu
  `CNN_MNIST_from_scratch_explained.ipynb`.

### `models/sklearn/ensemble_benchmarks.py` — Benchmark scikit-learn

| Hàm                       | Mô tả                                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------------------- |
| `benchmark(models, ...)` | Huấn luyện, đo Accuracy/F1/thời gian cho dict các mô hình sklearn; trả về DataFrame xếp hạng |

- Hỗ trợ: `BaggingClassifier`, `RandomForestClassifier`,
  `ExtraTreesClassifier`, `HistGradientBoostingClassifier`.
- Đây là **benchmark đối chiếu**, không phải thuật toán tự cài.

### `evaluation/metrics.py` — Hàm đánh giá dùng chung

| Hàm                       | Mô tả                                                      |
| -------------------------- | ------------------------------------------------------------ |
| `evaluate_mlp_full(...)` | Đánh giá MLP: Accuracy, Precision, Recall, F1, thời gian |
| `evaluate_cnn_full(...)` | Đánh giá CNN: Accuracy, Precision, Recall, F1, thời gian |

---

## Thư mục `docs/` — Tài liệu tham khảo

| Tệp                                       | Nội dung                                                                  |
| ------------------------------------------ | -------------------------------------------------------------------------- |
| `DoAn_ThucHanh_NhanDang.pdf`             | Đề bài đồ án thực hành môn Nhận dạng mẫu                       |
| `Lab1.ipynb`                             | Bài Lab 1 — Giới thiệu, làm quen dữ liệu                            |
| `Lab2.ipynb`                             | Bài Lab 2 — Naive Bayes (cơ sở cho `naive_bayes.py`)                 |
| `Lab3.ipynb`                             | Bài Lab 3 — Logistic Regression (cơ sở cho `logistic_regression.py`) |
| `Lab4.ipynb`                             | Bài Lab 4 — Neural Networks                                              |
| `CNN_MNIST_from_scratch_explained.ipynb` | Giải thích chi tiết kiến trúc CNN from scratch trên MNIST            |

Các bài Lab là **tài liệu đối chiếu** — phần triển khai sản phẩm nằm trong
`src/` và được trình bày trong `notebook/`.

---

## Thư mục `artifacts/` — Artifact mô hình

| Tệp                          | Nội dung                                                             |
| ----------------------------- | --------------------------------------------------------------------- |
| `split_indices.npz`         | Chỉ số chia Train/Validation/Test (đảm bảo tái lập)            |
| `phase1_gaussian_nb.npz`    | Tham số (mean, var, prior) của Gaussian NB đã huấn luyện        |
| `phase1_multinomial_nb.npz` | Tham số (feature_count, prior) của Multinomial NB đã huấn luyện |

---

## Thư mục `figures/` và `results/`

- **`figures/`**: Chứa các biểu đồ (confusion matrix, learning curve, so sánh
  mô hình, ...) được sinh bởi các notebook. Thư mục trống khi clone mới — chạy
  notebook để tạo lại.
- **`results/`**: Chứa các file CSV metrics từ mỗi phase thí nghiệm. Dự kiến:
  - `1_Bayes_metrics.csv` — Kết quả Naive Bayes
  - `2_Logistic_metrics.csv` — Kết quả Logistic Regression
  - `3_MLP_CNN_metrics.csv` — Kết quả MLP & CNN
  - `4_Ensemble_metrics.csv` — Kết quả Ensemble
  - `overall_comparison_metrics.csv` — Bảng so sánh tổng thể tất cả mô hình

---

## Thư mục `tests/` — Kiểm thử

### `test_package_smoke.py`

Bộ kiểm thử nhẹ (smoke test) trên dữ liệu giả nhỏ, gồm:

- **`test_naive_bayes_models_fit_and_predict`**: Kiểm tra Gaussian NB và
  Multinomial NB tự cài có fit/predict đúng trên 4 mẫu.
- **`test_logistic_regression_fit_and_predict`**: Kiểm tra Logistic Regression
  tự cài hội tụ đúng trên bài toán 2 lớp đơn giản.
- **`test_random_forest_fit_and_predict`**: Kiểm tra Random Forest tự cài
  fit/predict trả về shape và nhãn hợp lệ.
- **`test_mlp_manual_training_api_and_cnn_output_shape`**: Kiểm tra API MLP
  (train 2 epoch) và CNN (output shape 10 lớp).
- **`test_missing_processed_data_has_actionable_error`**: Kiểm tra thông báo lỗi
  rõ ràng khi chưa chạy preprocessing.

Mục đích: phát hiện lỗi import / API sau khi tách module; **không chạy** toàn
bộ thí nghiệm MNIST tốn thời gian.

---

## Ranh giới thuật toán — Tự cài vs. Thư viện

| Mô hình                     | Cách triển khai                                             | File                                           |
| ----------------------------- | ------------------------------------------------------------- | ---------------------------------------------- |
| Gaussian Naive Bayes          | **Tự cài** (NumPy)                                    | `models/from_scratch/naive_bayes.py`         |
| Multinomial Naive Bayes       | **Tự cài** (NumPy)                                    | `models/from_scratch/naive_bayes.py`         |
| Softmax Logistic Regression   | **Tự cài** (NumPy)                                    | `models/from_scratch/logistic_regression.py` |
| Multi-Layer Perceptron (MLP)  | **Tự cài** backprop (PyTorch tensor, không autograd) | `models/from_scratch/mlp.py`                 |
| Decision Tree CART            | **Tự cài** (NumPy)                                    | `models/from_scratch/random_forest.py`       |
| Random Forest                 | **Tự cài** (NumPy)                                    | `models/from_scratch/random_forest.py`       |
| CNN                           | **Thư viện** PyTorch (`nn.Module`, autograd, optim) | `models/pytorch/cnn.py`                      |
| Bagging, RF, ExtraTrees, HGBC | **Thư viện** scikit-learn (benchmark đối chiếu)    | `models/sklearn/ensemble_benchmarks.py`      |

---

## Cài đặt và chạy

### 1. Cài đặt môi trường

```powershell
# Tạo và kích hoạt virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Cài thư viện phụ thuộc
python -m pip install -r requirements.txt

# (Tùy chọn) Cài package dạng editable để import từ bất kỳ đâu
python -m pip install -e . --no-deps
```

### 2. Thư viện phụ thuộc

| Thư viện       | Vai trò                                                    |
| ---------------- | ----------------------------------------------------------- |
| `numpy`        | Tính toán ma trận, triển khai thuật toán from scratch |
| `pandas`       | Xử lý bảng dữ liệu, xuất CSV                          |
| `matplotlib`   | Vẽ biểu đồ, trực quan hóa                             |
| `seaborn`      | Biểu đồ thống kê nâng cao                             |
| `scipy`        | Hàm toán học bổ trợ                                    |
| `scikit-learn` | Mô hình benchmark, metrics đánh giá                    |
| `torch`        | Tensor cho MLP, kiến trúc/huấn luyện CNN                |
| `torchvision`  | Tải bộ dữ liệu MNIST gốc                               |
| `joblib`       | Lưu/tải mô hình                                         |
| `jupyter`      | Chạy notebook                                              |

### 3. Chạy notebook theo thứ tự

```
1. notebook/preprocessing.ipynb    <- chạy đầu tiên (tạo dữ liệu cho các bước sau)
2. notebook/1_Bayes.ipynb          <- Naive Bayes (Phase 1)
3. notebook/2_Logistic.ipynb       <- Logistic Regression (Phase 1)
4. notebook/3_MLP_CNN.ipynb        <- MLP & CNN (Phase 2)
5. notebook/4_Ensemble_Comparisons.ipynb  <- Ensemble & tổng hợp (Phase 3)
```

**Lưu ý quan trọng**: Các notebook cần chạy **tuần tự** vì:

- `preprocessing.ipynb` tạo file `data/processed/mnist_processed.npz` mà mọi
  notebook khác đều phụ thuộc.
- Notebook 4 đọc CSV metrics từ notebook 1–3 để tạo bảng so sánh tổng thể.

**Khi clone mới**: Thư mục `data/MNIST/` (raw data) bị `.gitignore` bỏ qua.
Chạy `preprocessing.ipynb` để tải và xử lý lại dữ liệu.

### 4. Import trong notebook

Các notebook tự thêm `src/` vào `sys.path` nên chạy được mà không cần cài
package editable. Lệnh `pip install -e . --no-deps` là tùy chọn tiện lợi khi
muốn import package từ script bên ngoài notebook.

---

## Chạy kiểm thử

```powershell
python -m unittest discover -s tests -v
```

---

## Tệp cấu hình gốc

| Tệp                 | Mô tả                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------- |
| `pyproject.toml`   | Metadata dự án (tên, phiên bản, Python >= 3.10, setuptools)                              |
| `requirements.txt` | Danh sách 10 thư viện phụ thuộc                                                          |
| `.gitignore`       | Loại trừ: `.venv/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `data/MNIST/` |

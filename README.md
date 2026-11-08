# Nhận dạng chữ số viết tay MNIST — Đồ án Nhận dạng mẫu

> **Đồ án môn học Nhận dạng mẫu (Pattern Recognition)**
> Trường Đại học Khoa học Tự nhiên — ĐHQG TP.HCM
> Đề tài: Xây dựng, huấn luyện, tối ưu và đánh giá toàn diện các mô hình phân lớp trên bộ dữ liệu ảnh chữ số viết tay **MNIST** (70.000 ảnh kích thước $28 \times 28$ pixel, 10 chữ số từ 0 đến 9).

---

## Bảng tổng hợp kết quả thực nghiệm đối chuẩn

Toàn bộ các mô hình được huấn luyện trên cùng tập **Train ($50.000$ mẫu)**, tinh chỉnh siêu tham số trên tập **Validation ($10.000$ mẫu)** và đánh giá độc lập trên tập **Test ($10.000$ mẫu)** với seed ngẫu nhiên cố định `42`:

| Mô hình                          | Trường phái / Triển khai                                      | Test Accuracy (%) | Test Macro-F1 (%) | Thời gian Fit (s) | Thời gian Infer / mẫu |
| :--------------------------------- | :---------------------------------------------------------------- | :---------------: | :---------------: | :----------------: | :---------------------: |
| **CNN**                      | Mạng tích chập PyTorch (`Conv2D-MaxPool-FC`)                 | **98.76%** | **98.76%** |      51.61 s      |        ~0.043 ms        |
| **MLP**                      | Mạng nơ-ron đa tầng (**From scratch**, Manual Backprop) | **98.26%** | **98.25%** |      33.57 s      |   **~0.002 ms**   |
| **AdaBoost**                 | Học quần thể (`DecisionTreeClassifier`, depth=3)             | **91.75%** | **91.70%** |     11019.24 s     |        ~0.267 ms        |
| **Logistic Regression**      | Tuyến tính One-vs-Rest (**From scratch**, Batch GD)       | **86.84%** | **86.55%** |      10.70 s      |   **~0.003 ms**   |
| **Multinomial Naïve Bayes** | Sinh xác suất MAP + Laplace (**From scratch**, NumPy)     | **83.56%** | **83.41%** |  **0.05 s**  |        ~0.027 ms        |

> [!TIP]
> **Nhận xét then chốt:**
>
> - **Tốc độ nhanh nhất:** Multinomial Naïve Bayes hoàn tất huấn luyện $50.000$ mẫu chỉ trong $0.05$ giây nhờ thuật toán thống kê giải tích đếm tần suất pixel mà không cần vòng lặp gradient lặp lại.
> - **Cân bằng tuyến tính:** Logistic Regression đạt $86.84\%$ với tốc độ hội tụ ổn định qua 30 epoch ở tốc độ học $\eta^* = 0.5$.
> - **Độ chính xác vượt trội:** Các mô hình học sâu (MLP đạt $98.26\%$ và CNN đạt $98.76\%$) khai thác xuất sắc mối tương quan không gian phi tuyến của ảnh chữ số viết tay.

---

## Mục tiêu dự án

1. **Tự cài đặt (From Scratch)**: Hiện thực hóa thuật toán cốt lõi bằng NumPy/PyTorch mà **không sử dụng các hàm huấn luyện có sẵn** của thư viện:
   - Multinomial Naïve Bayes (tính kỳ vọng pixel, làm mượt Laplace, suy luận MAP miền Log).
   - One-vs-Rest Logistic Regression (chèn bias trick, mặt nạ không phạt bias, vector hóa Gradient Descent song song 10 lớp).
   - Multi-Layer Perceptron (lan truyền thuận, lan truyền ngược thủ công tính đạo hàm BP1–BP4, Inverted Dropout, SGD Momentum).
   - AdaBoost Ensemble (phân loại mẫu yếu Decision Stumps, cập nhật phân phối trọng số mẫu $D_t$ và trọng số biểu quyết $\alpha_t$).
2. **Kiểm chứng và đối chiếu chuẩn mực**: So sánh kết quả tự cài đặt với các mô hình chuẩn tương ứng từ `scikit-learn` và `torch.nn` để bảo đảm tính chuẩn xác toán học.
3. **Phân tích thực nghiệm đa tiêu chí**: Trực quan hóa động học hội tụ, ma trận nhầm lẫn (Confusion Matrix), phân tích các ca nhận dạng lỗi sai điển hình, so sánh Pareto Đánh đổi Độ chính xác — Thời gian và biểu đồ Radar đa chiều.

---

## Cấu trúc thư mục dự án

```text
Pattern-Recognition/
│
├── notebook/                             # Chuỗi Jupyter Notebook thực nghiệm chính
│   ├── 0_Preprocessing_EDA.ipynb         #   Khám phá dữ liệu (EDA), phân bố lớp, chuẩn hóa, phân tầng dữ liệu
│   ├── 1_Bayes.ipynb                     #   Mô hình 1: Multinomial Naïve Bayes from scratch & Laplace smoothing
│   ├── 2_Logistic.ipynb                  #   Mô hình 2: One-vs-Rest Logistic Regression from scratch & Vectorized GD
│   ├── 3_MLP_CNN.ipynb                   #   Mô hình 3: MLP from scratch (Backprop thủ công) & CNN PyTorch
│   ├── 4_Ensemble.ipynb                  #   Mô hình 4: AdaBoost Ensemble & Cây quyết định from scratch
│   └── Compairisons.ipynb                #   Đánh giá đối chuẩn tổng hợp, Trade-offs & Biểu đồ Radar
│
├── src/pattern_recognition/              # Package Python tái sử dụng thuật toán
│   ├── __init__.py                       #   Package initialization
│   ├── config.py                         #   Hằng số cấu hình hệ thống (SEED=42, đường dẫn dữ liệu)
│   ├── data.py                           #   Hàm tải và giải nén dữ liệu mnist_processed.npz
│   │
│   ├── models/                           #   Module triển khai thuật toán
│   │   ├── from_scratch/                 #     Mô hình tự cài đặt (NumPy / Manual Tensor)
│   │   │   ├── naive_bayes.py            #       Multinomial Naïve Bayes
│   │   │   ├── logistic_regression.py    #       One-vs-Rest Logistic Regression (Batch GD)
│   │   │   ├── mlp.py                    #       MLP lan truyền ngược thủ công (BP1-BP4)
│   │   │   └── ensemble.py               #       AdaBoost Ensemble & Decision Stumps from scratch
│   │   ├── pytorch/                      #     Mô hình PyTorch chính thức
│   │   │   └── cnn.py                    #       CNN Conv2D-MaxPool2D-FC
│   │   └── sklearn/                      #     Bộ mô hình đối chuẩn (Benchmark)
│   │       └── ensemble_benchmarks.py    #       Bagging, RF, ExtraTrees, HistGradientBoosting
│   │
│   └── evaluation/                       #   Đánh giá và đo đạc metrics dùng chung
│       └── metrics.py                    #     Accuracy, Precision, Recall, Macro-F1, Confusion Matrix
│
├── data/                                 # Thư mục dữ liệu
│   └── processed/                        #   Chứa mnist_processed.npz (sinh tự động từ notebook 0)
│
├── artifacts/                            # Lưu trữ tham số và chỉ số phân chia dữ liệu
│   └── split_indices.npz                 #   Chỉ số phân tầng Train (50k) / Val (10k) / Test (10k)
│
├── figures/                              # 33 biểu đồ thực nghiệm chất lượng cao (PNG)
├── results/                              # File CSV ghi nhận số liệu định lượng chi tiết
│   ├── 1_Bayes_metrics.csv               #   Kết quả Naïve Bayes
│   ├── 2_Logistic_metrics.csv            #   Kết quả Logistic Regression
│   ├── 3_MLP_CNN_metrics.csv             #   Kết quả MLP & CNN
│   ├── 4_Ensemble_metrics.csv            #   Kết quả AdaBoost & Ensemble
│   └── overall_comparison_metrics.csv    #   Bảng so sánh tổng hợp toàn diện các mô hình
│
├── tests/                                # Kiểm thử tự động (Unit Test / Smoke Test)
│   └── test_package_smoke.py             #   Kiểm tra tính toàn vẹn của API và pipeline
│
├── pyproject.toml                        # Cấu hình chuẩn hóa Package (PEP 517/518)
├── requirements.txt                      # Danh sách thư viện phụ thuộc
└── README.md                             # Tài liệu hướng dẫn đồ án
```

---

## Chi tiết các Notebook thực nghiệm

### 1. `notebook/0_Preprocessing_EDA.ipynb` — Khám phá dữ liệu & Tiền xử lý

- **Tải và khám phá dữ liệu (EDA):** Tải 70.000 ảnh MNIST gốc; kiểm tra tỷ lệ cân bằng giữa 10 chữ số (dao động hẹp quanh mức lý thuyết $10.0\%$, không bị lệch lớp).
- **Độ toàn vẹn:** Xác nhận không có giá trị khuyết thiếu (`NaN`) và không có bản ghi trùng lặp.
- **Phát hiện 67 pixel viền chết ($= 0$):** Phát hiện 67 pixel ở 4 mép viền ngoài luôn bằng $0$ trên toàn bộ các mẫu, khiến phương sai $\sigma^2 = 0$. Đây là cơ sở lý giải vì sao mô hình Gaussian Naïve Bayes thất bại khi chia cho 0 và phải chuyển sang Multinomial Naïve Bayes.
- **Chiến lược phân chia:** Phân tầng (Stratified Split) với seed ngẫu nhiên `42` thành: Train ($50.000$), Validation ($10.000$), Test ($10.000$).
- **Kỹ thuật chuẩn hóa:**
  1. Chuẩn hóa Min-Max $[0, 1]$ ($x_{\text{norm}} = x / 255.0$): Bảo toàn tính không âm của cường độ pixel cho Naïve Bayes và Logistic Regression.
  2. Chuẩn hóa Z-Score Standardized: Phục vụ mạng sâu MLP và CNN.
- Lưu dữ liệu nén vào `data/processed/mnist_processed.npz` và `artifacts/split_indices.npz`.

---

### 2. `notebook/1_Bayes.ipynb` — Multinomial Naïve Bayes

- **Nguyên lý toán học:** Mô hình hóa pixel theo phân phối đa thức; suy luận phân lớp theo quy tắc hậu nghiệm cực đại (MAP) trong miền Logarithm để tránh tràn số:
  $$
  \hat{y} = \arg\max_{c \in \{0, \dots, 9\}} \left[ \ln P(y = c) + \sum_{j=1}^d x_j \ln \theta_{cj} \right]
  $$
- **Làm mượt Laplace:** Khảo sát dải siêu tham số $\alpha \in [10^{-6}, 10^2]$ trên tập Validation; tìm ra điểm cực trị tối ưu tại $\alpha^* = 10^{-4}$ (đạt $82.15\%$ trên Val và $83.56\%$ trên Test).
- **Đối chiếu Scikit-Learn:** Chênh lệch độ chính xác so với `MultinomialNB` của Scikit-Learn chỉ $0.07\%$ với tỷ lệ đồng thuận dự đoán đạt trên $99.8\%$.
- **Phân tích lỗi:** Nhầm lẫn tập trung ở các cặp chữ số tương đồng nét mực như $(4 \leftrightarrow 9)$ và $(5 \to 3)$ do hạn chế của giả định độc lập có điều kiện bỏ qua liên kết không gian.

---

### 3. `notebook/2_Logistic.ipynb` — One-vs-Rest Logistic Regression

- **Chiến lược One-vs-Rest (OvR):** Huấn luyện đồng thời 10 bộ phân loại nhị phân tương ứng với 10 lớp chữ số.
- **Kỹ thuật chèn Bias & Mặt nạ điều chuẩn $L_2$:**
  - Chèn cột 1 vào ma trận đặc trưng: $\tilde{\mathbf{X}} = [\mathbf{X}, \mathbf{1}] \in \mathbb{R}^{N \times 785}$.
  - Áp dụng mặt nạ $\mathbf{m} = [1, \dots, 1, 0]^T$ trong số hạng điều chuẩn $L_2$ ($\lambda = 10^{-4}$) để không phạt hệ số tự do $w_0$, giữ siêu phẳng tự do tịnh tiến.
- **Vector hóa Gradient Descent song song:** Tính toán gradient và cập nhật toàn bộ ma trận trọng số $\mathbf{W} \in \mathbb{R}^{785 \times 10}$ cùng lúc:
  $$
  \nabla_{\mathbf{W}} J = \frac{1}{N}\tilde{\mathbf{X}}^T(\mathbf{P} - \mathbf{T}) + \lambda(\mathbf{m} \odot \mathbf{W}), \quad \mathbf{W} \leftarrow \mathbf{W} - \eta \nabla_{\mathbf{W}} J
  $$
- **Chống tràn số:** Tính hàm mất mát Binary Cross-Entropy ổn định bằng `np.logaddexp`.
- **Tối ưu siêu tham số:** Quét tốc độ học $\eta \in [0.1, 0.5, 1.0, 2.0]$; xác định $\eta^* = 0.5$ qua 30 epoch hội tụ tối ưu nhất (Validation Accuracy đạt $85.96\%$, Test Accuracy đạt $86.84\%$).

---

### 4. `notebook/3_MLP_CNN.ipynb` — Multi-Layer Perceptron & CNN

- **Multi-Layer Perceptron (From scratch):**
  - Kiến trúc mạng: $784 \to 256 \to 128 \to 10$.
  - Lan truyền ngược thủ công: Tự tính toán gradient giải tích theo chuỗi đạo hàm giải tích (BP1–BP4) mà không dùng `torch.autograd`.
  - Hàm kích hoạt ReLU, khởi tạo He Normal, cơ chế Inverted Dropout ($p=0.2$).
  - Bộ tối ưu SGD Momentum ($\beta=0.5$, $\text{lr}=0.1$), cơ chế Early Stopping theo dõi validation loss.
  - Phân tích hiện tượng tiêu biến gradient (Vanishing Gradient) và kiểm chứng bảo toàn phương sai kích hoạt qua các tầng ẩn.
  - Kết quả Test Accuracy: **$98.26\%$** với thời gian suy luận siêu nhanh ($0.002$ ms/mẫu).
- **Convolutional Neural Network (PyTorch):**
  - Kiến trúc chuẩn mực: `Conv2D(32 bộ lọc 4x4, stride=1)` $\to$ `ReLU` $\to$ `MaxPool2D(2x2)` $\to$ `Flatten` $\to$ `Dense(128)` $\to$ `Dense(10)`.
  - Tối ưu hóa bằng Adam ($\text{lr}=10^{-3}$), đạt Test Accuracy cao nhất toàn đồ án: **$98.76\%$**.
  - Khả năng trích xuất bất biến không gian vượt trội, giải quyết triệt để các trường hợp chữ viết tay nghiêng hoặc biến dạng nặng mà các mô hình tuyến tính thất bại.

---

### 5. `notebook/4_Ensemble.ipynb` — AdaBoost Ensemble

- **Cơ chế Boosting:** Kết hợp tuần tự các bộ phân lớp yếu (Weak Learners) thông qua cập nhật phân phối trọng số mẫu dữ liệu $D_t$ và trọng số biểu quyết $\alpha_t = \frac{1}{2}\ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$.
- **Khảo sát cấu hình bộ phân loại cơ sở:**
  - Cây quyết định nông (Decision Stumps, `max_depth=1`): Đạt $78.53\%$ trên tập Test.
  - Cây quyết định mở rộng (`max_depth=3`, $200$ estimators): Đạt Test Accuracy ấn tượng **$91.75\%$**.
  - Khảo sát biến thể kết hợp Logistic Regression và Gaussian Naïve Bayes.

---

### 6. `notebook/Compairisons.ipynb` — So sánh đối chuẩn toàn diện

- Đọc và đồng bộ dữ liệu từ 5 file CSV trong thư mục `results/`.
- Phân tích đường biên hiệu quả Pareto (Accuracy vs. Training Time & Inference Latency).
- Trực quan hóa ma trận nhầm lẫn tổng hợp trên cùng thang đo và vẽ biểu đồ Radar đa tiêu chí đánh giá toàn diện 5 chiều:
  1. Độ chính xác tổng thể (Accuracy)
  2. Khả năng khái quát hóa (Macro-F1)
  3. Tốc độ huấn luyện (Training Speed)
  4. Tốc độ suy luận thực tế (Inference Latency)
  5. Độ phức tạp tính toán và bộ nhớ (Model Footprint)

---

## Ranh giới triển khai thuật toán

| Mô hình                          | Thuật toán cốt lõi                                 |           Cách triển khai           | File cài đặt                                                        |
| :--------------------------------- | :----------------------------------------------------- | :------------------------------------: | :--------------------------------------------------------------------- |
| **Multinomial Naïve Bayes** | Đếm tần suất pixel, MAP Log-Domain, Laplace        |     **From Scratch (NumPy)**     | `src/pattern_recognition/models/from_scratch/naive_bayes.py`         |
| **Logistic Regression**      | Vectorized Batch GD, One-vs-Rest, L2 Mask              |     **From Scratch (NumPy)**     | `src/pattern_recognition/models/from_scratch/logistic_regression.py` |
| **Multi-Layer Perceptron**   | Lan truyền ngược thủ công (BP1-BP4), SGD Momentum | **From Scratch (Manual Tensor)** | `src/pattern_recognition/models/from_scratch/mlp.py`                 |
| **Decision Tree & Ensemble** | AdaBoost Boosting, Cập nhật trọng số mẫu $D_t$  |   **From Scratch / Benchmark**   | `src/pattern_recognition/models/from_scratch/ensemble.py`            |
| **CNN**                      | Conv2D $\to$ MaxPool2D $\to$ Dense, Adam           |      **PyTorch Framework**      | `src/pattern_recognition/models/pytorch/cnn.py`                      |
| **Benchmark Scikit-Learn**   | MNB, OvR Logistic, Bagging, Random Forest              |     **Scikit-Learn Library**     | `src/pattern_recognition/models/sklearn/ensemble_benchmarks.py`      |

---

## Hướng dẫn cài đặt và chạy thí nghiệm

### 1. Khởi tạo môi trường ảo

```powershell
# Di chuyển vào thư mục dự án
cd "Pattern-Recognition"

# Tạo môi trường ảo Python
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Cài đặt toàn bộ thư viện phụ thuộc
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# (Tùy chọn) Cài đặt package dạng editable để import module thuận tiện
python -m pip install -e . --no-deps
```

### 2. Thứ tự thực thi các Notebook

Các notebook cần được chạy theo đúng thứ tự logic để đảm bảo sinh đầy đủ dữ liệu và số liệu thực nghiệm:

```text
1. notebook/0_Preprocessing_EDA.ipynb   # Bắt buộc chạy đầu tiên để sinh mnist_processed.npz
2. notebook/1_Bayes.ipynb               # Huấn luyện Naïve Bayes -> xuất 1_Bayes_metrics.csv
3. notebook/2_Logistic.ipynb            # Huấn luyện Logistic Regression -> xuất 2_Logistic_metrics.csv
4. notebook/3_MLP_CNN.ipynb             # Huấn luyện MLP & CNN -> xuất 3_MLP_CNN_metrics.csv
5. notebook/4_Ensemble.ipynb            # Huấn luyện AdaBoost -> xuất 4_Ensemble_metrics.csv
6. notebook/Compairisons.ipynb          # Đọc toàn bộ CSV và vẽ biểu đồ so sánh tổng kết
```

### 3. Chạy kiểm thử tự động (Unit Tests)

Kiểm tra tính toàn vẹn của mã nguồn và các hàm toán học:

```powershell
python -m unittest discover -s tests -v
```

---

## Danh mục thư viện chính (`requirements.txt`)

* **Tính toán khoa học & Xử lý ma trận:** `numpy`, `scipy`, `pandas`
* **Trực quan hóa đồ thị:** `matplotlib`, `seaborn`
* **Mô hình học sâu & Xử lý tensor:** `torch`, `torchvision`
* **Mô hình đối chuẩn & Đánh giá metrics:** `scikit-learn`
* **Môi trường tính toán tương tác:** `jupyter`, `ipykernel`

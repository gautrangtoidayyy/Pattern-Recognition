# Nhận dạng Chữ số Viết tay MNIST

| Thông tin           | Chi tiết                                                                                                 |
| :------------------- | :-------------------------------------------------------------------------------------------------------- |
| **Môn học**  | Nhận dạng mẫu (Pattern Recognition)                                                                    |
| **Khoa**       | Toán - Tin học, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM                                    |
| **Dữ liệu**  | MNIST Handwritten Digits (70.000 ảnh $28 \times 28$, gồm 10 lớp chữ số 0–9)                       |
| **Mục tiêu** | Tự xây dựng thuật toán, tối ưu siêu tham số và đánh giá  4 nhóm mô hình                  |
| **Mã nguồn** | [github.com/gautrangtoidayyy/Pattern-Recognition](https://github.com/gautrangtoidayyy/Pattern-Recognition) |
| **Báo cáo**  | [Link Drive](https://drive.google.com/file/d/12Ti7kMtoMD5moBOjm66D8Z9XRSd5UXHf/view?usp=sharing)           |
| **Slide**      | [Link Canva](https://canva.link/wmztn5ue5ay06y9)                                                           |
| **Demo Web**   | [Link](https://pattern-recognition-mnist-handwritten-digit-recognition.r2u-hcmus.workers.dev/)             |

---

## 1. Kết quả thực nghiệm đối chuẩn

Tất cả các mô hình đều được huấn luyện trên cùng tập **Train ($50.000$ mẫu)**, tinh chỉnh siêu tham số trên tập **Validation ($10.000$ mẫu)** và đánh giá độc lập trên tập **Test ($10.000$ mẫu)** với seed ngẫu nhiên cố định `42`:

| Mô hình                          | Phương pháp                           |  Test Accuracy  |  Test Macro-F1  | Thời gian huấn luyện |     Độ trễ suy luận     |
| :--------------------------------- | :--------------------------------------- | :--------------: | :--------------: | :---------------------: | :-------------------------: |
| **CNN**                      | PyTorch (Conv2D + MaxPool + FC)          | **98.76%** | **98.76%** |         51.61 s         | **42.85 µs / mẫu** |
| **MLP**                      | Tự cài đặt (Manual Backprop)         | **98.26%** | **98.25%** |         33.57 s         |  **2.16 µs / mẫu**  |
| **AdaBoost**                 | Cây quyết định mở rộng (depth = 3) | **91.75%** | **91.70%** |       ~3.06 giờ       | **266.81 µs / mẫu** |
| **Logistic Regression**      | One-vs-Rest Batch GD                     | **86.84%** | **86.55%** |         10.70 s         |  **2.88 µs / mẫu**  |
| **Multinomial Naïve Bayes** | Thống kê MAP + Laplace                 | **83.56%** | **83.41%** |    **0.054 s**    | **26.89 µs / mẫu** |

### Nhận xét thực tế:

* **Độ chính xác:** **CNN** cho kết quả tốt nhất khi đạt **98.76%** (toàn bộ 10.000 ảnh kiểm thử chỉ dự đoán sai đúng **124 ảnh**) nhờ khả năng bảo toàn cấu trúc không gian 2D và trích xuất đặc trưng cục bộ của nét vẽ.
* **Cân bằng tối ưu giữa thời gian và hiệu năng:** **MLP tự cài đặt** là mô hình hiệu quả nhất khi chỉ mất **33.57 giây** huấn luyện, đạt độ chính xác **98.26%** (chỉ kém CNN 0.50%) và có tốc độ suy luận nhanh nhất (**2.16 µs/mẫu** — xử lý được hơn 460.000 ảnh/giây trên CPU đơn nhân).
* **Huấn luyện nhanh nhất:** **Multinomial Naïve Bayes** hoàn tất huấn luyện trong **0.054 giây** nhờ công thức thống kê giải tích đếm tần suất pixel, không cần vòng lặp gradient.
* **Đánh đổi kém hiệu quả:** **AdaBoost** tốn hơn **3 giờ** huấn luyện 200 cây quyết định nhưng độ chính xác chỉ đạt **91.75%** (kém mạng nơ-ron hơn 7%), cho thấy giải pháp Boosting chuỗi không tối ưu trên dữ liệu ảnh thô so với mạng nơ-ron.

---

## 2. Hướng dẫn cài đặt và chạy thử nghiệm

Dưới đây là các bước thiết lập môi trường và chạy thử nghiệm từ mã nguồn dự án:

### Bước 1: Tải mã nguồn về máy (Clone repository)

Mở Terminal và thực hiện lệnh:

```bash
git clone https://github.com/gautrangtoidayyy/Pattern-Recognition.git
cd Pattern-Recognition
```

### Bước 2: Tạo và kích hoạt môi trường ảo Python

Khuyến nghị sử dụng **Python 3.10 trở lên**:

* **Trên Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .venv\Scripts\Activate.ps1
  ```
* **Trên Windows (Command Prompt - CMD):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```
* **Trên Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Bước 3: Cài đặt các thư viện cần thiết

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

### Bước 4: Chạy thử nghiệm và kiểm chứng kết quả

Dự án cung cấp 2 phương thức kiểm chứng:

#### Lựa chọn 1: Trải nghiệm nhanh qua Giao diện Web Demo

Ứng dụng Web trực quan nằm trong thư mục `web/`, hoạt động độc lập mà không cần huấn luyện lại mô hình:

```bash
# Khởi động máy chủ cục bộ tại thư mục gốc dự án
python -m http.server 8000
```

Mở trình duyệt và truy cập:
👉 **`http://localhost:8000/web/`**
*(Vẽ trực tiếp chữ số bằng chuột trên Canvas hoặc chọn mẫu ảnh có sẵn từ tập Test để quan sát mô hình nhận diện thời gian thực)*.

---

#### Lựa chọn 2: Chạy kiểm chứng trên Jupyter Notebook

Toàn bộ dữ liệu tiền xử lý chuẩn `mnist_processed.npz` **đã được lưu sẵn trong thư mục `data/processed/`**, do đó không cần tải lại dữ liệu ban đầu.

* **Nếu sử dụng VS Code:**
  1. Mở thư mục dự án `Pattern-Recognition` trong VS Code.
  2. Mở notebook bất kỳ trong thư mục `notebook/`.
  3. Chọn Python Kernel ở góc trên bên phải là môi trường ảo `.venv` vừa tạo và chọn **Run All**.
* **Nếu chạy qua trình duyệt:**
  ```bash
  jupyter lab
  # hoặc: jupyter notebook
  ```

**Thứ tự thực thi các notebook trong thư mục `notebook/`:**

1. `0_Preprocessing_EDA.ipynb`: Phân tích khám phá dữ liệu (EDA), phân bố nhãn, phát hiện 67 pixel viền chết và chuẩn hóa dữ liệu.
2. `1_Bayes.ipynb`: Mô hình Multinomial Naïve Bayes tự cài đặt, khảo sát siêu tham số Laplace và đối chiếu Scikit-Learn.
3. `2_Logistic.ipynb`: Mô hình One-vs-Rest Logistic Regression tự cài đặt, tối ưu tốc độ học $\eta = 0.5$ và đối chiếu Scikit-Learn.
4. `3_MLP_CNN.ipynb`: Mô hình MLP tự cài đặt lan truyền ngược thủ công và mô hình CNN xây dựng trên PyTorch.
5. `4_Ensemble.ipynb`: Mô hình AdaBoost SAMME đa lớp kết hợp Cây quyết định.
6. `Compairisons.ipynb`: Tổng hợp kết quả từ các mô hình, phân tích đánh đổi độ chính xác — thời gian và vẽ biểu đồ Radar 5 chiều.

---

## 3. Chi tiết nội dung các Notebook

### 1. `0_Preprocessing_EDA.ipynb` — Khám phá dữ liệu & Tiền xử lý

* **Phân bố dữ liệu:** 70.000 ảnh MNIST có phân bố 10 lớp rất đồng đều (~10% mỗi chữ số), không bị lệch lớp.
* **Phát hiện 67 pixel viền chết ($= 0$):** Có 67 pixel ở 4 góc viền ngoài luôn bằng 0 trên mọi mẫu ảnh, khiến phương sai $\sigma^2 = 0$. Đây là lý do mô hình Gaussian Naïve Bayes bị lỗi chia cho 0 và cần chuyển sang Multinomial Naïve Bayes.
* **Phân chia dữ liệu:** Phân tầng ngẫu nhiên (Stratified Split) với seed `42` thành: Train (50.000), Validation (10.000) và Test (10.000).
* **Kỹ thuật chuẩn hóa:** Chuẩn hóa Min-Max $[0, 1]$ cho Naïve Bayes & Logistic Regression; chuẩn hóa Z-Score cho MLP & CNN.

### 2. `1_Bayes.ipynb` — Multinomial Naïve Bayes

* Mô hình hóa pixel theo phân phối đa thức; suy luận nhãn theo quy tắc hậu nghiệm cực đại (MAP) trong miền Logarithm để chống tràn số dưới.
* Áp dụng **Làm mượt Laplace (Laplace smoothing)** để khắc phục các pixel bằng 0. Giá trị tối ưu tìm được qua thực nghiệm là $\alpha^* = 10^{-4}$ (đạt $83.56\%$ trên tập Test).
* Kết quả tự cài đặt bám sát mô hình `MultinomialNB` của Scikit-Learn ($83.63\%$) với tỷ lệ đồng thuận trên $99.8\%$.

### 3. `2_Logistic.ipynb` — One-vs-Rest Logistic Regression

* Huấn luyện 10 bộ phân loại nhị phân song song tương ứng với 10 lớp chữ số.
* **Kỹ thuật tối ưu:**
  - Chèn cột bias trick $[\mathbf{X}, \mathbf{1}]$ để gom trọng số thành ma trận $\mathbf{W} \in \mathbb{R}^{785 \times 10}$.
  - Sử dụng mặt nạ $\mathbf{m} = [1, \dots, 1, 0]^T$ trong số hạng điều chuẩn $L_2$ để **không phạt hệ số tự do bias**, giữ cho các siêu phẳng phân lớp tự do dịch chuyển.
  - Vector hóa toàn bộ quá trình tính Gradient Descent bằng phép nhân ma trận song song 10 lớp, giúp hoàn thành 30 epoch trong khoảng 10 giây.
* Tốc độ học tối ưu tìm được là $\eta^* = 0.5$, đạt độ chính xác Test **$86.84\%$**.

### 4. `3_MLP_CNN.ipynb` — Multi-Layer Perceptron & CNN

* **Mạng MLP tự cài đặt:**
  - Cấu trúc mạng: $784 \to 256 \to 128 \to 10$.
  - Tính toán đạo hàm giải tích và lan truyền ngược (Backpropagation) hoàn toàn thủ công bằng các công thức BP1–BP4, không sử dụng tính năng autograd tự động.
  - Kết hợp hàm kích hoạt ReLU, khởi tạo He Normal, Inverted Dropout ($p=0.2$) và thuật toán tối ưu SGD Momentum.
  - Đạt độ chính xác Test **$98.26\%$** với tốc độ suy luận nhanh (**$2.16$ µs/mẫu**).
* **Mạng CNN (PyTorch):**
  - Cấu trúc: `Conv2D(32 bộ lọc 4x4)` $\to$ `ReLU` $\to$ `MaxPool2D(2x2)` $\to$ `Dense(128)` $\to$ `Dense(10)`.
  - Tối ưu bằng thuật toán Adam, đạt độ chính xác cao nhất toàn đồ án: **$98.76\%$** (chỉ sai 124 mẫu trên toàn bộ 10.000 mẫu kiểm thử).

### 5. `4_Ensemble.ipynb` — AdaBoost SAMME

* Triển khai giải thuật **AdaBoost SAMME đa lớp** ($K=10$), tuần tự cập nhật phân phối trọng số mẫu sai $D_t$ và tính trọng số biểu quyết $\alpha_t$ có hiệu chỉnh theo số lượng lớp.
* So sánh các bộ phân loại yếu: Decision Stumps (độ sâu 1 đạt $78.53\%$), Cây quyết định độ sâu 3 (đạt $91.75\%$), Logistic Regression và Gaussian NB.

### 6. `Compairisons.ipynb` — So sánh đối chuẩn toàn diện

* Đọc dữ liệu tự động từ các file kết quả CSV trong thư mục `results/`.
* Phân tích đường biên Pareto giữa độ chính xác và chi phí thời gian tính toán.
* Trực quan hóa ma trận nhầm lẫn tổng hợp và vẽ biểu đồ Radar đa tiêu chí (Accuracy, Macro-F1, Tốc độ học, Tốc độ suy luận, Độ phức tạp mô hình).

---

## 4. Phương pháp hiện thực các mô hình

Bảng tóm tắt phương pháp hiện thực cho từng mô hình và file mã nguồn tương ứng trong dự án:

| Mô hình                           | Thuật toán cốt lõi                                  |         Cách triển khai         | File cài đặt tương ứng                                           |
| :---------------------------------- | :------------------------------------------------------ | :-------------------------------: | :--------------------------------------------------------------------- |
| **Multinomial Naïve Bayes**  | Thống kê hạt mực, Laplace smoothing, Log-MAP        |     **Tự cài đặt**     | `src/pattern_recognition/models/from_scratch/naive_bayes.py`         |
| **Logistic Regression**       | Vectorized Batch GD, One-vs-Rest, L2 Mask               |     **Tự cài đặt**     | `src/pattern_recognition/models/from_scratch/logistic_regression.py` |
| **Multi-Layer Perceptron**    | Lan truyền ngược thủ công (BP1–BP4), SGD Momentum |     **Tự cài đặt**     | `src/pattern_recognition/models/from_scratch/mlp.py`                 |
| **AdaBoost Ensemble**         | Thuật toán SAMME đa lớp, cập nhật trọng số mẫu |     **Tự cài đặt**     | `src/pattern_recognition/models/from_scratch/ensemble.py`            |
| **CNN**                       | Conv2D $\to$ MaxPool2D $\to$ Dense, Adam            |   **Thư viện PyTorch**   | `src/pattern_recognition/models/pytorch/cnn.py`                      |
| **Đối chuẩn Scikit-Learn** | MNB, Logistic, Bagging, Random Forest                   | **Thư viện Scikit-Learn** | `src/pattern_recognition/models/sklearn/ensemble_benchmarks.py`      |

---

## 5. Cấu trúc thư mục dự án

```text
Pattern-Recognition/
│
├── web/                                  # Ứng dụng Web Demo tương tác (HTML5 Canvas + Three.js + TF.js)
│   ├── index.html                        #   Giao diện bảng vẽ và bảng điều khiển trực quan
│   ├── app.js, engine.js, panels.js      #   Xử lý nét vẽ và suy luận nhãn thời gian thực
│   ├── network3d.js, cnn-animation.js    #   Mô phỏng 3D kiến trúc nơ-ron và hoạt hóa bộ lọc
│   └── assets/                           #   Trọng số mô hình và dữ liệu mẫu kiểm thử
│
├── notebook/                             # Chuỗi Jupyter Notebook thực nghiệm chính
│   ├── 0_Preprocessing_EDA.ipynb         #   Tiền xử lý và phân tích khám phá dữ liệu
│   ├── 1_Bayes.ipynb                     #   Mô hình Multinomial Naïve Bayes
│   ├── 2_Logistic.ipynb                  #   Mô hình One-vs-Rest Logistic Regression
│   ├── 3_MLP_CNN.ipynb                   #   Mô hình MLP (tự cài đặt) & CNN
│   ├── 4_Ensemble.ipynb                  #   Mô hình AdaBoost SAMME
│   └── Compairisons.ipynb                #   So sánh đối chuẩn tổng hợp & biểu đồ Radar
│
├── src/pattern_recognition/              # Package Python tái sử dụng thuật toán
│   ├── __init__.py
│   ├── config.py                         #   Cấu hình seed=42 và  đường dẫn
│   ├── data.py                           #   Hàm đọc dữ liệu đã tiền xử lý
│   ├── models/                           #   Mã nguồn các mô hình tự cài đặt & PyTorch
│   └── evaluation/metrics.py             #   Hàm tính toán các chỉ số đánh giá
│
├── data/processed/                       # Dữ liệu mnist_processed.npz đã xử lý sẵn
├── artifacts/split_indices.npz           # Chỉ số phân chia tập Train / Val / Test cố định
├── figures/                              # Các biểu đồ thực nghiệm xuất ra từ notebook
├── results/                              # Các file CSV lưu trữ metrics định lượng
├── tests/test_package_smoke.py           # Bộ kiểm thử tự động
├── requirements.txt                      # Danh sách thư viện phụ thuộc
└── README.md                             # Tài liệu thuyết minh đồ án
```

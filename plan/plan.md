# KẾ HOẠCH TRIỂN KHAI ĐỒ ÁN GIỮA KỲ — NHẬN DẠNG CHỮ SỐ VIẾT TAY (MNIST)

> **Tài liệu nguồn:** `DoAn_ThucHanh_NhanDang.pdf`, phần **ĐỒ ÁN 1 (GIỮA KỲ)** ở trang 1–2 và **PHỤ LỤC** trang 4.  
> **Notebook tham khảo:** `BTVN_Lab01.ipynb`, `NaiveBayes(1).ipynb`, `TL_Lab03.ipynb`.  
> **Đầu ra của bước lập kế hoạch này:** `plan.md` và 3 notebook **chỉ gồm các ô Markdown giao việc, chưa viết code/chưa có kết quả thực nghiệm**.

## 0. Phạm vi và cách đọc yêu cầu

- [ ] **Bốn mô hình bắt buộc (theo PDF):** (1) Gaussian Naïve Bayes, (2) Logistic Regression tuyến tính huấn luyện Gradient Descent, (3) MLP huấn luyện backpropagation, (4) Ensemble. Kế hoạch **chọn AdaBoost đa lớp SAMME với decision stump** làm Ensemble để có thể tự cài đặt tính toán lõi. Các lựa chọn thuật toán cụ thể này là **đề xuất triển khai**, không phải bắt buộc phải chọn đúng biến thể đó.
- [ ] **CNN đơn giản** là **phần nâng cao của PDF**, nhưng **bắt buộc trong phạm vi 3 phase theo yêu cầu hiện tại**. Báo cáo so sánh **4 mô hình cốt lõi** theo rubric, và bổ sung **CNN thành mô hình thứ 5** ở bảng/mục riêng.
- [ ] PDF yêu cầu **ít nhất một thí nghiệm siêu tham số cho toàn đồ án**; phạm vi được giao ở đây đặt mức cao hơn: **mỗi phase phải có ≥ 1 thí nghiệm và biểu đồ**.
- [ ] Các phần **dùng chung ở cả ba notebook**: tái lập dữ liệu MNIST/split, kiểm tra dữ liệu và nhãn, chặn rò rỉ dữ liệu, đánh giá đủ **train / validation / test** (accuracy, precision, recall, F1 **macro**, confusion matrix), đo thời gian, thí nghiệm siêu tham số có biểu đồ, phân tích lỗi, lưu kết quả, nguồn tham khảo/nhận xét.
- [ ] Không triển khai đồ án 2 về khuôn mặt, LBP, PCA/LDA hay clustering; đó là phần **cuối kỳ** ở PDF, không thuộc nhiệm vụ này.

## 1. Quy định kỹ thuật không được bỏ sót

### 1.1. Quy định của đề

- [ ] **Tự viết phần tính toán lõi** của Naïve Bayes, bộ phân lớp tuyến tính và Ensemble. `scikit-learn` chỉ dùng để **đối chiếu** mô hình, hoặc dùng công cụ phụ trợ không thay thế lõi (chia tập, metric, vẽ). Không nộp lời giải chỉ gọi `GaussianNB.fit`, `LogisticRegression.fit` hay `AdaBoostClassifier.fit` làm mô hình chính.
- [ ] Với **MLP**, thể hiện rõ **forward / loss / backpropagation / cập nhật trọng số**. Đề cho phép PyTorch/TensorFlow làm phần khung huấn luyện MLP; ưu tiên tự triển khai kỹ thuật chính và diễn giải công thức/pseudo-code.
- [ ] Với **CNN**, đây là phần nâng cao; tự thiết kế/cài đặt thành phần cần thiết. **PDF không nêu rõ quyền dùng khung PyTorch/TensorFlow cho CNN** như đối với MLP: thống nhất với giảng viên trước khi dùng framework cho CNN, hoặc tự cài đặt phần lõi tương ứng, tránh vi phạm phụ lục.
- [ ] Mã/ý tưởng kế thừa từ lab hoặc tài liệu bên ngoài phải ghi nguồn rõ ràng; AI được phép hỗ trợ nhưng **phải khai báo trong báo cáo** và nhóm phải tự chịu trách nhiệm hiểu code.
- [ ] Khi có bảng/biểu đồ/kết luận, không ghi sẵn con số kỳ vọng; chỉ báo cáo số đo chạy thực tế. Không sử dụng validation/test để học tham số hay chọn đặc trưng; chỉ **validation** để chọn hyperparameter; **test** dành cho đánh giá cuối sau khi chốt thiết lập.

### 1.2. Quy ước dữ liệu và tái lập (đề xuất thống nhất)

- [ ] Dùng **MNIST 28 × 28, 10 lớp từ 0 đến 9**; nguồn chuẩn **60.000 ảnh train gốc + 10.000 ảnh test gốc**. *Không nhầm `sklearn.datasets.load_digits` (8 × 8) với MNIST (28 × 28)!*
- [ ] Chỉ chia **60.000 mẫu train gốc** thành **train = 54.000, validation = 6.000** (tỉ lệ 90/10, `stratify`, `SEED=42`). **10.000 mẫu test gốc** giữ nguyên; không trộn vào train/validation.
- [ ] **Bắt buộc dùng chung một tệp `artifacts/split_indices.npz`:** Phase 1 tạo một lần (hoặc kiểm tra và dùng lại nếu đã có); lưu `train_idx`, `val_idx` trên 60.000 ảnh train gốc, `test_idx` trên 10.000 ảnh test gốc, `seed=42`, `train_source_sha256`, `test_source_sha256` (SHA-256 ảnh + nhãn nguồn theo **đúng thứ tự**). Phase 2/3 **đọc bắt buộc** tệp này, không tự chia lại chỉ dựa trên seed; nếu thiếu/sai hash thì dừng và yêu cầu tạo/sửa ở Phase 1. Mỗi notebook **chạy độc lập với kernel** nhưng cùng dùng tệp chia đã khóa.
- [ ] **Quy ước kiểm hash giống nhau ở cả 3 phase:** lấy **ảnh MNIST gốc** với dtype `uint8`, shape `(N,28,28)` và nhãn gốc `uint8`, shape `(N,)`; tính `sha256(X_raw.tobytes(order="C") + y_raw.tobytes(order="C")).hexdigest()` riêng cho tập train nguồn 60k và test nguồn 10k (trước khi chia, scale hay xáo trộn). Như vậy hash kiểm **cả nội dung lẫn thứ tự ảnh/nhãn** giữa các notebook.
- [ ] Chuẩn hóa ảnh theo **pixel / 255.0** thành `float32` trong [0, 1]; duy trì bản ảnh **(N, 28, 28)**, bản vector phẳng **(N, 784)**, và bản tensor cho CNN **(N, 1, 28, 28)**. MLP dùng 784 chiều; NB, Logistic, AdaBoost dùng vector 784 chiều. Không áp đặt z-score cho CNN/NB nếu không có lý do thực nghiệm.
- [ ] Với phép biến đổi phải `fit` tham số từ **train** (nếu có như z-score) rồi chỉ `transform` validation/test; không tạo leakage. **Ghi rõ pipeline tiền xử lý riêng theo mô hình**: GaussianNB/AdaBoost trên pixel `/255`, Logistic có thể thêm z-score fit train, MLP/CNN giữ representation và chuẩn hóa phù hợp. Khóa **cùng thứ tự mẫu + nhãn** nhưng không ép các mô hình dùng chung scaler.
- [ ] Dùng chung `labels = [0, 1, ..., 9]`, quy ước `zero_division=0` khi lớp không có dự đoán và cùng chính sách đo thời gian cho cả phase.

### 1.3. Giao diện kết quả chung để phase 3 so sánh được

**Các tệp khuyến nghị tạo khi bắt đầu code (chưa được tạo trong bộ khung này):**

```text
project/
├── plan.md
├── phase1.ipynb
├── phase2.ipynb
├── phase3.ipynb
├── BTVN_Lab01.ipynb            # notebook tham khảo (nếu đặt cùng thư mục)
├── NaiveBayes(1).ipynb         # notebook tham khảo
├── TL_Lab03.ipynb              # notebook tham khảo
├── README.md                   # cần viết ở giai đoạn hoàn thiện
├── artifacts/
│   └── split_indices.npz       # BẮT BUỘC: chỉ số + hash nguồn/nhãn theo đúng thứ tự
├── models/                     # tham số/checkpoint do từng phase xuất ra
├── results/
│   ├── phase1_metrics.csv
│   ├── phase2_metrics.csv
│   ├── phase3_metrics.csv
│   └── comparison.csv          # tổng hợp ở phase 3
└── figures/                    # biểu đồ lưu từ notebook
```

- [ ] Dùng **một hàng cho mỗi `(model, split)`**; yêu cầu tối thiểu các cột: `phase, model, split, accuracy, precision_macro, recall_macro, f1_macro, train_seconds, predict_seconds, n_samples, seed, params`.
- [ ] Giữ cách gọi `split ∈ {train, validation, test}` và tên mô hình ổn định: `GaussianNB-NP`, `LogReg-GD-NP`, `MLP`, `CNN`, `AdaBoost-SAMME-NP`.
- [ ] `train_seconds`: thời gian **fit**; `predict_seconds`: thời gian dự đoán trên **từng split**. Không tính thời gian tải dữ liệu, vẽ biểu đồ vào fit/predict; ghi rõ CPU/GPU/phiên bản khi so sánh tốc độ (đặc biệt CNN vs NumPy).
- [ ] Mỗi phase xuất bảng đủ các độ đo trên **cả 3 tập**, hình confusion matrix cho **cả 3 tập**, thêm biểu đồ siêu tham số và một đoạn biện luận ngắn. Có thể lưu thêm báo cáo per-class và ma trận nhầm lẫn thành tệp riêng.

## 2. Kế thừa chính xác từ 3 notebook đã nộp kèm

| Notebook gốc | Phần có thể tái sử dụng | Chuyển sang MNIST / việc còn thiếu |
|---|---|---|
| `BTVN_Lab01.ipynb` | Ô **Lý thuyết về các độ đo phân lớp**: confusion matrix, accuracy, precision, recall, F1, macro; quy tắc luồng `DATA → X,y → split → preprocessing → fit → predict → metrics → visualize → analyze`, kiểm tra missing/duplicate/nhãn, phản biện theo biểu đồ. | Đây chủ yếu là **lý thuyết và checklist**; code nạp Iris/Wine không phải code MNIST. Bổ sung metric 10 lớp trên train/val/test và vẽ ảnh thay scatter Iris/Wine. |
| `NaiveBayes(1).ipynb` | **Phong cách trình bày**: đầu bài có bảng 6 bước, `### mục`, `**TODO:**`, câu hỏi biện luận; mục `2.1 GaussianNB`, `2.2 MultinomialNB (Digits)` nêu tham số, 10 ảnh, phân bố lớp, ảnh trung bình, top lỗi, alpha và confusion matrix. | Notebook này **chỉ có 3 ô Markdown, không có code triển khai lõi để sao chép**. `Digits` ở đây là dữ liệu **8×8**, phải thay bằng MNIST **28×28**. Phần dùng `GaussianNB`/`MultinomialNB` của sklearn **chỉ dùng đối chiếu**, tự viết NB chính. |
| `TL_Lab03.ipynb` | Phần `0–1`: import, `SEED`, `add_bias`, `Standardizer`, `split_and_scale` (mới train/test); phần `2.2`: `sigmoid`, khung `LogisticRegressionNP`; phần `2.4`: `OneVsRest`; phần `3.3`: `evaluate`, `summarize`, `time.perf_counter`; phần `4`: đối chiếu scikit-learn; phần `5`: câu hỏi thảo luận. | Cần sửa thành **train/validation/test** và **10 lớp, 784 chiều**. `LogisticRegressionNP.fit` chứa **`# TODO` và `raise NotImplementedError`**, phải tự hoàn thành Gradient Descent; không sao chép rồi coi như đã xong. `OneVsRest` có thể kế thừa nhưng cần xử lý/tối ưu cho MNIST. `LinearSVMNP.fit` cũng có `NotImplementedError`, không chọn SVM nếu chưa hoàn thiện. |

**Quy tắc tái sử dụng:** giữ ý tưởng/luồng, tận dụng các hàm đã hoàn chỉnh; không sử dụng code gọi model dựng sẵn của sklearn làm lời giải chính; dẫn nguồn notebook học phần trong báo cáo/code comment khi áp dụng.

## 3. PHASE 1 — Dữ liệu + Gaussian Naïve Bayes + Logistic Regression

Notebook: **`phase1.ipynb`**. Đây là phase thiết lập chuẩn dữ liệu để các phase sau chạy cùng nền.

### A. Chuẩn bị và khám phá MNIST
- [ ] 0. Thiết lập seed, thư viện, cấu trúc thư mục output, ghi môi trường thực thi.
- [ ] 1.1. Nạp **MNIST chuẩn**; kiểm `shape`, `dtype`, miền pixel, kích thước, nhãn 0–9; kiểm **60k/10k**.
- [ ] 1.2. EDA chất lượng ảnh: missing, ảnh trùng lặp (ghi số lượng; không tự ý loại), lớp thiếu, lớp lệch, pixel hằng số; giải thích vì sao dữ liệu ảnh khác dữ liệu bảng.
- [ ] 1.3. Tạo **≥ 3 hình** tham khảo format NaiveBayes/BTVN: (i) lưới ảnh gán nhãn 0–9, (ii) phân bố 10 lớp ở train/val/test, (iii) ảnh trung bình 10 lớp hoặc phân bố mức xám; tiêu đề/trục/chú thích phù hợp.
- [ ] 1.4. Chia train/val 54k/6k có `stratify`, giữ test 10k; **tạo/khóa `artifacts/split_indices.npz` với train_idx/val_idx/test_idx và hash nguồn đúng thứ tự**; kiểm tập không chồng.
- [ ] 1.5. Chuẩn hóa [0,1], làm phẳng `(N,784)`, kiểm min/max/shape; tránh fit scaler trước split. Viết nhận xét về đặc trưng pixel và cặp số dễ nhầm lẫn.

### B. Gaussian Naïve Bayes **tự cài đặt**
- [ ] 2.1. Nêu giả định độc lập có điều kiện và công thức log-posterior / log-Gaussian; tham số prior, mean, variance per class.
- [ ] 2.2. Viết `fit`: học `class_prior_`, `theta_`, `var_` trên **train**; thêm `var_smoothing`/epsilon cho phương sai 0.
- [ ] 2.3. Viết `predict` / `predict_log_proba` từ tổng log-likelihood tránh underflow; kiểm output class 0–9.
- [ ] 2.4. In/giải thích tham số (shape 10×784, prior 10 lớp); không dùng `sklearn.naive_bayes.GaussianNB` ở lõi.
- [ ] 2.5. Đối chiếu sklearn GaussianNB **trên cùng split, cùng chế độ variance smoothing**, ghi chênh lệch có lý giải.
- [ ] 2.6. Đánh giá **train/validation/test**: accuracy, precision/recall/F1 macro, confusion matrix 10×10 cả ba tập, per-class nếu cần; top lỗi và vài ảnh dự đoán sai; nhận xét overfit/underfit.

### C. Logistic Regression tuyến tính **tự cài đặt, GD**
- [ ] 3.1. Giải thích sigmoid, binary cross-entropy + L2, gradient, cập nhật GD, ý nghĩa learning rate/epoch; chiến lược **one-vs-rest 10 lớp**.
- [ ] 3.2. Kế thừa `add_bias`, `sigmoid`, bộ khung `LogisticRegressionNP`, `OneVsRest` từ `TL_Lab03.ipynb`; **hoàn thiện ô code đang `NotImplementedError`** và điều chỉnh mini-batch nếu cần hiệu năng.
- [ ] 3.3. Huấn luyện 10 bộ phân lớp nhị phân trên **train**, chọn số epoch/lr/L2 trên validation; lưu loss, kiểm gradient/hội tụ, tránh xác suất tràn số và ảnh hưởng của mất cân bằng OvR.
- [ ] 3.4. Đối chiếu `sklearn` LogisticRegression + OneVsRest **chỉ làm baseline**, cùng split và điều chuẩn tương thích; ghi khác biệt tối ưu/scale nếu có.

### D. Thí nghiệm, xuất dữ liệu và nhận xét
- [ ] 4.1. Chuẩn hóa hàm đánh giá và schema CSV chung cho hai mô hình, nhất quán nhãn 0–9 và thời gian dự đoán trên mỗi split.
- [ ] 4.2. Làm **≥ 1 thí nghiệm có đồ thị** ở phase này: **learning rate** của Logistic Regression (đề xuất `η={0.01,0.05,0.1}`; giữ seed, cách chia, số epoch và L2 cố định), trục x là lr, trục y validation F1 macro/accuracy; ghi thời gian nếu hữu ích. Có thể trình bày thêm biểu đồ `var_smoothing` của NB **đã chọn trên validation trước khi đánh giá test ở Phase 1 mục 2.5**.
- [ ] 4.3. Đánh giá Logistic đã chốt bằng validation trên **train/validation/test**; confusion matrix cả ba tập, top lỗi có ảnh, loss curve và phân tích.
- [ ] 4.4. Bảng nội bộ **Naïve Bayes vs Logistic**: trên train/val/test và fit/predict time; chỉ kết luận dựa trên số liệu.
- [ ] 4.5. Xuất `results/phase1_metrics.csv`, mô hình/tham số nếu cần, `figures/` và split indices; ghi chú phương án được chốt theo validation.
- [ ] 4.6. Viết tóm tắt cho báo cáo: dữ liệu/EDA, công thức + pseudo-code, siêu tham số, bảng độ đo, biểu đồ, các nhầm lẫn điển hình, tài liệu/AI acknowledgement.

**Definition of Done — Phase 1:** hai mô hình chạy lõi độc lập sklearn; metric + CM đủ 3 split cho mỗi mô hình; ≥1 biểu đồ hyperparameter; notebook chạy lại được; CSV chuẩn chung có 6 dòng (2 mô hình × 3 split).

## 4. PHASE 2 — MLP backpropagation + CNN nâng cao

Notebook: **`phase2.ipynb`**.

### A. Tái sử dụng tiền xử lý
- [ ] 1.1. Đọc MNIST nguồn gốc và **bắt buộc đọc `artifacts/split_indices.npz` từ Phase 1**, kiểm hash/thứ tự ảnh + nhãn, 54k/6k/10k, lớp 0–9, shape, pixel/255; không phụ thuộc biến sống trong kernel Phase 1.
- [ ] 1.2. Tạo **2 representation từ cùng ảnh**: flattened `(N,784)` cho MLP, tensor `(N,1,28,28)` cho CNN; kiểm ảnh trước/sau reshape khớp tuyệt đối.
- [ ] 1.3. Có ô kiểm phân bố 10 lớp / vài ảnh và thống kê, để chứng minh các phase cùng dữ liệu; xác nhận không augmentation cho validation/test.

### B. MLP huấn luyện backpropagation
- [ ] 2.1. Vẽ/diễn tả kiến trúc ví dụ `784 → 256 → 128 → 10`, activation ReLU, softmax/cross-entropy; tính kích thước tham số.
- [ ] 2.2. Trình bày forward, loss, chain rule, backprop và công thức SGD/mini-batch; cài đặt phần toán học lõi theo quy định (NumPy hoặc MLP framework được phép).
- [ ] 2.3. Tự viết training loop: mini-batch, `forward → loss → backward → optimizer step`, fixed seed, loss/accuracy train–validation theo epoch; kiểm gradient/cập nhật tham số và hình dạng đầu ra.
- [ ] 2.4. Dùng **validation** chọn checkpoint/early stopping, ghi epochs thực tế, learning rate, batch size và thiết bị.
- [ ] 2.5. Đánh giá **train/validation/test** với đủ 5 tiêu chí, confusion matrix 10×10 cho từng tập, so sánh lỗi thường gặp với Phase 1.
- [ ] 2.6. Đối chiếu MLP bằng `sklearn.neural_network.MLPClassifier` **chỉ để kiểm chứng** trên cùng split (nếu chạy khả thi); nêu khác biệt optimizer/kiến trúc, không đưa baseline sklearn thay mô hình MLP chính.

### C. CNN đơn giản và so sánh với MLP
- [ ] 3.1. Thiết kế CNN **không pretrained** (ví dụ 2 Conv–ReLU–Pool, Flatten, Dense→10); ghi shape từng lớp, số tham số, lý do tận dụng cấu trúc không gian.
- [ ] 3.2. Cài đặt, train loop, backprop và lựa chọn framework theo **quy định PDF/phê duyệt giảng viên**, giữ augmentation/seed/split công bằng với MLP.
- [ ] 3.3. Chọn checkpoint theo validation, không xem test để tuning; lưu learning curves.
- [ ] 3.4. Đánh giá **train/validation/test**, đủ metric/confusion matrix; ảnh lỗi chung và lỗi riêng; bảng **CNN vs MLP** về accuracy/F1 macro, fit/predict time, số tham số, overfit và khả năng giải thích.

### D. Siêu tham số, biểu đồ, đầu ra
- [ ] 4.1. **≥1 thí nghiệm + biểu đồ**: số nơ-ron hidden của MLP (`64, 128, 256`) hoặc learning rate (`0.001, 0.01, 0.05` tùy optimizer); cố định phần còn lại, vẽ validation F1 macro/accuracy vs hyperparameter. Có thể kèm train–validation loss theo epoch.
- [ ] 4.2. Ghi chênh lệch thời gian CPU/GPU và giới hạn so sánh; chỉ kết luận từ giá trị đo được.
- [ ] 4.3. Xuất `results/phase2_metrics.csv` (6 dòng: MLP + CNN × 3 split), checkpoints và figures; ghi công thức, nguồn lab/chính sách AI.

**Definition of Done — Phase 2:** MLP có backprop đúng, CNN nâng cao được triển khai hợp quy định; mỗi mô hình có 3 × metrics/CM; ≥1 hyperparameter experiment có đồ thị; MLP vs CNN có bảng/biện luận và `phase2_metrics.csv` đúng schema.

## 5. PHASE 3 — Ensemble tự cài đặt + đối chiếu 4 mô hình bắt buộc

Notebook: **`phase3.ipynb`**.

### A. Dữ liệu và chuẩn đánh giá chung
- [ ] 1.1. Nạp MNIST 60k/10k và **bắt buộc đọc `artifacts/split_indices.npz`**, kiểm hash nguồn/thứ tự mẫu + nhãn, 54k/6k/10k; dùng vector 784 chiều [0,1] riêng cho AdaBoost, có thống kê nhanh và hình kiểm tra.
- [ ] 1.2. Viết/lấy lại một hàm đánh giá thống nhất cho `train, validation, test`: metrics 5 loại; thời gian fit và predict, confusion matrix 10×10 từng split; kiểm không đọc nhãn validation/test để xây weak learner.

### B. AdaBoost-SAMME đa lớp, lõi tự cài đặt
- [ ] 2.1. Nêu ý tưởng boosting, trọng số mẫu, mục tiêu giảm lỗi, và quy tắc SAMME cho **10 lớp** (khác AdaBoost nhị phân).
- [ ] 2.2. Cài **decision stump** 1 tầng có trọng số: tìm `(feature, threshold)` và nhãn dự đoán của hai nhánh theo lỗi có trọng số; quản lý tie/ảnh pixel nhiều mức và tốc độ trên 784 features.
- [ ] 2.3. Cài loop AdaBoost: khởi tạo trọng số đều; chọn stump theo trọng số; lỗi có trọng số; alpha SAMME; cập nhật + chuẩn hóa trọng số; chặn lỗi epsilon/weak learner không đạt điều kiện; lưu danh sách stump/alpha.
- [ ] 2.4. Cài `decision_function/predict` bằng tổng điểm có trọng số theo 10 lớp; kiểm nhãn hợp lệ và accuracy tăng qua nhiều vòng trên tập thử nhỏ.
- [ ] 2.5. Đối chiếu với `sklearn.ensemble.AdaBoostClassifier` theo **cùng base stump/thuật toán tương thích với phiên bản**, **chỉ để kiểm chứng**; không dùng sklearn ở lõi.

### C. Siêu tham số có biểu đồ
- [ ] 3.1. **≥1 thí nghiệm**: số weak learners / số vòng boosting (`n_estimators = 10, 30, 60` là gợi ý, điều chỉnh theo runtime thực tế); giữ seed, depth stump và learning_rate không đổi; vẽ **validation accuracy/F1 macro** theo số vòng, có thể vẽ **thời gian train**.
- [ ] 3.2. Có thể khảo sát bổ sung `learning_rate`/ngưỡng/feature subsampling; lựa chọn bằng validation, không tuning theo test.
- [ ] 3.3. **Sau khi chốt siêu tham số theo validation**, đánh giá Ensemble trên **train/validation/test** đủ accuracy/precision_macro/recall_macro/F1_macro/confusion matrix; top lỗi/ảnh nhầm; ghi fit/predict time và hạn chế thuật toán. Test chỉ đánh giá cuối.

### D. So sánh tổng hợp và biện luận
- [ ] 4.1. Import `results/phase1_metrics.csv`, `results/phase2_metrics.csv` (hoặc chạy lại các phase nếu chưa có); kiểm schema/missing, cùng split/seed/preprocess.
- [ ] 4.2. Xuất **bảng chính đúng 4 mô hình PDF**: `GaussianNB-NP`, `LogReg-GD-NP`, `MLP`, `AdaBoost-SAMME-NP` với accuracy/F1 macro/precision/recall trên test, thời gian huấn luyện và dự đoán. Có thể thêm bảng train/validation/test hoặc lọc từ dataset hợp nhất.
- [ ] 4.3. **Bảng phụ bổ sung CNN** (mô hình thứ 5); ghi CNN là nâng cao, không làm mất bảng 4 phương pháp bắt buộc. Trình bày so sánh riêng MLP–CNN từ Phase 2.
- [ ] 4.4. **Biểu đồ so sánh** F1 macro hoặc accuracy của 4 mô hình và biểu đồ fit/predict time; ghi rõ **so sánh tốc độ khác phần cứng CPU/GPU không trực tiếp công bằng**.
- [ ] 4.5. Bảng **ưu / nhược điểm** của mỗi mô hình: chính xác, fit time, inference time, giải thích (prior/mean-var NB, hệ số LR, hidden layers MLP, weak-learners AdaBoost, conv features CNN).
- [ ] 4.6. Thảo luận cặp số dễ nhầm theo confusion matrix, chênh lệch train–validation–test, lý do hợp lý; không bịa kết quả.
- [ ] 4.7. Xuất `results/phase3_metrics.csv`, `results/comparison.csv`, hình, mô hình/siêu tham số; tạo **dàn ý phần thí nghiệm và kết luận báo cáo**.

**Definition of Done — Phase 3:** AdaBoost tự cài đặt và kiểm chứng được; đủ metric/CM 3 split; ≥1 hyperparameter figure; có bảng riêng 4 phương pháp bắt buộc + CNN bổ sung; đủ 4 tiêu chí so sánh (độ chính xác, train time, predict time, khả năng giải thích).

## 6. Checklist chung lặp lại ở TẤT CẢ notebook

| Hạng mục | Phase 1 | Phase 2 | Phase 3 |
|---|:---:|:---:|:---:|
| Nạp đúng MNIST 28×28, giữ test gốc | ☐ | ☐ | ☐ |
| Đọc split_indices.npz, kiểm hash/thứ tự, seed 42 | ☐ | ☐ | ☐ |
| Kiểm shape, nhãn 0–9 và dữ liệu sau tiền xử lý | ☐ | ☐ | ☐ |
| Không data leakage, tuning theo validation | ☐ | ☐ | ☐ |
| Cài đặt thuật toán theo phụ lục PDF | ☐ | ☐ | ☐ |
| Accuracy, precision_macro, recall_macro, f1_macro cho cả 3 split | ☐ | ☐ | ☐ |
| Confusion matrix 10×10 cho cả 3 split | ☐ | ☐ | ☐ |
| Đo `train_seconds`, `predict_seconds` | ☐ | ☐ | ☐ |
| ≥ 1 thí nghiệm siêu tham số với biểu đồ | ☐ | ☐ | ☐ |
| Phân tích mẫu sai, nhận xét dựa trên số liệu | ☐ | ☐ | ☐ |
| Đối chiếu sklearn **không dùng làm lõi** | ☐ | ☐* | ☐ |
| Lưu CSV đúng schema + hình + giải thích | ☐ | ☐ | ☐ |
| Tài liệu tham khảo, ghi nhận tái sử dụng/AI | ☐ | ☐ | ☐ |

*Phase 2 có thể đối chiếu MLP với `sklearn.neural_network.MLPClassifier` (baseline, không thay phần tự cài/khung được phép); CNN so sánh trực tiếp với MLP. PyTorch/TensorFlow chỉ dùng đúng phạm vi được PDF và giảng viên cho phép.*

## 7. Phân công và lịch triển khai tham khảo (PDF: nhóm 2–3 người, 4–5 tuần)

| Giai đoạn | Việc chính | Bàn giao |
|---|---|---|
| Tuần 1 | Thống nhất split/metric/schema; EDA; code NB và LR từ scratch | Phase 1 runnable + CSV/plots |
| Tuần 2 | Hoàn thiện Phase 1, rà soát leakage; xây MLP + kiểm backprop | Phase 1 đã kiểm, MLP baseline |
| Tuần 3 | CNN nâng cao, chọn hyperparams bằng validation; bàn giao Phase 2 | Phase 2 runnable + CSV/plots |
| Tuần 4 | Decision stump + SAMME tự cài; hyperparameter study, so sánh 4 mô hình + CNN | Phase 3 runnable + comparison |
| Tuần 5 (nếu có) | Kiểm tái lập, README, báo cáo 10–15 trang, slide 10–15 phút, phản biện | Bộ nộp cuối |

- [ ] Nhóm 2–3 thành viên tự **gán tên, đầu việc, tỷ lệ % đóng góp (tổng 100%)** trong bảng phân công thực tế; đừng tự điền tên/điểm/đóng góp giả định.
- [ ] Chia việc có thể theo module nhưng tất cả phải kiểm tra chéo độ đúng của metric/split và có khả năng giải thích công thức code do nhóm nộp.

## 8. Checklist sản phẩm nộp cuối kỳ giữa kỳ (theo PDF, không phải 4 file đang giao)

- [ ] Repository Git công khai hoặc `.zip`, **README rõ cách cài, chạy từng phase, data source, seed, phiên bản, thời gian chạy**, mã nguồn có cấu trúc.
- [ ] Notebook Jupyter tái tạo được kết quả chính; biểu đồ/tables xuất đầy đủ; không còn `NotImplementedError`, placeholder hoặc kết quả không tái hiện được.
- [ ] **Báo cáo PDF 10–15 trang** có: giới thiệu, dữ liệu, phương pháp (**công thức + pseudo-code**), thí nghiệm, kết quả, thảo luận, kết luận, tài liệu tham khảo, **phân công công việc** + **tỷ lệ đóng góp**, khai báo AI nếu dùng.
- [ ] **Slide thuyết trình 10–15 phút**; chuẩn bị giải thích sự khác biệt giữa phương pháp tự cài và baseline sklearn, ý nghĩa macro metrics, data leakage, confusion matrix.
- [ ] Có thể chọn thêm Fashion-MNIST/EMNIST, augmentation, giao diện demo nếu dư thời gian; **không phải điều kiện bắt buộc của PDF**. CNN đã nằm ở Phase 2 theo yêu cầu.
- [ ] Tuân thủ quy định nguồn trích dẫn: sao chép thiếu trích dẫn bị xử phạt nặng; nộp trễ bị trừ **10%/ngày**, tối đa **3 ngày**; quá hạn hơn không nhận (theo phụ lục).

## 9. Đối chiếu rubric giữa kỳ khi tự kiểm

| Tiêu chí theo PDF | Điểm | Nơi phải thể hiện |
|---|---:|---|
| Chuẩn bị dữ liệu, tiền xử lý, EDA | 1.0 | Phase 1 + ô kiểm nhất quán Phase 2/3 |
| Cài đúng công thức, đối chiếu sklearn, đủ 4 phương pháp | 3.0 | NB/LR (P1), MLP (P2), Ensemble (P3) |
| Đủ metric, phân tích hyperparameter và biểu đồ | 2.0 | Tất cả phase (mỗi phase ≥1 plot) |
| So sánh và thảo luận có cơ sở | 1.5 | P1/P2 nội bộ, P3 tổng hợp |
| Báo cáo khoa học, hình/công thức rõ | 1.0 | Báo cáo + hình tạo ở phase |
| Trình bày và phản biện | 1.0 | Slide + ô câu hỏi kết luận |
| Điểm cộng (nâng cao/demo/sáng tạo) | 0.5–1.5 | CNN P2; tùy chọn khác |

**Lưu ý:** Đây là **kế hoạch và notebook skeleton** để phân việc. Không có code mô hình, kết quả hay biểu đồ thực nghiệm được tạo ở bước này; tất cả mục TODO sẽ được thực hiện ở các bước lập trình sau.

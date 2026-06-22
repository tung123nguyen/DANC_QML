# CHƯƠNG 3 — PHƯƠNG PHÁP ĐỀ XUẤT (BẢN CHI TIẾT)

> Chương này trình bày đầy đủ phương pháp luận của luận án: từ phát biểu bài toán hình thức, toàn bộ quy trình dữ liệu, ba đóng góp phương pháp (trainable feature map, multi-observable readout, hybrid architecture), thuật toán huấn luyện, đến phân tích độ phức tạp và giao thức đánh giá. Mọi mô tả được neo trực tiếp vào mã nguồn hiện thực (`src/`).

---

## 3.0. Ký hiệu và quy ước

| Ký hiệu | Ý nghĩa |
|---|---|
| `n` | số qubit (trong luận án `n = 4`) |
| `d` | số đặc trưng đầu vào sau khi chọn lọc (`d = n = 4`) |
| `L`, `depth` | số lớp ansatz / số lần nạp lại dữ liệu (`L = 4`) |
| `N` | số mẫu huấn luyện **mỗi lớp** (100 / 250 / 500) |
| `x ∈ ℝ^d` | vectơ đặc trưng một luồng mạng |
| `y ∈ {0,1}` | nhãn: 0 = BENIGN, 1 = ATTACK |
| `φ(·)` | hàm chuẩn hóa góc (angle_clip), ánh xạ về `[−π, π]` |
| `θ` | toàn bộ vectơ tham số học được (mạch + readout head) |
| `⟨O⟩` | giá trị kỳ vọng của quan sát được `O` |
| `σ(·)` | hàm sigmoid logistic |
| `B` | kích thước lô (batch), `B = 32` |

Quy ước phân loại: lớp dương (positive) là **ATTACK** (`y = 1`); mọi chỉ số precision/recall mặc định tính cho lớp này.

---

## 3.1. Tổng quan hệ thống

Hệ thống được tổ chức thành một **pipeline điều khiển bằng cấu hình** (config-driven): mỗi thí nghiệm là một file YAML mô tả đầy đủ (dữ liệu, mô hình, huấn luyện, đánh giá), và một điểm vào duy nhất (`src/experiment.py`) đọc config, dựng dữ liệu, huấn luyện, đánh giá, rồi ghi kết quả kèm siêu dữ liệu tái lập (git commit, thời gian, cấu hình đã dùng).

```
                ┌──────────────────────── config.yaml ────────────────────────┐
                │ experiment{name,seed} · data{scenario,N,test,n_features}     │
                │ model{type, ...}      · training{epochs,lr,batch} · eval{}   │
                └───────────────────────────────┬──────────────────────────────┘
                                                 ▼
   CSV thô  ──►  loader  ──►  sampling (tách train/test rời rạc)  ──►  preprocessing
 (CIC-IDS2017)  (làm sạch,    (cân bằng lớp, không trùng hàng)        (blacklist → scale
                cache parquet)                                          → MI feature select)
                                                 │
                                  X_train, y_train, X_test, y_test  (X ∈ ℝ^{·×4})
                                                 │
                    ┌────────────────────────────┴───────────────────────────┐
                    ▼                                                          ▼
       ┌──────────────────────────┐                          ┌──────────────────────────────┐
       │ NHÁNH CỔ ĐIỂN            │                          │ NHÁNH LƯỢNG TỬ (QNN)         │
       │ build_classical()        │                          │ build_qnn(): mã hóa × ansatz  │
       │  → fit() → predict_proba │                          │  × readout + hybrid readout head  │
       │ LR/SVM/RF/MLP/mlp_tiny   │                          │  → train_qnn() (Adam, BCE)    │
       └────────────┬─────────────┘                          └───────────────┬──────────────┘
                    │              (y_pred, y_prob)                           │
                    └────────────────────────┬───────────────────────────────┘
                                              ▼
              compute_train_test_metrics(): F1, precision/recall(attack),
              FNR, FPR, ROC-AUC, confusion matrix, khoảng cách train–test
                                              ▼
                       results/<timestamp>_<name>_seed<k>/ {metrics,history,config,meta}.json
```

**Nguyên tắc thiết kế cốt lõi — giao diện thống nhất.** Cả nhánh cổ điển và nhánh lượng tử đều trả về cặp `(nhãn dự đoán, xác suất)` với cùng ngữ nghĩa. Nhờ đó **một** hàm tính chỉ số duy nhất (`compute_train_test_metrics`) áp dụng cho mọi mô hình, loại bỏ mọi khác biệt đánh giá có thể gây thiên lệch khi so sánh. Đây là điều kiện cần cho một benchmark trung thực.

**Tính idempotent.** Trước khi chạy, `already_done()` kiểm tra xem cấu hình (theo `name` + `seed`) đã có `metrics.json` chưa; nếu rồi thì bỏ qua. Kết quả smoke-test (tiền tố `SMOKE_`, dữ liệu rút gọn) bị loại khỏi kiểm tra này để không "giả" là đã hoàn thành.

---

## 3.2. Phát biểu bài toán

Cho một tập huấn luyện `D_train = {(x_i, y_i)}_{i=1}^{2N}` cân bằng (N mẫu/lớp) và một tập kiểm thử `D_test` **rời rạc hoàn toàn** ở mức hàng. Mục tiêu là học một bộ phân loại xác suất

$$f_\theta : \mathbb{R}^d \rightarrow [0,1], \qquad f_\theta(x) = P(\text{ATTACK} \mid x),$$

bằng cách cực tiểu hóa rủi ro thực nghiệm với hàm mất mát entropy chéo nhị phân, sao cho `f_θ` **generalization** sang `D_test`. Dự đoán nhãn cứng dùng ngưỡng 0,5: `ŷ = 1[f_θ(x) > 0.5]`.

Hai chế độ đánh giá quy định bản chất của `D_test`:

- **Trong phân phối (S1):** `D_train` và `D_test` cùng phân phối nguồn (Friday DDoS) — đo năng lực học.
- **Dưới distribution shift (S3):** `D_train` từ Wednesday DoS, `D_test` từ Friday DDoS — đo năng lực **generalization "tính tấn công"** chứ không phải ghi nhớ dấu hiệu.

Bài toán đặt trong **chế độ dữ liệu hạn chế** (`N ∈ {100, 250, 500}`), mô phỏng thực tế khan hiếm nhãn của IDS.

---

## 3.3. Quy trình dữ liệu

### 3.3.1. Nạp và làm sạch (`src/data/loader.py`)

CIC-IDS2017 lưu mỗi ngày trong một file CSV (~80 cột đặc trưng do CICFlowMeter trích xuất). Quy trình làm sạch một lần, rồi cache parquet:

1. **Chuẩn hóa tên cột:** strip khoảng trắng đầu/cuối (`' Label'` → `'Label'`) — sửa lỗi nổi tiếng của bộ dữ liệu.
2. **Xử lý giá trị vô hạn:** thay `±inf` (sinh ra từ các tỉ số như `Flow Bytes/s` khi thời lượng = 0) bằng `NaN`, sau đó **loại bỏ mọi hàng chứa NaN**. Đây là các giá trị không sửa được nên phải bỏ.
3. **Gán nhãn nhị phân:** `binary_label = 0` nếu nhãn (chuẩn hóa hoa/thường, strip) là `BENIGN`, ngược lại `= 1`. Mọi biến thể tấn công (DoS Hulk, GoldenEye, slowloris, …, DDoS) đều gộp thành lớp 1.

Số liệu sau làm sạch:

| File | Tổng luồng | Phân bố |
|---|---|---|
| Friday DDoS | 225.711 | DDoS 128.025 · BENIGN 97.686 |
| Wednesday DoS | 691.406 | BENIGN 439.683 · DoS Hulk 230.124 · GoldenEye 10.293 · slowloris 5.796 · Slowhttptest 5.499 · Heartbleed 11 |

### 3.3.2. Danh sách đen đặc trưng (`src/data/feature_blacklist.py`)

Trước mọi xử lý số học, các cột sau **bị loại vĩnh viễn**. Đây là quyết định phương pháp quyết định tính trung thực của kịch bản chéo tấn công S3:

| Nhóm | Cột | Lý do loại |
|---|---|---|
| Định danh | Flow ID, Source/Destination IP | không phải đặc trưng hành vi |
| Thời gian | Timestamp | rò rỉ thời gian, không tồn tại lúc triển khai |
| Cổng | Source/Destination Port | DDoS nhắm cổng 80, PortScan chạm nhiều cổng → cho mô hình "ghi nhớ" loại tấn công |
| Giao thức | Protocol | từng giao thức gắn với loại tấn công cụ thể (ICMP/TCP) → lối tắt danh tính tấn công |
| Nhãn | Label, binary_label | là mục tiêu |

Nguyên tắc: loại bỏ mọi đặc trưng **gắn với danh tính** (identity) của tấn công, giữ lại đặc trưng **hành vi thống kê** của luồng (số byte, thời lượng, độ dài gói, IAT…). Nếu không, một mô hình có thể đạt điểm cao trên S1 bằng cách học "cổng 80 = DDoS" và sụp đổ hoàn toàn khi loại tấn công đổi.

### 3.3.3. Lấy mẫu rời rạc train/test (`src/data/sampling.py`)

Điểm tinh tế về ngữ nghĩa: "huấn luyện trên N mẫu/lớp" phải nghĩa là **đúng N hàng/lớp trong tập train**, không phải N hàng trước khi chia 70/30. Hàm `sample_train_test` đảm bảo điều này:

```
Thuật toán 1: Lấy mẫu rời rạc cân bằng (sample_train_test)
Vào: DataFrame df, train_per_class N_tr, test_per_class N_te, seed
với mỗi lớp c ∈ {0, 1}:
    group ← các hàng của df có nhãn c
    pool  ← group.sample(N_tr + N_te, random_state=seed)   # lấy không lặp
    train_c ← pool[0 : N_tr]                                # N_tr hàng đầu
    test_c  ← pool[N_tr : N_tr + N_te]                      # N_te hàng tiếp, KHÔNG trùng
train_df ← shuffle(concat(train_0, train_1), seed)
test_df  ← shuffle(concat(test_0, test_1), seed + 1)        # seed khác để xáo độc lập
Ra: (train_df, test_df) — bảo đảm không trùng hàng
```

Với S3 (chéo nguồn), train và test đến từ hai file khác nhau nên dùng `sample_balanced` riêng cho mỗi nguồn; test dùng offset `seed + 1000` để tuyệt đối không trùng chỉ số hàng với bất kỳ tập train nào dùng lại trong các kịch bản khác.

**Test cố định 500 mẫu/lớp** cho mọi cấu hình → mọi con số test-F1 so sánh trực tiếp được khi `N` thay đổi (chỉ kích thước *train* biến thiên).

### 3.3.4. Chuẩn hóa, feature selection, chống rò rỉ (`src/data/preprocessing.py`)

Sau khi đã tách train/test, hàm `scale_and_select` thực hiện:

```
Thuật toán 2: Tiền xử lý chống rò rỉ (scale_and_select)
Vào: train_df, test_df, n_features k
1. feature_cols ← (cột train ∩ cột test) \ blacklist        # giao để xử lý chênh schema S3
2. X_train, X_test ← ma trận giá trị; y_train, y_test ← nhãn
3. scaler ← StandardScaler().fit(X_train)                   # FIT CHỈ TRÊN TRAIN
   X_train ← scaler.transform(X_train);  X_test ← scaler.transform(X_test)
4. selector ← SelectKBest(mutual_info_classif, k).fit(X_train, y_train)  # FIT CHỈ TRÊN TRAIN
   X_train ← selector.transform(X_train); X_test ← selector.transform(X_test)
Ra: X_train, X_test ∈ ℝ^{·×k}, tên k đặc trưng đã chọn
```

**Quy tắc bất khả xâm phạm:** `scaler.fit()` và `selector.fit()` **chỉ nhìn thấy dữ liệu train**. Mọi tham số chuẩn hóa (trung bình, độ lệch chuẩn) và lựa feature selection (theo mutual information giữa đặc trưng và nhãn) đều ước lượng từ train, rồi *áp* lên test. Vi phạm quy tắc này (ví dụ fit scaler trên toàn bộ dữ liệu) sẽ làm rò rỉ thông tin test vào huấn luyện và thổi phồng kết quả.

**Vì sao `k = 4`:** khớp với số qubit `n = 4`, để angle encoding gán đúng một đặc trưng cho một qubit. Đồng thời giữ mạch đủ nhỏ để mô phỏng trên CPU. Mô hình cổ điển cũng chỉ thấy đúng 4 đặc trưng này → sân chơi công bằng về *đầu vào*.

---

## 3.4. Data encoding lượng tử

### 3.4.1. Chuẩn hóa góc (angle_clip) (`_scale_to_angle`)

Đặc trưng sau StandardScaler có phân phối ~chuẩn (trung bình 0, độ lệch chuẩn 1), khoảng `[−3, 3]` với đuôi ngoại lai. Đưa thẳng vào cổng quay `RY` là không an toàn do tính tuần hoàn `2π` (một giá trị 30 cuộn về ~0, làm sụp các đầu vào khác biệt). Phép chuẩn hóa:

$$\varphi(x) = \mathrm{clip}(x, -c, +c)\cdot \frac{\pi}{c}, \qquad c = \texttt{angle\_clip} = 3.0$$

- Cắt về `±c·σ` rồi co tuyến tính để `±c·σ ↦ ±π`, giữ mọi góc trong `[−π, π]` (vùng đơn điệu của `RY`, không cuộn vòng).
- `c = 3` giữ ~99,7% khối lượng Gaussian trong vùng tuyến tính.
- Đặt `c = None/0` tắt phép biến đổi (tái lập mã hóa thô — để đối chứng).

Vì `x` là dữ liệu (không bao giờ bị vi phân), phép này dùng NumPy thuần, tách bạch khỏi đồ thị tính gradient.

### 3.4.2. Hai chế độ nạp dữ liệu (`build_qnode`)

Cho `U_enc(x)` là khối angle encoding và `U_ℓ(θ_ℓ)` là lớp ansatz thứ `ℓ`:

**Angle encoding một lần (`angle`):**

$$|\psi(x,\theta)\rangle = U_L(\theta_L)\cdots U_1(\theta_1)\, U_{\text{enc}}(x)\,|0\rangle^{\otimes n}$$

```
mã hóa x một lần;  rồi áp L lớp ansatz liên tiếp
```

**Nạp lại dữ liệu (`reuploading`):**

$$|\psi(x,\theta)\rangle = \prod_{\ell=L}^{1} \big[ U_\ell(\theta_\ell)\, U_{\text{enc}}(x) \big]\,|0\rangle^{\otimes n}$$

```
lặp L lần:  ( mã hóa x  →  một lớp ansatz )
```

**Cơ sở lý thuyết (Fourier).** Schuld, Sweke & Meyer (2021) chứng minh một QNN với data re-uploading biểu diễn được một **chuỗi Fourier cắt cụt** theo `x`:

$$f_\theta(x) = \sum_{\omega \in \Omega} c_\omega(\theta)\, e^{i \omega x},$$

trong đó **tập tần số khả dụng** `Ω` mở rộng theo số lần nạp lại `L` (do phổ riêng của Hamiltonian mã hóa cộng dồn). Mã hóa một lần chỉ cho `Ω` nhỏ (hàm "tần số thấp"); nạp lại `L` lần mở rộng `Ω`, cho phép xấp xỉ ranh giới quyết định phức tạp hơn. Đây là lý do lý thuyết để kỳ vọng `reuploading` > `angle` — kiểm chứng thực nghiệm ở Chương 4.

### 3.4.3. Angle encoding cụ thể (`_angle_embed`)

```python
x' = φ(x)                          # chuẩn hóa angle_clip
if trainable_encoding:  x' = a * x' + b     # §3.5 — affine trainability
AngleEmbedding(x', wires=0..n-1, rotation="Y")   # áp RY(x'_i) lên qubit i
```

`AngleEmbedding` của PennyLane áp `RY(x'_i)` lên qubit `i` — mỗi đặc trưng điều khiển góc nghiêng của đúng một qubit quanh trục `Y` của Bloch sphere.

---

## 3.5. Đề xuất 1 — Trainable feature map

### 3.5.1. Động cơ

Trong angle encoding cố định, ánh xạ `x ↦ RY(φ(x))` là **tĩnh**: hình dạng không gian đặc trưng do người thiết kế ấn định. Theo lý thuyết Fourier, điều này tương đương **cố định các tần số** `ω` của khai triển — mô hình chỉ có thể điều chỉnh các hệ số `c_ω` (qua ansatz) chứ không chọn được tần số. Đề xuất của chúng tôi cho phép **chính feature map được học**.

### 3.5.2. Hình thức

Thêm một biến đổi affine **theo từng qubit** trước cổng quay:

$$\text{qubit } i:\quad RY\!\big(a_i\,\varphi(x_i) + b_i\big), \qquad a, b \in \mathbb{R}^n \text{ học được.}$$

- `a_i` (**scale**): độ "nhạy" của qubit `i` với đặc trưng — tương đương **học tần số** `ω_i` của khai triển Fourier (giãn/nén trục dữ liệu).
- `b_i` (**bias / dịch pha**): dịch điểm làm việc trên Bloch sphere — tương đương **pha** của thành phần Fourier.

### 3.5.3. Khởi tạo có chủ đích (then chốt cho so sánh công bằng)

$$a_i \leftarrow 1.0, \qquad b_i \leftarrow 0.0 \quad \text{lúc khởi tạo.}$$

Khi `a = 1, b = 0`, trainable feature map **trùng khít** với angle encoding cố định `RY(φ(x))`. Hệ quả:

1. Tại điểm khởi tạo, mô hình `tenc` **đồng nhất** với mô hình baseline — nên mọi chênh lệch hiệu năng quan sát được là do **học**, không phải do khởi tạo khác.
2. Trainable feature map chỉ có thể **giữ nguyên hoặc cải thiện** so với baseline (về mặt khả năng biểu diễn tại khởi điểm).

Khối `[a | b]` (gồm `n` phần tử `a` rồi `n` phần tử `b`) được đặt ở **đầu** vectơ tham số mạch; trình huấn luyện khởi tạo riêng hai khối này (xem §3.9). Với `reuploading`, cùng khối `[a | b]` được **dùng lại ở mọi lần nạp** (chia sẻ tham số) → chỉ tốn thêm `2n` tham số bất kể `L`.

---

## 3.6. Các ansatz biến phân (`_apply_ansatz`)

Ansatz là "phần học được" của mạch, quyết định sự đánh đổi giữa **expressivity** và **trainability**. Ba lựa chọn:

### 3.6.1. Basic Entangler (`basic_entangler`)

Mỗi lớp: một cổng quay đơn trên mỗi qubit, theo sau một **vòng CNOT** (ring of CNOTs). Hiện thực bằng `qml.BasicEntanglerLayers` với tham số dạng `(1, n)`.

```
mỗi lớp:  R(θ_q) trên qubit q = 0..n-1 ;  CNOT(0→1), CNOT(1→2), …, CNOT(n-1→0)
tham số/lớp:  n
```

Nhẹ (`n` tham số/lớp), ít rủi ro barren plateau — phù hợp dữ liệu nhỏ.

### 3.6.2. Strongly Entangling (`strongly_entangling`)

Mỗi lớp: ba cổng quay (góc Euler) trên mỗi qubit + các CNOT có **khoảng cách (range) biến thiên** giữa các lớp. Hiện thực bằng `qml.StronglyEntanglingLayers`, tham số `(1, n, 3)`.

```
mỗi lớp:  Rot(α_q, β_q, γ_q) trên qubit q ;  CNOT khoảng cách thay đổi theo lớp
tham số/lớp:  3n
```

Biểu cảm hơn (nhiều bậc tự do, entanglement tầm xa) nhưng tốn tham số (`3n`/lớp) và dễ gradient nhỏ hơn.

### 3.6.3. Rotation-only (`rotation_only`) — đối chứng

Ba cổng quay `RX, RY, RZ` trên mỗi qubit, **không có CNOT**:

```
mỗi lớp:  RX(θ_{3q}) RY(θ_{3q+1}) RZ(θ_{3q+2}) trên qubit q ;  (không entanglement)
tham số/lớp:  3n
```

Cùng số tham số với strongly_entangling nhưng **không entanglement** → dùng để **cô lập đóng góp của entanglement** (so sánh có/không CNOT ở cùng ngân sách tham số).

### 3.6.4. Số tham số mỗi lớp (`_params_per_layer`)

| Ansatz | Tham số/lớp `ppl` | Entanglement |
|---|---|---|
| `basic_entangler` | `n` | có (vòng CNOT) |
| `strongly_entangling` | `3n` | có (CNOT tầm biến thiên) |
| `rotation_only` | `3n` | không |

Tổng tham số mạch (chưa kể trainable feature map) = `L · ppl`.

---

## 3.7. Đề xuất 2 — Multi-observable readout (`_measure`, `_observables`)

Readout head quyết định **lượng thông tin** trích từ trạng thái `|ψ(x,θ)⟩`. Ba phương án, đều **không làm tăng độ sâu mạch**:

### 3.7.1. `z` — đơn quan sát (baseline)

Đo Pauli-Z trên từng qubit:

$$z = \big(\langle Z_0\rangle, \langle Z_1\rangle, \ldots, \langle Z_{n-1}\rangle\big) \in [-1,1]^n, \qquad n_{\text{obs}} = n = 4.$$

Tiêu chuẩn, rẻ, nhưng bỏ lỡ thông tin **tương quan giữa các qubit**.

### 3.7.2. `z+zz` — đa quan sát (đề xuất chính)

Bổ sung **toàn bộ tương quan hai qubit**:

$$z = \big(\underbrace{\langle Z_i\rangle}_{n}, \underbrace{\langle Z_iZ_j\rangle}_{i<j}\big), \qquad n_{\text{obs}} = n + \binom{n}{2} = 4 + 6 = 10.$$

Các số hạng `⟨Z_iZ_j⟩` đo **độ đồng pha** giữa cặp qubit — đại lượng **nhạy với entanglement** mà phép đo đơn qubit không thấy. Chúng làm giàu biểu diễn cho readout head cổ điển mà **vẫn dùng cùng một trạng thái** (chỉ đo thêm quan sát).

### 3.7.3. `probs` — phân bố cơ sở

Đo trực tiếp phân bố xác suất trên cơ sở tính toán của `readout_wires = 2` qubit đầu:

$$z = \big(P(00), P(01), P(10), P(11)\big), \qquad n_{\text{obs}} = 2^{\texttt{readout\_wires}} = 4, \quad \textstyle\sum_b z_b = 1.$$

Đây là readout giàu theo *chiều sâu* (phân bố đầy đủ trên 2 qubit) nhưng giới hạn *chiều rộng* (chỉ 2 qubit). Ràng buộc tổng = 1 khiến thực chất chỉ có 3 bậc tự do.

### 3.7.4. So sánh

| Readout | `n_obs` | Mở rộng theo | Thông tin |
|---|---|---|---|
| `z` | 4 | — | kỳ vọng đơn qubit |
| `z+zz` | 10 | **chiều rộng** (mọi qubit) | + tương quan cặp |
| `probs` | 4 | **chiều sâu** (2 qubit) | phân bố đầy đủ trên 2 qubit |

`n_obs` chính là **chiều đầu vào của linear head** (§3.8).

---

## 3.8. Đề xuất 3 — Hybrid architecture và linear head

### 3.8.1. Linear head

Vectơ kết quả đo `z ∈ ℝ^{n_obs}` được biến đổi thành một logit rồi xác suất bởi một **lớp tuyến tính cổ điển**:

$$\text{logit} = w \cdot z + b_{\text{head}}, \qquad p = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}} \in [0,1],$$

với `w ∈ ℝ^{n_obs}`, `b_head ∈ ℝ`. Readout head đóng vai trò "lớp ra" cổ điển trên đặc trưng lượng tử.

### 3.8.2. Vectơ tham số phẳng duy nhất (`build_qnn`)

Toàn bộ mô hình lai được tham số hóa bằng **một** vectơ phẳng `θ`, theo bố cục:

```
θ = [  a (n)  |  b (n)  |  ansatz (L·ppl)  |  w (n_obs)  |  b_head (1)  ]
       └──────────── tham số mạch  n_quantum ───────────┘  └─ hybrid readout head ─┘
```

- Khối `[a | b]` (mỗi khối `n` phần tử) **chỉ tồn tại** khi bật trainable feature map.
- `n_quantum = (2n nếu tenc) + L·ppl` là phần đưa vào mạch.
- `n_head = n_obs + 1` là phần readout head.
- `n_params = n_quantum + n_head`.

Hàm `build_qnn` trả về dict mô tả đầy đủ: `circuit`, `n_quantum_params`, `n_scale_params` (khối `a`), `n_encbias_params` (khối `b`), `n_obs`, `n_head_params`, `n_params`.

### 3.8.3. Lan truyền ngược end-to-end

Vì `θ` là một vectơ duy nhất và mọi phép toán (mạch → đo → linear head → sigmoid → BCE) đều khả vi, **gradient chảy xuyên suốt**: từ mất mát, qua readout head `(w, b_head)`, qua các quan sát được, vào tận các góc cổng lượng tử `(a, b, ansatz)` — tất cả bằng **vi phân tự động** (autograd) của PennyLane. Không có ranh giới "đóng băng" giữa phần lượng tử và phần cổ điển.

### 3.8.4. Forward pass vector hóa (`_circuit_probs`)

Một điểm hiệu năng quan trọng: nhờ **parameter broadcasting** của PennyLane, cả lô `B` mẫu được đánh giá trong **một** lần gọi mạch (không vòng lặp Python):

```
Thuật toán 3: Forward pass cho một lô X ∈ ℝ^{B×n}  (_circuit_probs)
Vào: θ, circuit, X, n_quantum, n_obs
1. q_params ← θ[0 : n_quantum]                 # phần mạch
   w        ← θ[n_quantum : n_quantum + n_obs] # trọng số readout head
   b_head   ← θ[n_quantum + n_obs]             # bias readout head
2. out ← circuit(q_params, X)                  # broadcast: chạy B mẫu một lần
3. nếu out là list/tuple (readout z / z+zz):
       z ← stack(out)        # (n_obs, B)
   ngược lại (readout probs):
       z ← out.T             # (n_obs, B)
4. logit ← w · z + b_head                      # (B,)
5. p ← σ(logit)                                # (B,)
Ra: p ∈ [0,1]^B
```

Toàn bộ viết bằng `pennylane.numpy` (pnp) để gradient chảy tới **cả** tham số mạch lẫn readout head.

---

## 3.9. Huấn luyện (`src/training/qnn_trainer.py`)

### 3.9.1. Hàm mất mát

Entropy chéo nhị phân trung bình trên lô, kẹp số học chống `log(0)`:

$$\mathcal{L}(\theta) = \frac{1}{B}\sum_{k=1}^{B}\Big[-y_k\log p_k - (1-y_k)\log(1-p_k)\Big], \quad p_k \in [10^{-7},\, 1-10^{-7}].$$

### 3.9.2. Khởi tạo tham số

```
init ← Uniform(−0.1, 0.1)^{n_params}        # khởi tạo nhỏ, gần 0
nếu tenc:  init[khối a] ← 1.0                # scale a = 1
           init[khối b] ← 0.0                # bias b = 0  (⇒ trùng mã hóa cố định)
init[cuối cùng] ← 0.0                        # b_head = 0  ⇒ logit ≈ 0 ⇒ p ≈ 0.5 lúc đầu
θ ← pnp.array(init, requires_grad=True)
```

Khởi tạo `b_head = 0` đảm bảo xác suất xuất phát ~0,5 (không thiên lệch lớp), tránh bão hòa sigmoid; khởi tạo nhỏ `±0.1` cho phần ansatz để tránh vùng gradient phẳng.

### 3.9.3. Early stopping với khôi phục tham số tốt nhất

Trước vòng lặp, cắt một lát kiểm định **phân tầng** (giữ tỉ lệ lớp) `val_fraction = 0.2` ra khỏi train; huấn luyện trên phần còn lại. Mỗi epoch theo dõi BCE kiểm định; nếu không cải thiện quá `patience = 15` epoch thì dừng và **khôi phục tham số có val-BCE tốt nhất** — không bao giờ trả về mô hình đã sụp đổ. Tập test **không bao giờ** bị động tới trong quá trình này.

### 3.9.4. Thuật toán huấn luyện đầy đủ

```
Thuật toán 4: train_qnn (Adam + mini-batch + early stopping)
Vào: qnn, X_train, y_train, epochs=80, lr=0.05, batch=32,
     early_stopping=True, val_fraction=0.2, patience=15, seed
khởi tạo θ (§3.9.2);  optimizer ← Adam(stepsize=lr)
(X_fit, y_fit), (X_val, y_val) ← stratified_split(X_train, y_train, val_fraction, seed)
best_val ← +∞;  best_θ ← None;  no_improve ← 0
với epoch = 0 … epochs−1:
    xáo trộn (X_fit, y_fit)
    với mỗi lô (xb, yb):
        obj(θ) ← BCE(θ, circuit, xb, yb, n_quantum, n_obs)
        nếu log_gradients:
            grad, loss ← optimizer.compute_grad(obj, θ)      # tính grad MỘT lần
            ghi ‖grad‖, var(grad), mean|grad|                # thống kê trainability
            θ ← optimizer.apply_grad(grad, θ)                # tái dùng grad cho cập nhật
        ngược lại:
            θ, loss ← optimizer.step_and_cost(obj, θ)
        cộng dồn loss
    nếu epoch ∈ {bội số của 10} ∪ {cuối}:  tính train_acc (forward toàn tập)
    nếu early_stopping:
        val_loss ← BCE(θ, …, X_val, y_val)
        nếu val_loss < best_val − min_delta:  best_val ← val_loss; best_θ ← copy(θ); no_improve ← 0
        ngược lại:  no_improve ← no_improve + 1
        nếu no_improve ≥ patience:  dừng (early stop)
nếu early_stopping và best_θ ≠ None:  θ ← best_θ   # khôi phục tốt nhất
Ra: θ, lịch sử (loss/acc/val/grad mỗi epoch), thời gian, stopped_epoch, best_val_loss
```

**Chi tiết tối ưu gradient:** khi bật ghi gradient, ta gọi `compute_grad` để lấy gradient **một lần**, dùng nó cho *cả* thống kê *và* cập nhật Adam (`apply_grad`) — tránh đánh giá mạch dư thừa lần hai (vốn xảy ra nếu dùng `step_and_cost` rồi tính grad riêng).

### 3.9.5. Siêu tham số huấn luyện (sweep chính)

`epochs = 80`, `learning_rate = 0.05` (Adam), `batch_size = 32`, `angle_clip = 3.0`, `depth = 4`, `n_qubits = 4`, thiết bị `lightning.qubit`. Ghi gradient **tắt** trong sweep chính (chỉ so F1/accuracy), **bật** cho tập con chẩn đoán `qnn_diag`.

### 3.9.6. Dự đoán (`qnn_predict`)

```
p ← _circuit_probs(θ, circuit, X, n_quantum, n_obs)
ŷ ← 1[p > 0.5]
Ra: (ŷ, p)
```

---

## 3.10. Phân tích trainability (gradient)

Cho tập con chẩn đoán, mỗi epoch ghi: **chuẩn gradient** `‖g‖`, **phương sai** `Var(g)`, **trung bình trị tuyệt đối** `mean|g|`, đều lấy **trung bình trên tất cả các lô** của epoch (tín hiệu mượt hơn so với chỉ lô cuối). Mục đích:

- Phát hiện **barren plateau**: nếu `Var(g)` triệt tiêu theo số qubit/độ sâu → vùng gradient phẳng, huấn luyện bất khả thi (McClean và cộng sự, 2018; Holmes và cộng sự, 2022).
- So sánh **expressivity vs trainability** giữa các ansatz/mã hóa (ansatz biểu cảm hơn thường có gradient nhỏ hơn).

---

## 3.11. Phân tích độ phức tạp tham số

### 3.11.1. Công thức (`build_qnn`, `_n_observables`)

$$
\begin{aligned}
n_{\text{obs}}(\texttt{z}) &= n, \\
n_{\text{obs}}(\texttt{z+zz}) &= n + \tbinom{n}{2}, \\
n_{\text{obs}}(\texttt{probs}) &= 2^{\texttt{readout\_wires}}, \\[2pt]
n_{\text{quantum}} &= \underbrace{2n \cdot \mathbb{1}[\text{tenc}]}_{\text{trainable feature map}} + \underbrace{L \cdot \text{ppl}}_{\text{ansatz}}, \\
n_{\text{params}} &= n_{\text{quantum}} + \underbrace{(n_{\text{obs}} + 1)}_{\text{readout head}}.
\end{aligned}
$$

### 3.11.2. Bảng tham số (n = 4, L = depth = 4)

| Mô hình | `n_quantum` | `n_head` | **Tổng** |
|---|---|---|---|
| q2 = angle + basic_entangler, `z` | 16 | 5 | **21** |
| q2 + `z+zz` | 16 | 11 | **27** |
| q2 + `probs` | 16 | 5 | **21** |
| q2 + trainable feature map (`tenc`) | 24 | 5 | **29** |
| q3 = angle + strongly_entangling, `z` | 48 | 5 | **53** |
| q4 = reuploading + basic_entangler, `z` | 16 | 5 | **21** |

QNN nằm gọn trong dải **21–53 tham số** — cực kỳ tiết kiệm so với mô hình cổ điển (xem §3.12).

---

## 3.12. Baseline cổ điển (`src/models/classical.py`)

Năm mô hình, dùng siêu tham số mặc định hợp lý (không tinh chỉnh lưới) để **so sánh dưới cùng điều kiện**:

| Mô hình | Cấu hình | ~Số tham số | Vai trò |
|---|---|---|---|
| `lr` | LogisticRegression(C=1, max_iter=1000) | 5 | ranh giới tuyến tính |
| `svm` | SVC(rbf, C=1, gamma=scale, probability=True) | (phi tham số) | nhân Hilbert — đối trọng khái niệm với QML |
| `rf` | RandomForestClassifier(100 cây) | ~2750 nút | khớp phi tuyến mạnh (xu hướng ghi nhớ) |
| `mlp` | MLPClassifier(hidden=(32,16)) | ~705 | mạng nơ-ron công suất lớn |
| **`mlp_tiny`** | MLPClassifier(hidden=(4,)) | **25** | **khớp tham số với QNN** |

### 3.12.1. Baseline khớp tham số (`mlp_tiny`) — đối chứng dung lượng

So sánh QNN (21 tham số) với MLP lớn (705) hay RF (~2750) là **không công bằng về dung lượng**. Để cô lập đóng góp của **kiểu biểu diễn** (lượng tử vs cổ điển) khỏi đóng góp của **dung lượng thô**, ta thêm `mlp_tiny`: một MLP **một lớp ẩn 4 nơ-ron**. Với `d = 4` đặc trưng vào và 1 nơ-ron ra, tổng tham số là

$$(d + 2)\,h + 1 = 6h + 1 \Big|_{h=4} = 25,$$

(16 trọng số vào–ẩn + 4 bias ẩn + 4 trọng số ẩn–ra + 1 bias ra), nằm đúng trong dải tham số của QNN. Cấu trúc **4 đặc trưng → 4 nơ-ron ẩn → 1 đầu ra** phản chiếu **4 qubit → mạch → 1 linear head**, cho một so sánh khớp dung lượng sạch sẽ. Như Chương 4 chỉ ra, ở cùng ngân sách tham số QNN **vượt** `mlp_tiny` trên cả hai kịch bản — bằng chứng cho hiệu quả tham số nội tại của biểu diễn lượng tử.

Lưu ý hiện thực: cả `mlp` và `mlp_tiny` đặt `early_stopping=False`. Ở chế độ dữ liệu nhỏ, `early_stopping=True` của sklearn cắt thêm một lát kiểm định khỏi fold huấn luyện vốn đã ít và dừng trước khi mạng học được gì, làm MLP sụp về toàn-benign (F1 = 0). Tắt nó giữ MLP cạnh tranh được dưới ngân sách mẫu nhỏ.

---

## 3.13. Giao thức đánh giá (`src/evaluation/metrics.py`)

### 3.13.1. Các chỉ số

Quy ước `0 = BENIGN`, `1 = ATTACK`. Với confusion matrix `[[TN, FP], [FN, TP]]`:

| Chỉ số | Công thức | Ý nghĩa an ninh |
|---|---|---|
| Accuracy | `(TP+TN)/tổng` | tổng thể (dễ đánh lừa khi mất cân bằng) |
| **F1** | `2·P·R/(P+R)` | **chỉ số chính** (cân bằng P/R lớp tấn công) |
| Precision (attack) | `TP/(TP+FP)` | trong các cờ báo, bao nhiêu là thật |
| Recall (attack) | `TP/(TP+FN)` | trong tấn công thật, bắt được bao nhiêu |
| **FNR** (bỏ sót) | `1 − recall = FN/(FN+TP)` | **nguy hiểm nhất**: tấn công bị bỏ lỡ |
| FPR (báo động giả) | `1 − recall_benign = FP/(FP+TN)` | phiền toái vận hành |
| ROC-AUC | diện tích dưới ROC | chất lượng xếp hạng độc lập ngưỡng |

ROC-AUC chỉ tính khi có xác suất và đủ hai lớp (ngược lại trả `NaN`).

### 3.13.2. Chỉ số khoảng cách (gap) — đo overfitting

Với mỗi chỉ số, báo cáo cả train và test, cùng **khoảng cách**:

$$\text{gap} = \text{metric}_{\text{train}} - \text{metric}_{\text{test}}.$$

`gap` dương lớn = overfitting. Trên S3, gap chính là thước đo trực tiếp của thất bại generalization chéo miền — và là nơi tương phản "ghi nhớ vs học hành vi" lộ ra (ví dụ RF: train-F1 ≈ 0,99 nhưng test-F1 ≈ 0,74 trên S3).

---

## 3.14. Ma trận thực nghiệm và tái lập

### 3.14.1. Sweep (`scripts/generate_configs.py`)

| Chiều | Giá trị |
|---|---|
| Kịch bản | S1, S3 |
| `N` train/lớp | 100, 250, 500 |
| Test/lớp | 500 (cố định) |
| Đặc trưng | 4 (= số qubit) |
| Seed | 0, 1, 2, 3, 4 |
| Cổ điển | LR, SVM, RF, MLP, mlp_tiny |
| QNN | q2 (angle+BE), q3 (angle+SE), q4 (reuploading+BE) |
| Biến thể QNN | `z` (base), `z+zz`, `probs`, `tenc` |

Tổng: **150 cấu hình cổ điển** (5×2×3×5) + **360 cấu hình QNN** (3×4×2×3×5) = **510 thí nghiệm**, mỗi con số trung bình hóa trên 5 seed; báo cáo kèm độ lệch chuẩn.

### 3.14.2. Tái lập

- **Seed cố định** cho NumPy, Python, và mạch (`set_all_seeds`).
- Mỗi run lưu `config.yaml` (cấu hình đã dùng) + `meta.json` (git commit, thời gian bắt đầu/kết thúc, cờ smoke-test) + `metrics.json` + `history.json`.
- **Idempotent**: chạy lại bỏ qua thí nghiệm đã hoàn thành.
- Stack: PennyLane (≥0.34, autograd + `lightning.qubit`), scikit-learn 1.9, NumPy, pandas; Python 3.12.

---

## 3.15. Tóm tắt chương

Chương đã trình bày một phương pháp luận hoàn chỉnh và có thể tái lập cho việc đánh giá QNN trên IDS dưới dữ liệu hạn chế:

1. **Quy trình dữ liệu chống rò rỉ** với danh sách đen đặc trưng chống shortcut learning, lấy mẫu train/test rời rạc, và fit scaler/selector chỉ trên train.
2. **Ba đóng góp phương pháp** — trainable feature map `RY(a·φ(x)+b)` (khởi tạo trùng baseline), multi-observable readout `z/z+zz/probs`, và hybrid architecture với linear head huấn luyện end-to-end qua một vectơ tham số phẳng.
3. **Thuật toán huấn luyện** Adam + mini-batch vector hóa + early stopping khôi phục tham số tốt nhất, kèm chẩn đoán gradient.
4. **Đối chứng nghiêm ngặt** gồm baseline cổ điển **khớp tham số** (`mlp_tiny`) để tách biệt lợi thế biểu diễn khỏi lợi thế dung lượng.
5. **Giao thức đánh giá thống nhất** ưu tiên F1/FNR và đo khoảng cách train–test để phơi bày overfitting.

Chương 4 áp dụng phương pháp này trên 510 cấu hình và trình bày kết quả benchmark, ablation, ổn định, và generalization.

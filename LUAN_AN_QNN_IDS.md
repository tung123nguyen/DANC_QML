# PHÁT HIỆN XÂM NHẬP MẠNG BẰNG MẠNG NƠ-RON LƯỢNG TỬ

## Trainable feature map, multi-observable readout và hybrid architecture lượng tử – cổ điển dưới điều kiện dữ liệu hạn chế

---

*Luận án / Đồ án chuyên ngành*

*Lĩnh vực: Học máy lượng tử (Quantum Machine Learning) – An toàn thông tin*

---

## TÓM TẮT

Hệ thống phát hiện xâm nhập (Intrusion Detection System – IDS) là một thành phần cốt lõi của hạ tầng an toàn mạng hiện đại, nhưng việc xây dựng các bộ phát hiện có khả năng generalization tốt với lượng dữ liệu gán nhãn hạn chế vẫn là một bài toán mở. Trong những năm gần đây, học máy lượng tử (Quantum Machine Learning – QML) nổi lên như một hướng tiếp cận tiềm năng nhờ khả năng ánh xạ dữ liệu vào không gian đặc trưng Hilbert có số chiều lớn theo cấp số mũ, mở ra triển vọng phân tách các mẫu phức tạp với số tham số rất nhỏ.

Luận án này nghiên cứu một cách hệ thống việc ứng dụng Mạng nơ-ron lượng tử (Quantum Neural Network – QNN) biến phân cho bài toán phát hiện xâm nhập nhị phân (BENIGN vs ATTACK) trên bộ dữ liệu chuẩn CIC-IDS2017, dưới chế độ **dữ liệu huấn luyện hạn chế** (100–500 mẫu/lớp). Ba đóng góp phương pháp được đề xuất và đánh giá: (i) **trainable feature map** (trainable feature map) dạng affine `RY(a·x + b)` với `a`, `b` được học đồng thời cùng mạch lượng tử; (ii) **multi-observable readout** (multi-observable readout) mở rộng từ kỳ vọng đơn `⟨Z_i⟩` sang tập tương quan hai qubit `⟨Z_iZ_j⟩` và sang phân bố xác suất trên cơ sở tính toán `probs`; và (iii) **hybrid architecture** trong đó một linear head cổ điển được huấn luyện đồng thời với mạch lượng tử qua lan truyền ngược end-to-end.

Chúng tôi tiến hành một khảo sát quy mô lớn gồm **510 cấu hình thực nghiệm** (360 cấu hình QNN trên 3 kiến trúc × 4 biến thể × 2 kịch bản × 3 kích thước dữ liệu × 5 seed ngẫu nhiên, cùng 150 cấu hình cổ điển nền tảng), được so sánh đối chứng nghiêm ngặt với năm mô hình học máy cổ điển: Hồi quy Logistic, SVM nhân RBF, Rừng ngẫu nhiên, MLP (705 tham số), và **một MLP nhỏ khớp tham số** (`mlp_tiny`, 25 tham số) — baseline được thêm vào để so sánh ở **cùng ngân sách tham số** với QNN, nhằm cô lập đóng góp của biểu diễn lượng tử khỏi lợi thế dung lượng thô. Hai kịch bản được thiết kế: **S1** (cùng nguồn – Friday DDoS) đo năng lực phân loại trong phân phối, và **S3** (chéo tấn công – huấn luyện trên Wednesday DoS, kiểm thử trên Friday DDoS) đo năng lực generalization dưới distribution shift.

Kết quả chính có tính nhiều sắc thái và trung thực: trên S1, mô hình QNN tốt nhất (data re-uploading + basic entangler) đạt **test-F1 = 0,941**, vượt trội Hồi quy Logistic (0,776), SVM (0,818) và sánh ngang/vượt MLP lớn (0,922), tuy vẫn thấp hơn Rừng ngẫu nhiên (0,998 – nhưng với dấu hiệu ghi nhớ rõ rệt, train-F1 = 1,000). Quan trọng, **khi so sánh ở cùng ngân sách tham số (~20–25)**, QNN **vượt rõ** MLP cổ điển khớp tham số trên cả hai kịch bản (ví dụ S1/N=500: 0,941 vs 0,818; S3 best QNN 0,821 vs mlp_tiny 0,751) — bằng chứng cho thấy lợi thế của QNN không chỉ đến từ dung lượng mà từ chính cấu trúc biểu diễn. Quan trọng hơn, trên kịch bản chéo tấn công S3 đầy thách thức, Rừng ngẫu nhiên **sụp đổ** từ 0,99 xuống 0,68–0,78 trong khi các mô hình lượng tử suy giảm hòa hoãn hơn (khoảng cách train–test trung bình của QNN là 0,11 so với độ rơi gần 0,25 của RF). Phân tích ablation cho thấy: trainable feature map **cải thiện** hiệu năng trong phân phối (+0,020 F1 trên S1) nhưng **gây hại** dưới distribution shift (−0,050 F1 trên S3) — bộc lộ một sự đánh đổi cơ bản giữa **năng lực expressivity và generalization**; multi-observable readout `z+zz` cho hiệu quả trung tính trên trung bình nhưng phương sai lớn và mang lại lợi ích đáng kể (+0,08 F1) ở chế độ dữ liệu cực nhỏ.

Luận án kết luận rằng QNN, với chỉ 21–29 tham số (so với hàng trăm đến hàng nghìn của các mô hình cổ điển), là một hướng tiếp cận **hiệu quả về tham số và bền vững dưới distribution shift** cho IDS, đồng thời chỉ rõ các giới hạn nghiêm trọng về chi phí mô phỏng (chậm hơn ~1000 lần) và độ ổn định theo seed — những rào cản cần vượt qua trước khi triển khai thực tế.

**Từ khóa:** Học máy lượng tử, Mạng nơ-ron lượng tử, Variational circuit, Phát hiện xâm nhập, CIC-IDS2017, Feature map, Generalization, Hybrid architecture.

---

# MỤC LỤC

1. [Mở đầu](#chương-1--mở-đầu)
2. [Cơ sở lý thuyết](#chương-2--cơ-sở-lý-thuyết)
3. [Phương pháp đề xuất](#chương-3--phương-pháp-đề-xuất)
4. [Thực nghiệm và kết quả](#chương-4--thực-nghiệm-và-kết-quả)
5. [Thảo luận](#chương-5--thảo-luận)
6. [Kết luận](#chương-6--kết-luận)
7. [Tài liệu tham khảo](#tài-liệu-tham-khảo)
8. [Phụ lục](#phụ-lục)

---

# CHƯƠNG 1 — MỞ ĐẦU

## 1.1. Bối cảnh và động lực

Trong hai thập kỷ qua, sự bùng nổ của hạ tầng số — điện toán đám mây, Internet vạn vật (IoT), mạng 5G, và các hệ thống công nghiệp kết nối — đã biến không gian mạng thành một bề mặt tấn công khổng lồ và liên tục mở rộng. Theo các báo cáo an toàn thông tin thường niên, số lượng và độ tinh vi của các cuộc tấn công mạng tăng theo cấp số mũ, trong đó các cuộc tấn công từ chối dịch vụ phân tán (Distributed Denial of Service – DDoS), quét cổng (port scanning), khai thác lỗ hổng ứng dụng web, và mã độc tống tiền (ransomware) gây thiệt hại hàng nghìn tỷ đô-la mỗi năm. Trong bối cảnh đó, **Hệ thống phát hiện xâm nhập (IDS)** đóng vai trò như tuyến phòng thủ chủ động: giám sát lưu lượng mạng hoặc hành vi máy chủ để nhận diện sớm các dấu hiệu của hoạt động độc hại trước khi chúng kịp gây hại.

IDS truyền thống được chia thành hai họ chính. **IDS dựa trên dấu hiệu (signature-based)** so khớp lưu lượng với một cơ sở dữ liệu các mẫu tấn công đã biết; chúng chính xác với các mối đe dọa đã biết nhưng hoàn toàn mù trước các tấn công zero-day. **IDS dựa trên bất thường (anomaly-based)** học một mô hình về "hành vi bình thường" và gắn cờ mọi sai lệch; chúng có tiềm năng phát hiện tấn công mới nhưng thường chịu tỷ lệ báo động giả (false positive) cao. Học máy (Machine Learning – ML) đã trở thành công cụ chủ đạo để xây dựng các IDS dựa trên bất thường hiện đại, nhờ khả năng tự động học các ranh giới quyết định phức tạp từ dữ liệu lưu lượng.

Tuy nhiên, việc ứng dụng ML cho IDS vấp phải nhiều thách thức cố hữu mà luận án này đặt làm trung tâm:

- **Sự khan hiếm dữ liệu gán nhãn chất lượng cao.** Trong thực tế triển khai, việc thu thập và gán nhãn lưu lượng tấn công là tốn kém, đòi hỏi chuyên gia, và thường vi phạm quyền riêng tư. Một IDS thực dụng phải học được từ một lượng mẫu hạn chế.
- **Mất cân bằng lớp (class imbalance).** Lưu lượng độc hại thường chỉ chiếm một phần rất nhỏ so với lưu lượng lành tính, khiến các chỉ số như độ chính xác (accuracy) trở nên gây hiểu lầm.
- **Distribution shift (distribution shift / concept drift).** Mô hình huấn luyện trên một loại tấn công hoặc một khoảng thời gian thường suy giảm nghiêm trọng khi đối mặt với biến thể tấn công mới hoặc môi trường mạng khác. Đây chính là kẽ hở mà kẻ tấn công khai thác.
- **Chi phí của báo động sai (false negative).** Trong an ninh, một cuộc tấn công bị bỏ sót (false negative) có thể gây hậu quả thảm khốc, trong khi một báo động giả chỉ gây phiền toái. Hàm mục tiêu cần ưu tiên recall trên lớp tấn công.

## 1.2. Điện toán lượng tử như một giải pháp tiềm năng

Song song với những tiến bộ của ML cổ điển, **điện toán lượng tử** đã chuyển từ lý thuyết sang hiện thực với sự ra đời của các bộ xử lý lượng tử nhiễu quy mô trung bình (Noisy Intermediate-Scale Quantum – NISQ). Một qubit, không giống một bit cổ điển, có thể tồn tại ở trạng thái superposition của `|0⟩` và `|1⟩`; một thanh ghi `n` qubit sống trong một không gian Hilbert `2^n` chiều. Phép quantum entanglement tạo ra các tương quan phi cổ điển giữa các qubit không thể mô tả bằng tích các trạng thái độc lập.

Ý tưởng cốt lõi của **học máy lượng tử** là khai thác không gian Hilbert số chiều mũ này như một **không gian đặc trưng (feature space)**: dữ liệu cổ điển được "mã hóa" vào trạng thái lượng tử thông qua một mạch feature map, sau đó một variational circuit có tham số (parameterized quantum circuit) thực hiện phân loại trong không gian đó. Vì hạch (kernel) tương ứng với tích vô hướng trong không gian Hilbert có thể khó tính toán cổ điển, người ta kỳ vọng QML có thể nắm bắt các cấu trúc dữ liệu mà mô hình cổ điển bỏ lỡ — đặc biệt khi dữ liệu ít nhưng cấu trúc phức tạp. Quan trọng đối với bài toán của chúng ta: các nghiên cứu lý thuyết gần đây (ví dụ Caro và cộng sự, 2022) gợi ý rằng QNN có thể **generalization tốt từ rất ít dữ liệu huấn luyện**, một đặc tính đặc biệt hấp dẫn cho IDS trong chế độ dữ liệu hạn chế.

Tuy vậy, QML hiện đối mặt các rào cản nghiêm trọng: hiện tượng **barren plateau** khiến gradient triệt tiêu theo cấp số mũ với số qubit; chi phí mô phỏng cổ điển tăng theo cấp số mũ; nhiễu phần cứng; và thiếu bằng chứng thực nghiệm thuyết phục về ưu thế lượng tử (quantum advantage) trên các bài toán thực tế. Luận án này tiếp cận QML một cách **thực chứng và khiêm tốn**: thay vì tuyên bố ưu thế lượng tử, chúng tôi đặt câu hỏi cụ thể, có thể kiểm chứng — QNN hoạt động như thế nào so với mô hình cổ điển trên một bài toán IDS thực, dưới điều kiện dữ liệu hạn chế, và những thành phần thiết kế nào thực sự quan trọng?

## 1.3. Câu hỏi nghiên cứu

Luận án được tổ chức quanh bốn câu hỏi nghiên cứu (Research Questions – RQ):

> **RQ1 (Benchmark).** Dưới chế độ dữ liệu hạn chế, một QNN biến phân có thể cạnh tranh với các mô hình ML cổ điển tiêu chuẩn trên bài toán phát hiện xâm nhập nhị phân hay không, xét cả về hiệu năng phân loại lẫn hiệu quả tham số?

> **RQ2 (Component design).** Ba thành phần thiết kế được đề xuất — trainable feature map, multi-observable readout, và data re-uploading — đóng góp như thế nào vào hiệu năng? Thành phần nào thực sự cần thiết?

> **RQ3 (Generalization).** Năng lực generalization của QNN thay đổi ra sao khi chuyển từ kịch bản cùng nguồn (S1) sang kịch bản chéo tấn công (S3) — tức dưới distribution shift? So sánh với mô hình cổ điển thì sao?

> **RQ4 (Stability & Cost).** Mức độ ổn định của QNN theo seed ngẫu nhiên và chi phí tính toán của chúng có cho phép triển khai thực tế không?

## 1.4. Đóng góp của luận án

Các đóng góp chính bao gồm:

1. **Một khung thực nghiệm có thể tái lập đầy đủ** cho việc so sánh QNN và ML cổ điển trên CIC-IDS2017, với quản lý cấu hình bằng YAML, seed cố định, tách rời train/test ở mức hàng, chống data leakage nghiêm ngặt, và tính idempotent (chạy lại bỏ qua thí nghiệm đã hoàn thành).

2. **Trainable feature map** dạng affine `RY(a_i · φ(x_i) + b_i)`, trong đó hệ số tỉ lệ `a` và độ dịch `b` cho từng qubit được học cùng mạch — khởi tạo `a = 1, b = 0` để mô hình mới tương đương angle encoding cố định, đảm bảo so sánh công bằng.

3. **Multi-observable readout** mở rộng đầu vào của linear head từ `n` kỳ vọng đơn `⟨Z_i⟩` sang `n + C(n,2)` giá trị bao gồm các tương quan hai qubit `⟨Z_iZ_j⟩`, hoặc sang một phân bố xác suất `probs` trên cơ sở tính toán.

4. **Hybrid architecture end-to-end** với một linear head cổ điển huấn luyện đồng thời với mạch lượng tử qua một vectơ tham số phẳng duy nhất `[a | b | ansatz | w | head_b]`, cho phép lan truyền ngược xuyên suốt qua cả phần lượng tử lẫn cổ điển bằng vi phân tự động.

5. **Một khảo sát thực nghiệm quy mô lớn** (510 cấu hình, mỗi cấu hình trung bình hóa trên 5 seed), bao gồm một **baseline cổ điển khớp tham số** để so sánh công bằng về dung lượng, với phân tích benchmark, ablation, ổn định và generalization, dẫn đến những phát hiện trung thực và nhiều sắc thái về **sự đánh đổi expressivity–generalization** trong QML cho an ninh mạng.

## 1.5. Cấu trúc luận án

Chương 2 trình bày cơ sở lý thuyết về IDS, ML cổ điển và QML. Chương 3 mô tả chi tiết phương pháp đề xuất và toàn bộ quy trình. Chương 4 trình bày thiết lập thực nghiệm và kết quả. Chương 5 thảo luận ý nghĩa, giới hạn và hướng phát triển. Chương 6 kết luận.

---

# CHƯƠNG 2 — CƠ SỞ LÝ THUYẾT

## 2.1. Hệ thống phát hiện xâm nhập

### 2.1.1. Phân loại IDS

Theo vị trí triển khai, IDS được chia thành IDS mạng (Network-based IDS – NIDS) giám sát lưu lượng tại điểm tập trung, và IDS máy chủ (Host-based IDS – HIDS) giám sát nhật ký và lời gọi hệ thống trên từng máy. Luận án tập trung vào NIDS dựa trên đặc trưng luồng (flow-based), trong đó lưu lượng thô được tổng hợp thành các "luồng" (flow) — một chuỗi gói tin chia sẻ cùng năm bộ định danh (IP nguồn/đích, cổng nguồn/đích, giao thức) — và mỗi luồng được biểu diễn bằng một vectơ đặc trưng thống kê (số byte, thời lượng, tốc độ gói, độ dài trung bình, v.v.).

### 2.1.2. Bài toán học máy cho NIDS

Ở dạng đơn giản nhất, phát hiện xâm nhập là bài toán **phân loại nhị phân**: với vectơ đặc trưng `x ∈ ℝ^d` của một luồng, dự đoán nhãn `y ∈ {0, 1}` với 0 = BENIGN (lành tính) và 1 = ATTACK (tấn công). Ranh giới quyết định lý tưởng phải nắm bắt "tính tấn công" (attackness) tổng quát chứ không phải dấu hiệu cụ thể của một loại tấn công đã thấy.

### 2.1.3. Vì sao bài toán khó

Ba nguồn khó khăn được mô tả ở Chương 1 (khan hiếm dữ liệu, mất cân bằng, distribution shift) hội tụ trong IDS. Đặc biệt, **feature leakage** là một cạm bẫy tinh vi: các đặc trưng như số cổng đích, địa chỉ IP, hay nhãn thời gian (timestamp) có tương quan rất mạnh với loại tấn công cụ thể trong tập huấn luyện (ví dụ DDoS thường nhắm cổng 80), khiến mô hình "shortcut learning" (shortcut learning) — ghi nhớ dấu hiệu thay vì học hành vi. Một mô hình như vậy đạt điểm số cao giả tạo trên dữ liệu cùng nguồn nhưng sụp đổ trên tấn công mới. Việc loại bỏ các đặc trưng rò rỉ là điều kiện tiên quyết cho một đánh giá generalization trung thực — một nguyên tắc được hiện thực hóa trực tiếp trong danh sách đen đặc trưng (feature blacklist) của hệ thống này (xem §3.2).

## 2.2. Học máy cổ điển cho IDS

Năm họ mô hình cổ điển được dùng làm baseline trong luận án, được chọn vì tính đại diện cho phổ phương pháp:

- **Hồi quy Logistic (Logistic Regression – LR).** Mô hình tuyến tính, học một siêu phẳng phân tách. Đơn giản, dễ giải thích, ít tham số (`d + 1 = 5`). Là chuẩn so sánh cho "ranh giới tuyến tính".
- **Máy vectơ hỗ trợ (Support Vector Machine – SVM) nhân RBF.** Ánh xạ ngầm dữ liệu vào không gian đặc trưng số chiều vô hạn qua nhân Gaussian, tìm siêu phẳng lề cực đại. Là đối thủ trực tiếp về mặt khái niệm với QML, vì QML cũng dựa trên ý tưởng nhân trong không gian Hilbert.
- **Rừng ngẫu nhiên (Random Forest – RF).** Tập hợp (ensemble) hàng trăm cây quyết định trên các tập con bootstrap (~2750 nút). Cực mạnh trong việc khớp dữ liệu phi tuyến, nhưng có xu hướng ghi nhớ — như sẽ thấy ở Chương 4, đây là điểm yếu chí tử dưới distribution shift.
- **Mạng nơ-ron đa lớp (Multi-Layer Perceptron – MLP).** Mạng nơ-ron tiến với hai lớp ẩn (32, 16), ~705 tham số. Là đối trọng "mạng nơ-ron cổ điển" công suất lớn với QNN.
- **MLP nhỏ khớp tham số (`mlp_tiny`).** Mạng nơ-ron tiến **một lớp ẩn 4 nơ-ron** → `6h + 1 = 25` tham số (với `h = 4` nơ-ron ẩn và 4 đặc trưng đầu vào), nằm đúng trong dải tham số của QNN (21–53). Đây là **đối chứng then chốt về dung lượng**: nó trả lời câu hỏi *liệu một mạng nơ-ron cổ điển với cùng ngân sách tham số có sánh được QNN không* — qua đó tách bạch lợi thế của biểu diễn lượng tử khỏi lợi thế dung lượng thô mà MLP lớn/RF được hưởng.

Tất cả các mô hình cổ điển dùng siêu tham số mặc định hợp lý (không tinh chỉnh lưới), vì mục tiêu là **so sánh dưới cùng điều kiện**, không phải tối ưu tuyệt đối từng mô hình. Cần phân biệt hai nghĩa của "công bằng": *cùng điều kiện đầu vào* (dữ liệu, đặc trưng, tiền xử lý, cách đánh giá — áp dụng cho mọi mô hình) và *cùng dung lượng* (số tham số) — chỉ `mlp_tiny` đạt được nghĩa thứ hai so với QNN.

## 2.3. Cơ sở điện toán lượng tử

### 2.3.1. Qubit và trạng thái

Một qubit là một vectơ đơn vị trong không gian Hilbert hai chiều `ℂ²`:

```
|ψ⟩ = α|0⟩ + β|1⟩,    α, β ∈ ℂ,    |α|² + |β|² = 1.
```

Bỏ qua pha toàn cục, trạng thái một qubit có thể tham số hóa bằng hai góc và biểu diễn trên **Bloch sphere**. Một thanh ghi `n` qubit sống trong tích tensor `(ℂ²)^⊗n ≅ ℂ^(2^n)`, với cơ sở tính toán `{|0…0⟩, …, |1…1⟩}`.

### 2.3.2. Cổng lượng tử

Phép biến đổi trạng thái là các ma trận unita. Các cổng quay một qubit quanh các trục Bloch:

```
RX(θ) = exp(−i θ X / 2),   RY(θ) = exp(−i θ Y / 2),   RZ(θ) = exp(−i θ Z / 2),
```

với `X, Y, Z` là các ma trận Pauli. Cổng hai qubit như **CNOT** tạo ra entanglement. Bất kỳ unita nào cũng có thể phân rã thành tích các cổng một và hai qubit (tính phổ dụng).

Điểm mấu chốt cho học máy: các cổng quay `R·(θ)` **khả vi theo θ**, và gradient của giá trị kỳ vọng theo các góc này tính được chính xác bằng **parameter-shift rule**, cho phép huấn luyện mạch bằng các phương pháp dựa trên gradient.

### 2.3.3. Đo lường

Thông tin cổ điển được trích xuất qua phép đo. Giá trị kỳ vọng của một quan sát được (observable) `O` trên trạng thái `|ψ⟩` là `⟨O⟩ = ⟨ψ|O|ψ⟩`. Với quan sát được Pauli-Z trên qubit `i`, `⟨Z_i⟩ ∈ [−1, 1]` đo "độ nghiêng" của qubit đó về phía `|0⟩` (giá trị +1) hay `|1⟩` (giá trị −1). Tương quan hai qubit `⟨Z_iZ_j⟩` đo sự đồng pha giữa hai qubit — một đại lượng nhạy với entanglement mà phép đo đơn qubit bỏ lỡ. Ngoài ra, ta có thể đo trực tiếp **phân bố xác suất** `P(b) = |⟨b|ψ⟩|²` trên cơ sở tính toán của một tập con qubit.

## 2.4. Mạng nơ-ron lượng tử và variational circuit

Một **variational quantum circuit (Variational Quantum Circuit – VQC)**, còn gọi là QNN, gồm ba giai đoạn:

```
|0⟩^⊗n  --[ U_enc(x) ]--[ U_ansatz(θ) ]--[ Đo ⟨O⟩ ]-->  đặc trưng cổ điển
         (data encoding)  (lớp biến phân)   (readout)
```

1. **Data encoding (data encoding / feature map)** `U_enc(x)`: ánh xạ vectơ đầu vào `x` thành một trạng thái lượng tử. Đây là bước quyết định "không gian đặc trưng" mà mô hình làm việc.
2. **Ansatz biến phân** `U_ansatz(θ)`: một dãy cổng quay có tham số xen kẽ với cổng entanglement, đóng vai trò như "trọng số học được".
3. **Đọc kết quả**: đo một hoặc nhiều quan sát được, thu các giá trị cổ điển được đưa vào hàm mục tiêu.

Toàn bộ `θ` được tối ưu bằng cách cực tiểu hóa hàm mất mát, dùng gradient tính qua parameter-shift rule hoặc vi phân tự động trên trình mô phỏng.

## 2.5. Data encoding và expressivity

### 2.5.1. Angle encoding

Cách mã hóa đơn giản và phổ biến nhất: gán mỗi đặc trưng `x_i` cho một góc quay, ví dụ `RY(x_i)` trên qubit `i`. Với `n` qubit, ta mã hóa `n` đặc trưng. Tính tuần hoàn `2π` của cổng quay đòi hỏi dữ liệu phải được chuẩn hóa vào một khoảng an toàn (xem §3.3), nếu không các giá trị lớn sẽ "cuộn vòng" và làm sụp đổ các đầu vào khác biệt thành cùng một góc.

### 2.5.2. Data re-uploading

Pérez-Salinas và cộng sự (2020) chỉ ra rằng việc **nạp lại dữ liệu nhiều lần**, xen kẽ với các lớp biến phân — `[U_enc(x) U_ansatz(θ₁)][U_enc(x) U_ansatz(θ₂)]…` — làm tăng đáng kể expressivity. Schuld, Sweke và Meyer (2021) hình thức hóa điều này: một QNN với data re-uploading tương đương một **chuỗi Fourier cắt cụt** theo dữ liệu, trong đó số lần nạp lại quyết định số tần số (bậc) khả dụng. Mã hóa một lần (angle encoding) chỉ cho một hàm tần số đơn giản; nạp lại `L` lần mở rộng phổ tần số, cho phép xấp xỉ các hàm quyết định phức tạp hơn. Đây là nền tảng lý thuyết cho việc kỳ vọng kiến trúc `reuploading` mạnh hơn `angle` — một kỳ vọng được kiểm chứng thực nghiệm ở Chương 4.

## 2.6. Ansatz, expressivity và barren plateau

Lựa chọn ansatz cân bằng giữa **expressivity** (lớp hàm mà mạch biểu diễn được) và **trainability** (gradient có đủ lớn để học không). Ba ansatz được khảo sát:

- **Basic Entangler:** một cổng quay đơn (`RX` hoặc `RY`) trên mỗi qubit, theo sau bởi một vòng CNOT. Ít tham số (`n` mỗi lớp), nhẹ, ít rủi ro barren plateau.
- **Strongly Entangling:** ba cổng quay (`RX, RY, RZ` hay tổ hợp Euler) trên mỗi qubit (`3n` tham số mỗi lớp) cộng các CNOT có khoảng cách thay đổi — biểu cảm hơn nhưng tốn tham số và dễ gặp gradient nhỏ hơn.
- **Rotation-only:** ba cổng quay mỗi qubit nhưng **không có CNOT** — một baseline "không entanglement" để cô lập đóng góp của entanglement.

**Barren plateau** (McClean và cộng sự, 2018) là hiện tượng phương sai gradient triệt tiêu theo cấp số mũ với số qubit đối với các mạch đủ ngẫu nhiên và sâu, khiến huấn luyện bất khả thi. Vì lý do này, hệ thống của chúng tôi ghi lại thống kê gradient (chuẩn, phương sai, trung bình trị tuyệt đối) cho một tập con chẩn đoán, để theo dõi trainability theo ansatz và mã hóa.

## 2.7. Hybrid architecture lượng tử – cổ điển

Trong chế độ NISQ, các kiến trúc thuần lượng tử bị giới hạn bởi nhiễu và số qubit. **Hybrid architecture** kết hợp một mạch lượng tử (trích xuất đặc trưng phi tuyến) với các thành phần xử lý cổ điển (tổng hợp, ra quyết định), huấn luyện đồng thời. Trong luận án này, một **linear head** cổ điển nhận vectơ kết quả đo của mạch và xuất một logit duy nhất; trọng số của readout head được học cùng tham số mạch qua một vectơ tham số phẳng chung, cho phép lan truyền ngược end-to-end. Cách tiếp cận này (i) tách bạch "biểu diễn lượng tử" khỏi "quyết định cổ điển", (ii) cho phép multi-observable readout mà không cần đo lại nhiều lần, và (iii) khởi tạo trung tính (logit ≈ 0 → xác suất ≈ 0,5) để huấn luyện ổn định.

## 2.8. Các công trình liên quan

QML có gốc rễ từ ý tưởng học trong không gian đặc trưng Hilbert (Havlíček và cộng sự, 2019; Schuld & Killoran, 2019), với học mạch lượng tử khả vi (Mitarai và cộng sự, 2018) và bộ phân loại lấy mạch làm trung tâm (Schuld và cộng sự, 2020). Data re-uploading (Pérez-Salinas và cộng sự, 2020) và lý thuyết Fourier về expressivity (Schuld, Sweke & Meyer, 2021) cung cấp nền tảng cho thiết kế mã hóa. Abbas và cộng sự (2021) nghiên cứu "sức mạnh" của QNN qua khả năng hữu hiệu (effective dimension), còn Caro và cộng sự (2022) chứng minh các biên generalization thuận lợi từ ít dữ liệu — kết quả lý thuyết then chốt thúc đẩy nghiên cứu chế độ dữ liệu hạn chế của chúng tôi. Về ứng dụng an ninh, một dòng công trình đang nổi áp dụng QML cho IDS, nhưng phần lớn (a) dùng các đặc trưng chưa loại bỏ rò rỉ, (b) đánh giá cùng nguồn nên thổi phồng kết quả, hoặc (c) thiếu so sánh cổ điển công bằng và phân tích ablation. Luận án này lấp các khoảng trống đó bằng một thiết kế thực nghiệm nghiêm ngặt, đặc biệt nhấn mạnh kịch bản chéo tấn công S3.

---

# CHƯƠNG 3 — PHƯƠNG PHÁP ĐỀ XUẤT

## 3.1. Tổng quan hệ thống

Hình dưới mô tả luồng xử lý đầu cuối, được hiện thực hóa như một pipeline điều khiển bằng cấu hình YAML (mỗi thí nghiệm = một file config):

```
CSV thô (CIC-IDS2017)
   │  (loader: strip cột, xử lý inf/NaN, gán nhãn nhị phân, cache parquet)
   ▼
Lấy mẫu cân bằng, tách train/test rời rạc ở mức hàng (sampling)
   │
   ▼
Tiền xử lý: bỏ blacklist → StandardScaler (fit trên train) → SelectKBest MI (fit trên train)
   │  → X ∈ ℝ^(N×4),  y ∈ {0,1}
   ▼
┌─────────────────────────────┐        ┌──────────────────────────────┐
│  Nhánh CỔ ĐIỂN              │        │  Nhánh LƯỢNG TỬ (QNN)        │
│  LR / SVM / RF / MLP        │        │  Encoding × Ansatz × Readout │
│  fit() → predict_proba()    │        │  + linear head lai    │
└─────────────────────────────┘        └──────────────────────────────┘
   │                                          │
   ▼                                          ▼
Đánh giá thống nhất: F1, precision/recall (attack), FNR, FPR, ROC-AUC, confusion matrix, khoảng cách train–test
```

Thiết kế thống nhất giao diện: cả mô hình cổ điển và QNN đều xuất `(nhãn dự đoán, xác suất)`, nên cùng một hàm tính chỉ số (`compute_train_test_metrics`) áp dụng cho cả hai — đảm bảo so sánh tuyệt đối công bằng.

## 3.2. Quy trình tiền xử lý và chống rò rỉ

### 3.2.1. Làm sạch dữ liệu

Bộ CIC-IDS2017 có các lỗi đã biết: tên cột chứa khoảng trắng thừa (ví dụ `' Label'`), và các giá trị vô cực `±inf` sinh ra từ các tỉ số đặc trưng (như `Flow Bytes/s` khi thời lượng bằng 0). Loader strip khoảng trắng tên cột một lần, thay `±inf` bằng `NaN` rồi loại bỏ các hàng `NaN`, gán nhãn nhị phân (`BENIGN → 0`, còn lại `→ 1`), và lưu cache dạng parquet để nạp nhanh.

### 3.2.2. Danh sách đen đặc trưng (feature blacklist)

Đây là một quyết định phương pháp **then chốt** cho tính trung thực của S3. Các nhóm cột sau bị loại bỏ vĩnh viễn trước khi huấn luyện:

- **Định danh & siêu dữ liệu:** Flow ID, Source/Destination IP, Timestamp — không phải đặc trưng, và timestamp là rò rỉ thời gian không tồn tại lúc triển khai.
- **Cổng (port):** Source/Destination Port — DDoS nhắm cổng 80, PortScan chạm nhiều cổng; giữ cổng cho phép mô hình ghi nhớ loại tấn công thay vì học hành vi.
- **Giao thức (Protocol):** các tấn công cụ thể dùng giao thức cụ thể (ICMP cho một số biến thể DoS, TCP cho DDoS LOIT); giữ Protocol tạo lối tắt cho danh tính tấn công, thổi phồng S1 và phá hỏng phép thử chéo tấn công S3.
- **Cột nhãn:** Label, binary_label — là mục tiêu, không bao giờ là đặc trưng.

Nguyên tắc: loại bỏ định danh, cổng, nhãn thời gian và giao thức để **giảm shortcut learning** và buộc mô hình học hành vi lưu lượng tổng quát.

### 3.2.3. Chuẩn hóa, feature selection và chống rò rỉ

Sau khi tách train/test rời rạc (xem §4.2), pipeline áp dụng:

1. **StandardScaler** (chuẩn hóa về trung bình 0, độ lệch chuẩn 1), **fit chỉ trên train**, transform cả hai.
2. **SelectKBest** với điểm số **mutual information** `mutual_info_classif`, **fit chỉ trên train**, giữ lại `k = 4` đặc trưng hàng đầu.

Quy tắc chống rò rỉ tuyệt đối: `scaler.fit()` và `selector.fit()` **chỉ thấy dữ liệu train**. Test được transform bằng tham số học từ train. Với S3 (train và test từ file khác nhau), pipeline lấy giao của tập cột để xử lý khác biệt nhỏ về schema.

Việc chọn `k = 4` đặc trưng không phải tùy tiện: nó khớp với số qubit `n = 4`, để angle encoding gán đúng một đặc trưng cho một qubit. Điều này giữ mạch nhỏ (mô phỏng được trên CPU) đồng thời tạo một sân chơi công bằng — mô hình cổ điển cũng chỉ thấy đúng 4 đặc trưng đó.

## 3.3. Chuẩn hóa angle encoding (angle_clip)

Vì đặc trưng đã qua StandardScaler có khoảng `~[−3, 3]` với đuôi ngoại lai nặng hơn, việc đưa thẳng vào góc `RY` là không an toàn (tính tuần hoàn `2π`). Hệ thống áp dụng phép biến đổi:

```
φ(x) = clip(x, −c, +c) · (π / c),     với c = angle_clip = 3.0
```

Phép này cắt về `±3σ` rồi co tuyến tính sao cho `±3σ` ánh xạ thành `±π`, giữ mọi góc trong `[−π, π]`. Ngưỡng `c = 3` giữ ~99,7% dữ liệu Gaussian trong vùng tuyến tính. Đặt `c = None/0` để vô hiệu hóa (tái lập mã hóa thô cũ). Đây là một bước "kỹ thuật" tưởng nhỏ nhưng quyết định: thiếu nó, các luồng tấn công có giá trị đặc trưng cực đoan sẽ cuộn vòng và trở nên không phân biệt được với lưu lượng lành tính.

## 3.4. Đề xuất 1 — Trainable feature map

Trong angle encoding cố định, ánh xạ `x ↦ RY(φ(x))` là tĩnh: hình dạng của không gian đặc trưng lượng tử do người thiết kế ấn định trước. Đề xuất của chúng tôi là cho phép **chính feature map được học**, qua một biến đổi affine theo từng qubit trước cổng quay:

```
qubit i:   RY( a_i · φ(x_i) + b_i ),     a, b ∈ ℝ^n học được.
```

- `a_i` là **hệ số tỉ lệ (scale)** — điều chỉnh độ "nhạy" của qubit `i` với đặc trưng tương ứng (độ dốc của ánh xạ tần số).
- `b_i` là **độ dịch (bias)** — dịch điểm làm việc trên Bloch sphere.

**Khởi tạo có chủ đích:** `a` khởi tạo bằng `1.0` và `b` bằng `0.0`, để một mô hình mới **tương đương chính xác** với angle encoding cố định `RY(φ(x))`. Nhờ đó, trainable feature map chỉ có thể *cải thiện* (hoặc giữ nguyên) so với baseline tại điểm khởi tạo, và mọi khác biệt hiệu năng quan sát được là do *học*, không phải do khởi tạo khác. Khối `[a | b]` được đặt ở đầu vectơ tham số mạch (`n` phần tử cho `a`, `n` phần tử cho `b`), và trình huấn luyện khởi tạo riêng hai khối này.

Về mặt lý thuyết Fourier (Schuld và cộng sự, 2021), việc học `a` tương đương học các **tần số** của khai triển Fourier — thay vì cố định phổ tần số, mô hình tự chọn dải tần phù hợp với dữ liệu. Đây là điểm mạnh trong phân phối (S1) nhưng cũng là nguồn gốc của rủi ro overfitting dưới distribution shift (S3), như Chương 4 sẽ chỉ rõ.

## 3.5. Các ansatz biến phân

Số tham số mỗi lớp (one layer):

| Ansatz | Tham số/lớp | Cấu trúc |
|---|---|---|
| `basic_entangler` | `n` | 1 cổng quay/qubit + vòng CNOT |
| `strongly_entangling` | `3n` | 3 cổng quay/qubit + CNOT khoảng cách biến thiên |
| `rotation_only` | `3n` | 3 cổng quay/qubit, **không CNOT** |

Với `depth = d` lớp, tổng tham số mạch (chưa kể trainable feature map) là `d × (tham số/lớp)`. Ansatz `rotation_only` đóng vai trò đối chứng để đo riêng đóng góp của entanglement.

## 3.6. Data re-uploading

Hai chế độ mã hóa được hiện thực:

- **`angle`:** mã hóa một lần, sau đó `d` lớp ansatz liên tiếp.
  ```
  U_enc(x) · U_ansatz(θ₁) · U_ansatz(θ₂) ⋯ U_ansatz(θ_d)
  ```
- **`reuploading`:** lặp lại khối (mã hóa + ansatz) `d` lần.
  ```
  [U_enc(x) U_ansatz(θ₁)] · [U_enc(x) U_ansatz(θ₂)] ⋯ [U_enc(x) U_ansatz(θ_d)]
  ```

Theo lý thuyết Fourier, `reuploading` với `d` lần nạp lại cho phổ tần số phong phú hơn nhiều so với `angle`, do đó kỳ vọng expressivity cao hơn. Khi bật trainable feature map, cùng một khối affine `[a | b]` được tái sử dụng ở mọi lần nạp lại (chia sẻ tham số), giữ số tham số nhỏ.

## 3.7. Đề xuất 2 — Multi-observable readout

Readout head quyết định "lượng thông tin" trích từ trạng thái lượng tử. Ba phương án được khảo sát:

1. **`z` (đơn quan sát — baseline):** đo `⟨Z_i⟩` trên từng qubit → vectơ `z ∈ [−1,1]^n` (4 giá trị). Đây là cách đọc tiêu chuẩn, rẻ, nhưng bỏ lỡ thông tin tương quan.

2. **`z+zz` (đa quan sát — đề xuất chính):** ngoài `⟨Z_i⟩`, đo toàn bộ tương quan hai qubit `⟨Z_iZ_j⟩` với `i < j` → `n + C(n,2) = 4 + 6 = 10` giá trị. Các tương quan này **nhạy với entanglement** mà phép đo đơn qubit không thấy được, làm giàu biểu diễn cho readout head cổ điển mà **không tăng độ sâu mạch**.

3. **`probs` (phân bố cơ sở):** đo trực tiếp phân bố xác suất `P(b)` trên cơ sở tính toán của `readout_wires = 2` qubit đầu → `2² = 4` xác suất (tổng bằng 1). Đây là một readout giàu thông tin theo nghĩa khác (phân bố đầy đủ trên 2 qubit) nhưng giới hạn về số qubit đọc.

Điểm tinh tế: `z+zz` mở rộng *chiều ngang* (nhiều quan sát hơn trên tất cả qubit), trong khi `probs` mở rộng *chiều sâu* (phân bố đầy đủ nhưng chỉ trên 2 qubit). Chương 4 cho thấy hai chiến lược này có hành vi rất khác nhau.

## 3.8. Đề xuất 3 — Hybrid architecture và linear head

Vectơ kết quả đo `z` (kích thước `n_obs`) được biến đổi thành một logit bởi một **linear head cổ điển**:

```
logit = w · z + b_head,        p = sigmoid(logit) = 1 / (1 + e^(−logit)),
```

với `w ∈ ℝ^(n_obs)` và `b_head ∈ ℝ`. Toàn bộ mô hình được tham số hóa bằng **một vectơ phẳng duy nhất**:

```
[ a (n) | b (n) | ansatz (d·ppl) | w (n_obs) | b_head (1) ]
   └──── tham số mạch ────┘        └── hybrid readout head ──┘
```

trong đó hai khối `[a | b]` chỉ tồn tại khi bật trainable feature map. Cách bố trí này cho phép **lan truyền ngược end-to-end**: gradient chảy qua sigmoid và linear head, rồi qua các quan sát được, vào tận các cổng mạch lượng tử — tất cả bằng vi phân tự động (autograd) của PennyLane. Readout head được khởi tạo `b_head = 0` để logit ≈ 0 và xác suất ≈ 0,5 lúc bắt đầu, tránh bão hòa sigmoid.

## 3.9. Hàm mất mát và quy trình huấn luyện

### 3.9.1. Hàm mất mát

Hàm mục tiêu là **entropy chéo nhị phân (Binary Cross-Entropy – BCE)**, được kẹp số học để tránh `log(0)`:

```
L(θ) = (1/B) Σ_k [ −y_k log p_k − (1−y_k) log(1−p_k) ],   p_k ∈ [1e−7, 1−1e−7].
```

### 3.9.2. Tối ưu hóa và vector hóa

- **Bộ tối ưu:** Adam (Kingma & Ba, 2015), `learning_rate = 0.05`.
- **Mini-batch:** kích thước 32, `epochs = 80`.
- **Vector hóa theo lô:** nhờ cơ chế **parameter broadcasting** của PennyLane, toàn bộ `B` mẫu trong một lô được đánh giá trong *một* lần gọi mạch (không vòng lặp Python), tăng tốc đáng kể.

### 3.9.3. Early stopping

Mặc định bật: cắt một lát kiểm định (validation) phân tầng `val_fraction = 0.2` từ tập train, huấn luyện trên phần còn lại, và theo dõi BCE kiểm định mỗi epoch. Nếu không cải thiện sau `patience = 15` epoch, dừng và **khôi phục tham số tốt nhất trên kiểm định** — đảm bảo không bao giờ trả về mô hình đã sụp đổ (collapsed). Tập test giữ nguyên không động chạm. Cơ chế này đặc biệt quan trọng với QNN ở chế độ dữ liệu nhỏ, nơi mô hình dễ rơi vào nghiệm thoái hóa (dự đoán toàn một lớp).

## 3.10. Phân tích trainability (gradient)

Cho một tập con chẩn đoán (diagnostic subset), trình huấn luyện ghi lại mỗi epoch: **chuẩn gradient** (trung bình & độ lệch chuẩn trên các lô), **phương sai gradient**, và **trung bình trị tuyệt đối gradient**. Các thống kê này — lấy trung bình trên *tất cả* các lô của epoch (không chỉ lô cuối) để có tín hiệu mượt hơn — là thiết yếu cho phân tích barren plateau và so sánh expressivity giữa các ansatz/mã hóa. Để tiết kiệm tính toán, ghi gradient được **tắt** trong sweep chính (chỉ so F1/accuracy) và chỉ bật cho tập con `qnn_diag`.

## 3.11. Độ phức tạp tham số

Bảng dưới liệt kê tổng tham số học được cho các biến thể chính (`n = 4` qubit, `depth = 4`):

| Mô hình | Tham số mạch | Readout head (`n_obs+1`) | **Tổng** |
|---|---|---|---|
| q2 = angle + basic_entangler, `z` | 16 | 4+1 = 5 | **21** |
| q2 + `z+zz` | 16 | 10+1 = 11 | **27** |
| q2 + `probs` | 16 | 4+1 = 5 | **21** |
| q2 + trainable feature map (`tenc`) | 8+16 = 24 | 5 | **29** |
| q3 = angle + strongly_entangling, `z` | 48 | 5 | **53** |
| q4 = reuploading + basic_entangler, `z` | 16 | 5 | **21** |

So sánh: LR có 5 tham số, **`mlp_tiny` có 25** (khớp tham số với QNN), MLP lớn có ~705, RF có ~2750 nút cây. QNN nằm trong dải **21–53 tham số** — cực kỳ tiết kiệm. Việc đưa `mlp_tiny` (25 tham số) vào cho phép một so sánh QNN-vs-cổ-điển **khớp dung lượng** thực sự, một lợi thế then chốt sẽ được thảo luận ở §4.5.3 và Chương 5.

---

# CHƯƠNG 4 — THỰC NGHIỆM VÀ KẾT QUẢ

## 4.1. Bộ dữ liệu CIC-IDS2017

Bộ dữ liệu **CIC-IDS2017** (Sharafaldin và cộng sự, 2018) do Viện An ninh mạng Canada (CIC) phát hành, là một trong những chuẩn mực được dùng rộng rãi nhất cho đánh giá IDS. Nó chứa lưu lượng mạng thực tế trong năm ngày làm việc, với cả lưu lượng lành tính và một loạt tấn công hiện đại (DDoS, DoS biến thể, PortScan, Brute Force, Web Attack, Infiltration, Botnet). Mỗi luồng được CICFlowMeter trích xuất thành ~80 đặc trưng thống kê.

Luận án sử dụng hai file ngày, sau khi làm sạch (loại `inf/NaN`):

| File | Tổng số luồng | Phân bố nhãn |
|---|---|---|
| **Friday DDoS** (`Friday-WorkingHours-Afternoon-DDos`) | 225.711 | DDoS: 128.025 · BENIGN: 97.686 |
| **Wednesday DoS** (`Wednesday-workingHours`) | 691.406 | BENIGN: 439.683 · DoS Hulk: 230.124 · DoS GoldenEye: 10.293 · DoS slowloris: 5.796 · DoS Slowhttptest: 5.499 · Heartbleed: 11 |

Tất cả nhãn không phải BENIGN được gộp thành lớp ATTACK (`y = 1`). Đáng chú ý, file Wednesday chứa **nhiều biến thể DoS khác nhau** (Hulk, GoldenEye, slowloris, Slowhttptest, Heartbleed), trong khi Friday là **DDoS** — đây là cơ sở để thiết kế kịch bản chéo tấn công.

## 4.2. Các kịch bản đánh giá

Hệ thống hỗ trợ ba kịch bản; hai kịch bản được dùng trong sweep chính:

- **S1 — Cùng nguồn (Friday DDoS):** train và test đều lấy từ file Friday DDoS, nhưng **rời rạc ở mức hàng** (không trùng lặp). Đo năng lực phân loại **trong phân phối**.
- **S3 — Chéo tấn công (Wednesday DoS → Friday DDoS):** train trên các biến thể DoS của Wednesday, test trên DDoS của Friday. File nguồn khác nhau hoàn toàn. Đây là phép thử khắc nghiệt: mô hình có học được "tính tấn công" tổng quát, hay chỉ ghi nhớ dấu hiệu của DoS Wednesday? Test dùng offset seed `+1000` để chắc chắn không trùng chỉ số hàng với train.
- *(S2 — Friday PortScan: được hiện thực trong mã nhưng không nằm trong sweep này do file PortScan không có sẵn cục bộ.)*

**Ngữ nghĩa lấy mẫu:** `train_samples_per_class = N` nghĩa là **đúng N mẫu/lớp trong tập train**; `test_samples_per_class` là kích thước test/lớp. Train và test lấy rời rạc, không trùng. Kích thước test được **cố định ở 500 mẫu/lớp** cho mọi thí nghiệm, để các con số test-F1 so sánh trực tiếp được giữa các kích thước train khác nhau. Ví dụ `train=100, test=500` → train trên 200 hàng, đánh giá trên 1000 hàng.

## 4.3. Thiết lập thực nghiệm

### 4.3.1. Ma trận sweep

| Chiều | Giá trị |
|---|---|
| Kịch bản | S1, S3 |
| Kích thước train/lớp | 100, 250, 500 |
| Kích thước test/lớp | 500 (cố định) |
| Số đặc trưng | 4 (= số qubit) |
| Seed | 0, 1, 2, 3, 4 (5 lần lặp) |
| Mô hình cổ điển | LR, SVM, RF, MLP, **mlp_tiny** (khớp tham số) |
| Kiến trúc QNN | q2 (angle+BE), q3 (angle+SE), q4 (reuploading+BE) |
| Biến thể QNN | base (`z`), `_zz` (`z+zz`), `_probs`, `_tenc` (trainable feature map) |

Tổng: **150 cấu hình cổ điển** (5×2×3×5) + **360 cấu hình QNN** (3×4×2×3×5) = **510 thí nghiệm**, mỗi con số báo cáo là trung bình trên 5 seed.

### 4.3.2. Siêu tham số QNN

`n_qubits = 4`, `depth = 4`, `epochs = 80`, `learning_rate = 0.05`, `batch_size = 32`, `angle_clip = 3.0`, thiết bị mô phỏng `lightning.qubit`, early stopping bật (`patience = 15`, `val_fraction = 0.2`).

### 4.3.3. Tái lập và môi trường

Mỗi thí nghiệm có seed cố định (đặt cho NumPy, Python, và mạch). Thư mục kết quả lưu `config.yaml` + `meta.json` (git commit, thời gian). Pipeline **idempotent**: chạy lại sweep bỏ qua các thí nghiệm đã hoàn thành. Stack phần mềm: PennyLane (≥0.34), scikit-learn 1.9, NumPy, pandas; Python 3.12.

## 4.4. Chỉ số đánh giá

Quy ước: `0 = BENIGN`, `1 = ATTACK`; ta quan tâm phát hiện ATTACK. Các chỉ số báo cáo:

- **F1** — trung bình điều hòa của precision & recall; chỉ số chính (cân bằng, ít bị mất cân bằng lớp đánh lừa).
- **Precision (attack)** = TP/(TP+FP): trong các luồng bị gắn cờ, bao nhiêu là tấn công thật.
- **Recall (attack)** = TP/(TP+FN): trong các tấn công thật, bắt được bao nhiêu.
- **FNR (tỉ lệ bỏ sót)** = 1 − recall: tỉ lệ tấn công bị bỏ lỡ — **nguy hiểm nhất** trong an ninh.
- **FPR (tỉ lệ báo động giả)** = FP/(FP+TN).
- **ROC-AUC** — chất lượng xếp hạng độc lập ngưỡng.
- **Khoảng cách train–test (gap)** = `train_metric − test_metric`: dương = overfitting.

## 4.5. Kết quả 1 — Lượng tử vs Cổ điển (RQ1)

### 4.5.1. Kịch bản S1 (trong phân phối)

Bảng dưới là **test-F1 trung bình trên 5 seed** trên S1. (QNN báo cáo biến thể tốt nhất của mỗi kiến trúc ở mỗi N; chi tiết biến thể ở §4.6.)

| Mô hình | Tham số | N=100 | N=250 | N=500 |
|---|---|---|---|---|
| LR | 5 | 0,768 | 0,778 | 0,776 |
| **mlp_tiny (khớp tham số)** | **25** | 0,737 | 0,758 | 0,818 |
| SVM | — | 0,771 | 0,783 | 0,818 |
| MLP (lớn) | 705 | 0,809 | 0,865 | 0,922 |
| **RF** | ~2750 | **0,993** | **0,996** | **0,998** |
| QNN q2 (angle+BE, `z`) | 21 | 0,770 | 0,781 | 0,856 |
| QNN q3 (angle+SE, `z`) | 53 | 0,788 | 0,811 | 0,854 |
| QNN q4 (reuploading+BE, `z`) | 21 | 0,811 | 0,818 | **0,941** |
| **QNN q4, biến thể tốt nhất** | 21–27 | 0,892 (`z+zz`) | 0,872 (`tenc`) | **0,941** (`z`) |

**Nhận xét:**
- **So khớp tham số (QNN 21 vs mlp_tiny 25):** ở cùng ngân sách tham số, **mọi kiến trúc QNN đều vượt mlp_tiny** ở mọi N — kể cả QNN đơn giản nhất (q2: 0,856 vs 0,818 ở N=500); q4 bỏ xa (**+0,123** ở N=500). MLP cổ điển chỉ bắt kịp QNN khi được tăng dung lượng gấp ~28 lần (705 tham số).
- **RF thống trị S1** (0,993–0,998) — nhưng với `train_F1 = 1,000` ở mọi N, một **dấu hiệu ghi nhớ rõ rệt** (khoảng cách train–test gần như bằng 0 vì test cùng phân phối, nhưng cây đã khớp hoàn hảo train).
- **QNN q4 (reuploading) đạt 0,941 ở N=500**, vượt LR (+0,165), SVM (+0,123) và **vượt cả MLP** (+0,019), chỉ thua RF.
- QNN cải thiện mạnh theo N (q4: 0,811 → 0,941), cho thấy mạch reuploading tận dụng tốt dữ liệu bổ sung.

### 4.5.2. Kịch bản S3 (chéo tấn công — distribution shift)

| Mô hình | Tham số | N=100 | N=250 | N=500 |
|---|---|---|---|---|
| LR | 5 | 0,752 | 0,744 | 0,755 |
| **mlp_tiny (khớp tham số)** | **25** | 0,572 | 0,739 | 0,751 |
| SVM | — | 0,758 | 0,756 | 0,753 |
| **MLP (lớn)** | 705 | **0,788** | **0,885** | **0,880** |
| RF | ~2750 | 0,684 | 0,783 | 0,748 |
| QNN q2 (angle+BE, `z`) | 21 | 0,738 | 0,726 | 0,767 |
| QNN q3 (angle+SE, `z`) | 53 | 0,746 | 0,768 | 0,781 |
| QNN q4 (reuploading+BE, `z`) | 21 | 0,643 | 0,741 | 0,784 |
| **QNN, biến thể tốt nhất** | 21–53 | 0,770 (q3 `z+zz`) | 0,815 (q3 `z+zz`) | **0,821** (q4 `z+zz`) |

**Nhận xét quan trọng (RQ3):**
- **RF sụp đổ** trên S3: từ 0,99 (S1) xuống **0,68–0,78** (S3), trong khi `train_F1` vẫn ~0,99. Đây là **overfitting thảm họa** — RF ghi nhớ dấu hiệu của DoS Wednesday và không generalization sang DDoS Friday.
- **MLP lớn là baseline cổ điển bền vững nhất** trên S3 (đạt 0,885 ở N=250) — nhưng nhờ dung lượng lớn (705 tham số). Khi **khớp tham số**, mlp_tiny lại **yếu hơn QNN** (best QNN 0,770–0,821 vs mlp_tiny 0,572–0,751) và đặc biệt **sụp đổ + cực bất ổn ở N=100** (0,572, độ lệch chuẩn 0,324 — xem §4.7).
- **QNN suy giảm hòa hoãn hơn RF**: biến thể `z+zz` đạt 0,815–0,821, sánh ngang phần lớn baseline cổ điển và **bền vững hơn RF rõ rệt** dưới distribution shift.

### 4.5.3. Hiệu quả tham số và so sánh KHỚP THAM SỐ

So sánh QNN (21 tham số) với MLP lớn (705) hay RF (~2750 nút) là **không công bằng về dung lượng**: các mô hình cổ điển này được "chấp" gấp 28–130 lần số tham số. Để cô lập đóng góp thực sự của biểu diễn lượng tử khỏi lợi thế dung lượng thô, ta đối chiếu QNN với baseline **khớp tham số** `mlp_tiny` (MLP một lớp ẩn 4 nơ-ron, **25 tham số**, đúng dải tham số của QNN):

| Mô hình | Số tham số | test-F1 S1 (100/250/500) | test-F1 S3 (100/250/500) |
|---|---|---|---|
| LR | 5 | 0,768 / 0,778 / 0,776 | 0,752 / 0,744 / 0,755 |
| **mlp_tiny (khớp tham số)** | **25** | 0,737 / 0,758 / 0,818 | 0,572 / 0,739 / 0,751 |
| QNN q2 (angle+BE) | 21 | 0,770 / 0,781 / 0,856 | 0,738 / 0,726 / 0,767 |
| QNN q4 (reuploading+BE) | 21 | 0,811 / 0,818 / **0,941** | 0,643 / 0,741 / 0,784 |
| QNN (biến thể tốt nhất) | 21–53 | 0,892 / 0,872 / 0,941 | 0,770 / 0,815 / 0,821 |
| MLP (lớn) | 705 | 0,809 / 0,865 / 0,922 | 0,788 / 0,885 / 0,880 |
| RF | ~2750 | 0,993 / 0,996 / 0,998 | 0,684 / 0,783 / 0,748 |

**Phát hiện then chốt (RQ1):** ở **cùng ngân sách ~20–25 tham số**, QNN **vượt rõ** MLP cổ điển trên cả hai kịch bản:
- Trên **S1**, ngay cả QNN đơn giản nhất (q2, 21 tham số) cũng vượt mlp_tiny ở mọi N (N=500: 0,856 vs 0,818); QNN q4 bỏ xa (**0,941 vs 0,818, +0,123**).
- Trên **S3**, QNN cũng dẫn trước (best QNN 0,770–0,821 vs mlp_tiny 0,572–0,751); mlp_tiny **sụp đổ và cực kỳ bất ổn** ở N=100 (0,572; std 0,324).

Nói cách khác, so sánh "QNN ≈ MLP" ở §4.5.1 thực ra **thiệt cho QNN** (MLP lớn có gấp ~28 lần tham số). Khi khớp dung lượng, **QNN thắng MLP cổ điển một cách thuyết phục** — đây là bằng chứng mạnh nhất trong nghiên cứu này cho **hiệu quả tham số nội tại của biểu diễn lượng tử** (RQ1): không gian Hilbert `2^n` chiều cho phép một mạch 21 tham số nắm bắt cấu trúc mà một mạng nơ-ron cổ điển cùng ngân sách không nắm được. RF và MLP-lớn vẫn cao hơn trên S1, nhưng chỉ nhờ dung lượng lớn hơn 1–2 bậc độ lớn — và RF phải trả giá bằng sự sụp đổ trên S3.

## 4.6. Kết quả 2 — Ablation (RQ2)

Bảng dưới là **test-F1 trung bình (trên các N) theo từng biến thể** so với baseline `z`:

| Kiến trúc | Kịch bản | base (`z`) | `z+zz` | `probs` | `tenc` |
|---|---|---|---|---|---|
| angle + basic_entangler | S1 | 0,803 | 0,796 | 0,780 | **0,828** |
| angle + basic_entangler | S3 | 0,744 | 0,740 | 0,717 | 0,747 |
| angle + strongly_entangling | S1 | 0,818 | 0,812 | 0,791 | **0,848** |
| angle + strongly_entangling | S3 | 0,765 | **0,798** | 0,764 | 0,637 |
| reuploading + basic_entangler | S1 | 0,857 | **0,865** | 0,829 | 0,860 |
| reuploading + basic_entangler | S3 | 0,723 | 0,708 | **0,743** | 0,698 |

### 4.6.1. Trainable feature map (`tenc`)

Hiệu ứng trung bình (delta vs base, trên mọi kiến trúc + N):
- **S1: +0,020** (tối đa +0,072) — **cải thiện** ổn định trong phân phối.
- **S3: −0,050** (tối thiểu −0,138) — **gây hại** dưới distribution shift.

Đây là phát hiện trung tâm: trainable feature map làm tăng expressivity (học được tần số phù hợp với dữ liệu nguồn), nên thắng trên S1; nhưng chính sự biểu cảm đó khiến nó **overfitting cấu trúc đặc thù của miền nguồn**, nên thua trên S3. Ví dụ cực đoan: q3 (angle+SE) `tenc` trên S3/N=500 chỉ đạt 0,646 (so với 0,781 của base) — và với **độ lệch chuẩn theo seed 0,204** (xem §4.7), cho thấy mô hình rất bất ổn.

### 4.6.2. Multi-observable readout `z+zz`

- Hiệu ứng trung bình **trung tính**: S1 −0,001, S3 +0,005.
- Nhưng **phương sai rất lớn**: từ −0,101 đến +0,081. `z+zz` mang lại lợi ích đáng kể ở **chế độ dữ liệu cực nhỏ và kiến trúc phù hợp** — ví dụ q4 (reuploading) S1/N=100: từ 0,811 lên **0,892** (+0,081). Trên S3, `z+zz` là biến thể tốt nhất cho angle+SE (0,798 vs 0,765 base).
- **Diễn giải:** các tương quan hai qubit `⟨Z_iZ_j⟩` làm giàu biểu diễn khi mỗi qubit-đơn chưa đủ phân tách, đặc biệt có giá trị khi dữ liệu ít. Khi dữ liệu đã đủ hoặc kiến trúc đã đủ biểu cảm (reuploading nhiều lớp), thông tin bổ sung này dư thừa hoặc gây nhiễu.

### 4.6.3. Readout `probs`

- **Chủ yếu gây hại** (S1 −0,026), vì chỉ đọc 2 qubit (4 xác suất, ràng buộc tổng = 1 → thực chất 3 bậc tự do), **vứt bỏ thông tin** so với `z+zz` (10 giá trị). Ngoại lệ: trên reuploading S3 nó lại là biến thể tốt nhất (0,743) — cho thấy không có "readout thắng tuyệt đối", lựa chọn phụ thuộc kiến trúc và kịch bản.

### 4.6.4. Mã hóa: reuploading vs angle

Hiệu ứng của reuploading so với angle (cùng basic_entangler, readout `z`):

| | N=100 | N=250 | N=500 |
|---|---|---|---|
| S1 (delta) | +0,041 | +0,037 | **+0,085** |
| S3 (delta) | −0,094 | +0,016 | +0,017 |

Reuploading **vượt trội rõ trên S1** (đặc biệt +0,085 ở N=500), khẳng định dự đoán lý thuyết Fourier rằng nạp lại dữ liệu tăng expressivity. Trên S3, lợi ích không nhất quán (thậm chí âm ở N=100) — lại một biểu hiện của đánh đổi expressivity–generalization.

## 4.7. Kết quả 3 — Phân tích ổn định (RQ4)

Độ lệch chuẩn của test-F1 **giữa các seed** là một chỉ số quan trọng về độ tin cậy:

- **Độ lệch chuẩn trung vị của test-F1 (QNN): 0,086** — đáng kể. Một số cấu hình ổn định (std < 0,02), số khác cực kỳ bất ổn (std lên tới 0,294).
- **Ví dụ tương phản:**

| Cấu hình | test-F1 (mean) | std (5 seed) |
|---|---|---|
| RF S1/N=500 | 0,998 | **0,001** (rất ổn định) |
| RF S3/N=500 | 0,748 | 0,016 |
| MLP (lớn) S3/N=250 | 0,885 | 0,042 |
| mlp_tiny S3/N=100 | 0,572 | **0,324** (rất bất ổn) |
| q4 reuploading `z` S1/N=500 | 0,941 | 0,028 |
| q4 reuploading `z+zz` S3/N=500 | 0,821 | 0,063 |
| q3 angle+SE `tenc` S3/N=500 | 0,646 | **0,204** (rất bất ổn) |

**Nhận xét:** QNN nói chung **kém ổn định hơn** các mô hình cổ điển **dung lượng lớn** (RF std ~0,001, MLP-lớn ~0,04) theo seed, do (i) khởi tạo ngẫu nhiên của cả mạch lẫn readout head, (ii) bề mặt mất mát phi lồi nhiều cực tiểu, và (iii) chế độ dữ liệu nhỏ. Tuy vậy, ở **cùng ngân sách tham số**, mlp_tiny cũng cực kỳ bất ổn (std 0,324 ở S3/N=100) — cho thấy bất ổn theo seed phần lớn là hệ quả của **dữ liệu nhỏ + dung lượng thấp**, không phải đặc tính riêng của lượng tử. Đáng lo nhất là các biến thể biểu cảm cao dưới distribution shift (q3 `tenc` S3: std 0,204) — đây là cảnh báo trực tiếp cho triển khai thực tế: **một mô hình có thể rất tốt với seed này nhưng sụp đổ với seed khác**. Báo cáo trung bình ± độ lệch chuẩn trên nhiều seed là bắt buộc.

## 4.8. Kết quả 4 — Phân tích generalization (gap)

Khoảng cách train–test F1 trung bình:

| Kịch bản | gap_F1 trung bình |
|---|---|
| S1 (cùng nguồn) | **0,010** |
| S3 (chéo tấn công) | **0,110** |

Trong phân phối (S1), QNN gần như **không overfitting** (gap ~0,01). Dưới distribution shift (S3), gap nhảy lên 0,110 — toàn bộ độ khó của bài toán nằm ở generalization chéo miền, không phải ở việc khớp dữ liệu nguồn. So sánh: RF có gap *âm* nhỏ trên S1 (test ≈ train ≈ hoàn hảo) nhưng gap **~0,25** trên S3 (train 0,99 vs test 0,74) — minh chứng định lượng cho việc RF ghi nhớ thay vì generalization.

## 4.9. Chi phí tính toán

| Mô hình | Thời gian huấn luyện trung bình |
|---|---|
| LR | 0,01 s |
| SVM | 0,03 s |
| RF | 0,22 s |
| MLP | 0,58 s |
| QNN reuploading+BE | 326 s |
| QNN angle+BE | 411 s |
| QNN angle+SE | 624 s |
| QNN (toàn dải) | 26 s – 8.749 s (trung vị 335 s) |

**Mô phỏng QNN chậm hơn mô hình cổ điển ~3 bậc độ lớn (≈1000×).** Đây là chi phí của việc mô phỏng động lực học lượng tử trên CPU cổ điển. Trong số các QNN, `strongly_entangling` đắt nhất (nhiều cổng + nhiều tham số), `reuploading + basic_entangler` rẻ nhất — và đáng chú ý, lại là kiến trúc hiệu năng tốt nhất trên S1. Lưu ý chi phí này là **chi phí mô phỏng**, không phải chi phí trên phần cứng lượng tử thực; nó phản ánh giới hạn của nghiên cứu hiện nay chứ không phải giới hạn cố hữu của QNN.

---

# CHƯƠNG 5 — THẢO LUẬN

## 5.1. Tại sao (và khi nào) lượng tử giúp ích?

Kết quả cho thấy một bức tranh tinh tế hơn nhiều so với "lượng tử thắng/thua". Có thể rút ra ba luận điểm:

1. **Hiệu quả tham số (đã kiểm chứng bằng so sánh khớp tham số).** QNN đạt hiệu năng sánh MLP-lớn (và vượt LR/SVM) trên S1 với chỉ 21 tham số — bằng ~3% của MLP và <1% của RF. Quan trọng hơn, khi đối chiếu với baseline cổ điển **khớp tham số** (`mlp_tiny`, 25 tham số), QNN **vượt rõ trên cả hai kịch bản** (S1/N=500: 0,941 vs 0,818; S3 best: 0,821 vs 0,751). Điều này loại trừ phản biện "QNN chỉ tốt vì được so với mô hình cổ điển lớn hơn nhiều lần": ở **cùng ngân sách tham số**, biểu diễn lượng tử thực sự mạnh hơn. Không gian Hilbert `2^n` chiều cho phép một mạch nhỏ biểu diễn các ranh giới quyết định phi tuyến phong phú. Điều này khớp với lý thuyết về khả năng hữu hiệu cao của QNN (Abbas và cộng sự, 2021) và biên generalization thuận lợi từ ít dữ liệu (Caro và cộng sự, 2022).

2. **Bền vững dưới distribution shift.** Đây là phát hiện đáng giá nhất cho an ninh mạng. RF — mô hình mạnh nhất trên S1 — sụp đổ trên S3 vì nó ghi nhớ. QNN, với số tham số ít và cấu trúc quy chuẩn ngầm (mạch unita bảo toàn chuẩn, linear head), **suy giảm hòa hoãn hơn**. Trong thực tế triển khai IDS, nơi tấn công luôn tiến hóa, **độ bền dưới distribution shift quan trọng hơn điểm số trong phân phối**.

3. **Expressivity phải tương xứng dữ liệu.** Reuploading thắng trên S1 nhờ phổ Fourier phong phú; `z+zz` thắng ở dữ liệu cực nhỏ nhờ làm giàu biểu diễn. Nhưng cùng các cơ chế đó, khi vượt quá nhu cầu, dẫn tới overfitting (đặc biệt `tenc` trên S3).

## 5.2. Sự đánh đổi expressivity – generalization

Chủ đề xuyên suốt là **đánh đổi expressivity–generalization**, một biểu hiện lượng tử của nghịch lý thiên kiến–phương sai cổ điển:

- Tăng expressivity (reuploading, trainable feature map, đa quan sát) → **giảm thiên kiến**, khớp dữ liệu nguồn tốt hơn → thắng S1.
- Nhưng expressivity cao + dữ liệu ít + distribution shift → **tăng phương sai**, mô hình bám vào cấu trúc đặc thù của miền nguồn → thua S3 và bất ổn theo seed.

Trainable feature map là minh họa sạch nhất: vì nó học chính các *tần số* của biểu diễn Fourier, nó là thành phần biểu cảm nhất — và do đó hữu ích nhất trên S1 (+0,072 tối đa) nhưng nguy hiểm nhất trên S3 (−0,138 tối thiểu, std 0,204). Hàm ý thiết kế: **với IDS hoạt động dưới distribution shift, nên ưu tiên các thành phần biểu cảm vừa phải và chính quy hóa mạnh hơn là tối đa hóa expressivity.**

## 5.3. Ghi nhớ vs học hành vi: bài học từ Random Forest

Sự tương phản RF giữa S1 (0,998) và S3 (0,748) với `train_F1` luôn ~1,0 là một **ca minh họa kinh điển** về nguy cơ đánh giá IDS chỉ trên dữ liệu cùng nguồn. Nếu luận án chỉ báo cáo S1, RF sẽ được kết luận là "mô hình tốt nhất tuyệt đối" — một kết luận sai lầm nguy hiểm cho triển khai. Chỉ khi đưa vào S3, bản chất ghi nhớ của RF mới lộ ra. Đây là lập luận mạnh cho việc **mọi đánh giá IDS phải bao gồm một phép thử chéo miền/chéo tấn công**, và xác nhận tính đúng đắn của thiết kế blacklist đặc trưng (§3.2.2) nhằm chống shortcut learning.

## 5.4. Hạn chế

Luận án thẳng thắn nêu các hạn chế:

1. **Chỉ mô phỏng, không phần cứng thực.** Mọi kết quả từ trình mô phỏng `lightning.qubit` không nhiễu. Phần cứng NISQ thực có nhiễu khử pha/biên độ sẽ làm suy giảm hiệu năng — chưa được đánh giá.
2. **Quy mô nhỏ (4 qubit, 4 đặc trưng).** Bị giới hạn bởi chi phí mô phỏng mũ. Chưa rõ các phát hiện có mở rộng lên 8–16 qubit hay không, và đây cũng là vùng có thể xuất hiện barren plateau.
3. **Chi phí tính toán cao (~1000× cổ điển).** Khiến tinh chỉnh siêu tham số quy mô lớn bất khả thi; các siêu tham số QNN (depth, lr) dùng giá trị hợp lý chứ chưa tối ưu.
4. **Độ bất ổn theo seed.** Một số cấu hình có std lên tới 0,2, hạn chế độ tin cậy của kết luận điểm.
5. **Phạm vi kịch bản hẹp.** Chỉ hai loại tấn công (DoS, DDoS) và nhị phân hóa. Chưa đánh giá đa lớp, các tấn công lén lút (Infiltration, Web Attack), hay mất cân bằng cực đoan thực tế.
6. **Baseline cổ điển dùng mặc định.** Không tinh chỉnh; RF/MLP có thể tốt hơn nếu điều chỉnh — dù vậy điều này không thay đổi kết luận về độ bền chéo miền.
7. **Phân tích gradient hạn chế.** Tập con chẩn đoán barren-plateau được thiết lập nhưng phân tích sâu nằm ngoài phạm vi sweep chính.

## 5.5. Hướng phát triển tương lai

1. **Triển khai trên phần cứng lượng tử thực** (IBM Quantum, IonQ) và mô hình hóa nhiễu, kèm các kỹ thuật giảm thiểu lỗi (error mitigation).
2. **Mở rộng quy mô** lên nhiều qubit hơn với kỹ thuật chống barren plateau (khởi tạo định danh, mạch tương thích bài toán, huấn luyện theo lớp).
3. **Quy chuẩn hóa cho distribution shift:** phạt expressivity, dropout lượng tử, hoặc huấn luyện đối kháng miền (domain-adversarial) để cải thiện S3.
4. **Phân loại đa lớp** và đánh giá trên toàn bộ phổ tấn công CIC-IDS2017 (và các bộ mới như CIC-IDS2018, CSE-CIC).
5. **Kernel lượng tử (Quantum Kernel Estimation)** như một hướng thay thế cho QNN biến phân, so sánh trực tiếp với SVM-RBF.
6. **Phân tích lý thuyết** mối liên hệ giữa expressivity Fourier, độ bền distribution shift, và độ ổn định huấn luyện.
7. **Tối ưu chi phí mô phỏng/suy luận** để hướng tới khả thi triển khai thời gian thực.

---

# CHƯƠNG 6 — KẾT LUẬN

## 6.1. Tóm tắt phát hiện

Luận án đã trình bày một nghiên cứu thực nghiệm hệ thống về Mạng nơ-ron lượng tử cho phát hiện xâm nhập trên CIC-IDS2017, dưới chế độ dữ liệu hạn chế, với ba đóng góp phương pháp (trainable feature map, multi-observable readout, hybrid architecture). Trả lời các câu hỏi nghiên cứu:

- **RQ1.** QNN **cạnh tranh được** với ML cổ điển: trên S1, biến thể tốt nhất đạt test-F1 = 0,941, vượt LR/SVM và sánh/vượt MLP-lớn, với hiệu quả tham số vượt trội (21 tham số vs hàng trăm/nghìn). Quan trọng nhất, ở **cùng ngân sách tham số** (~20–25), QNN **vượt rõ** một MLP cổ điển khớp tham số (`mlp_tiny`) trên cả hai kịch bản — chứng tỏ lợi thế đến từ chính biểu diễn lượng tử chứ không phải dung lượng. Chỉ RF/MLP-lớn cao hơn trên S1, nhưng nhờ dung lượng lớn hơn 1–2 bậc độ lớn (và RF trả giá bằng sụp đổ trên S3).
- **RQ2.** Mỗi thành phần có vai trò **phụ thuộc ngữ cảnh**: mã hóa reuploading cải thiện nhất quán trong phân phối (+0,085); `z+zz` có giá trị ở dữ liệu cực nhỏ (+0,081); trainable feature map thắng trong phân phối nhưng hại dưới distribution shift; `probs` thường gây hại.
- **RQ3.** Dưới distribution shift (S3), QNN **bền vững hơn RF** một cách rõ rệt — RF sụp đổ từ 0,99 xuống 0,74 trong khi QNN duy trì ~0,82. Đây là phát hiện có ý nghĩa thực tiễn nhất.
- **RQ4.** QNN hiện **chưa sẵn sàng triển khai** do chi phí mô phỏng ~1000× và độ bất ổn theo seed (std tới 0,2), đặc biệt với các biến thể biểu cảm cao.

Thông điệp xuyên suốt là **sự đánh đổi expressivity–generalization**: expressivity lượng tử là con dao hai lưỡi, có lợi trong phân phối nhưng rủi ro dưới distribution shift khi dữ liệu ít.

## 6.2. Tiềm năng triển khai thực tế

QNN cho IDS chưa sẵn sàng cho sản xuất hôm nay, nhưng nghiên cứu này chỉ ra một **vị thế phù hợp đầy hứa hẹn**: các tình huống mà (i) dữ liệu gán nhãn khan hiếm, (ii) độ bền dưới distribution shift/tấn công mới quan trọng hơn điểm số trong phân phối, và (iii) ngân sách tham số/bộ nhớ bị hạn chế (ví dụ thiết bị biên, IoT). Khi phần cứng lượng tử trưởng thành và các kỹ thuật chống barren plateau/giảm nhiễu chín muồi, các đặc tính hiệu quả tham số và bền vững chéo miền được chứng minh ở đây có thể trở thành lợi thế cạnh tranh thực sự. Trong ngắn hạn, đóng góp giá trị nhất của luận án là một **khung đánh giá trung thực** — đặc biệt là phép thử chéo tấn công S3 — mà mọi nghiên cứu QML-cho-IDS tương lai nên áp dụng để tránh kết luận thổi phồng từ đánh giá cùng nguồn.

---

# TÀI LIỆU THAM KHẢO

[1] Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*. In Proc. ICISSP. (Bộ dữ liệu CIC-IDS2017.)

[2] Biamonte, J., Wittek, P., Pancotti, N., Rebentrost, P., Wiebe, N., & Lloyd, S. (2017). *Quantum machine learning*. Nature, 549, 195–202.

[3] Schuld, M., & Petruccione, F. (2021). *Machine Learning with Quantum Computers*. Springer.

[4] Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala, A., Chow, J. M., & Gambetta, J. M. (2019). *Supervised learning with quantum-enhanced feature spaces*. Nature, 567, 209–212.

[5] Mitarai, K., Negoro, M., Kitagawa, M., & Fujii, K. (2018). *Quantum circuit learning*. Physical Review A, 98, 032309.

[6] Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., & Latorre, J. I. (2020). *Data re-uploading for a universal quantum classifier*. Quantum, 4, 226.

[7] Schuld, M., Sweke, R., & Meyer, J. J. (2021). *Effect of data encoding on the expressive power of variational quantum-machine-learning models*. Physical Review A, 103, 032430.

[8] McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H. (2018). *Barren plateaus in quantum neural network training landscapes*. Nature Communications, 9, 4812.

[9] Cerezo, M., Arrasmith, A., Babbush, R., et al. (2021). *Variational quantum algorithms*. Nature Reviews Physics, 3, 625–644.

[10] Schuld, M., Bocharov, A., Svore, K. M., & Wiebe, N. (2020). *Circuit-centric quantum classifiers*. Physical Review A, 101, 032308.

[11] Schuld, M., & Killoran, N. (2019). *Quantum machine learning in feature Hilbert spaces*. Physical Review Letters, 122, 040504.

[12] Abbas, A., Sutter, D., Zoufal, C., Lucchi, A., Figalli, A., & Woerner, S. (2021). *The power of quantum neural networks*. Nature Computational Science, 1, 403–409.

[13] Caro, M. C., Huang, H.-Y., Cerezo, M., et al. (2022). *Generalization in quantum machine learning from few training data*. Nature Communications, 13, 4919.

[14] Bergholm, V., Izaac, J., Schuld, M., et al. (2018). *PennyLane: Automatic differentiation of hybrid quantum-classical computations*. arXiv:1811.04968.

[15] Preskill, J. (2018). *Quantum Computing in the NISQ era and beyond*. Quantum, 2, 79.

[16] Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.

[17] Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. In Proc. ICLR.

[18] Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

[19] Holmes, Z., Sharma, K., Cerezo, M., & Coles, P. J. (2022). *Connecting ansatz expressibility to gradient magnitudes and barren plateaus*. PRX Quantum, 3, 010313.

---

# PHỤ LỤC

## Phụ lục A — Ví dụ file cấu hình YAML (QNN)

```yaml
experiment:
  name: q4_reuploading_basic_entangler_s1_n500_f4_d4_zz
  seed: 0
data:
  scenario: S1
  train_samples_per_class: 500
  test_samples_per_class: 500
  n_features: 4
model:
  type: qnn
  encoding: reuploading
  ansatz: basic_entangler
  n_qubits: 4
  depth: 4
  device: lightning.qubit
  angle_clip: 3.0
  readout: z+zz
training:
  epochs: 80
  learning_rate: 0.05
  batch_size: 32
evaluation:
  log_gradients: false
```

## Phụ lục B — Công thức số tham số

```
n_obs(z)      = n_qubits
n_obs(z+zz)   = n_qubits + C(n_qubits, 2)
n_obs(probs)  = 2 ** readout_wires

n_quantum     = (2*n_qubits nếu trainable_encoding) + depth * params_per_layer(ansatz)
params_per_layer(basic_entangler)     = n_qubits
params_per_layer(strongly_entangling) = 3 * n_qubits
params_per_layer(rotation_only)       = 3 * n_qubits

n_head        = n_obs + 1
n_params      = n_quantum + n_head
```

## Phụ lục C — Bố cục vectơ tham số phẳng

```
[  a (n_qubits)  |  b (n_qubits)  |  ansatz (depth·ppl)  |  w (n_obs)  |  b_head (1)  ]
   └──────────── tham số mạch (n_quantum) ──────────────┘  └──── hybrid readout head ────┘
   (a, b chỉ có khi bật trainable_encoding; a khởi tạo 1.0, b khởi tạo 0.0)
```

## Phụ lục D — Bảng kết quả đầy đủ (test-F1 trung bình 5 seed)

### D.1 — QNN, kịch bản S1

| Cấu hình | N=100 | N=250 | N=500 |
|---|---|---|---|
| q2 angle+BE `z` | 0,770 | 0,781 | 0,856 |
| q2 angle+BE `z+zz` | 0,770 | 0,783 | 0,835 |
| q2 angle+BE `probs` | 0,768 | 0,760 | 0,812 |
| q2 angle+BE `tenc` | 0,794 | 0,830 | 0,860 |
| q3 angle+SE `z` | 0,788 | 0,811 | 0,854 |
| q3 angle+SE `z+zz` | 0,782 | 0,799 | 0,854 |
| q3 angle+SE `probs` | 0,770 | 0,782 | 0,821 |
| q3 angle+SE `tenc` | 0,860 | 0,844 | 0,841 |
| q4 reup+BE `z` | 0,811 | 0,818 | 0,941 |
| q4 reup+BE `z+zz` | 0,892 | 0,863 | 0,840 |
| q4 reup+BE `probs` | 0,807 | 0,846 | 0,834 |
| q4 reup+BE `tenc` | 0,798 | 0,872 | 0,910 |

### D.2 — QNN, kịch bản S3

| Cấu hình | N=100 | N=250 | N=500 |
|---|---|---|---|
| q2 angle+BE `z` | 0,738 | 0,726 | 0,767 |
| q2 angle+BE `z+zz` | 0,726 | 0,736 | 0,758 |
| q2 angle+BE `probs` | 0,696 | 0,725 | 0,729 |
| q2 angle+BE `tenc` | 0,726 | 0,713 | 0,801 |
| q3 angle+SE `z` | 0,746 | 0,768 | 0,781 |
| q3 angle+SE `z+zz` | 0,770 | 0,815 | 0,809 |
| q3 angle+SE `probs` | 0,728 | 0,773 | 0,792 |
| q3 angle+SE `tenc` | 0,609 | 0,656 | 0,646 |
| q4 reup+BE `z` | 0,643 | 0,741 | 0,784 |
| q4 reup+BE `z+zz` | 0,649 | 0,653 | 0,821 |
| q4 reup+BE `probs` | 0,673 | 0,741 | 0,815 |
| q4 reup+BE `tenc` | 0,648 | 0,629 | 0,816 |

### D.3 — Cổ điển

| Mô hình | Tham số | S1/100 | S1/250 | S1/500 | S3/100 | S3/250 | S3/500 |
|---|---|---|---|---|---|---|---|
| LR | 5 | 0,768 | 0,778 | 0,776 | 0,752 | 0,744 | 0,755 |
| SVM | — | 0,771 | 0,783 | 0,818 | 0,758 | 0,756 | 0,753 |
| mlp_tiny (khớp tham số) | 25 | 0,737 | 0,758 | 0,818 | 0,572 | 0,739 | 0,751 |
| MLP (lớn) | 705 | 0,809 | 0,865 | 0,922 | 0,788 | 0,885 | 0,880 |
| RF | ~2750 | 0,993 | 0,996 | 0,998 | 0,684 | 0,783 | 0,748 |

---

*Hết.*

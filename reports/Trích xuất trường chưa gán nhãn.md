# Dạy mô hình đọc schema, đừng học thuộc prompt

Nhóm nên đổi bài toán học từ "(prompt cố định, tài liệu) → JSON cố định" sang hai lớp tách biệt. Lớp thứ nhất là một mô hình **trích xuất có điều kiện theo schema** (schema-conditioned extraction). Mô hình này chỉ học các fact in trên tài liệu, ví dụ số lượng kèm đơn vị đúng như in: `[{qty: 3, unit: "Pallet"}, {qty: 5, unit: "Carton"}]`. Lớp thứ hai là **lớp rule có version theo khách hàng** trong post-processing; lớp này tính các trường nghiệp vụ suy diễn. Mô hình được huấn luyện trên schema ngẫu nhiên hóa: field dropout, trường vắng mặt với `null` tường minh, đổi tên, mô tả, giá trị ví dụ. Nhờ đó thêm hay bớt trường trở thành việc sửa schema, không phải fine-tune lại. Đề xuất MLM/NSP không chạm vào chỗ hỏng. SSL dạy đọc chữ và bố cục, kỹ năng Gemma 4 và Qwen3.5 đã có. SSL không dạy ánh xạ tên trường → giá trị, và không nguồn nào trong khảo sát cho thấy SSL giúp trích xuất trường chưa từng gán nhãn. Với ví dụ 5 → 7 trường: theo công thức SFT cố định thì có, phải gán nhãn lại và fine-tune lại, vì mô hình học trên tập trường cố định chỉ đạt **3–5 F1** trên trường bị giữ lại ([GenKIE](https://arxiv.org/abs/2310.16131)). Với công thức mới, trường mới khởi đầu ở mức zero-shot, khoảng **67 micro-F1** trên CORD, lên khoảng **90** sau 10 tài liệu có nhãn ([LMDX](https://arxiv.org/abs/2309.10952)). Nó cần thêm một đợt retrain để đạt 95%. Điều đó chấp nhận được, vì dự án chỉ yêu cầu ≥95% sau fine-tune. Thay đổi đến hàng tuần trong giai đoạn go-live, nên không thể chạy một chu kỳ gán nhãn → train → merge → W4A16 → regression cho mỗi thay đổi. Giữa các đợt release, thay đổi đi qua schema, rule và fallback; các đợt retrain định kỳ gộp những nhãn đã được sửa. Hai điều kiện làm các con số có nghĩa. Thứ nhất, test set phải được người kiểm tra, vì nhãn hiện tại do chính Qwen3.6-35B-A3B-FP8 sinh ra. Thứ hai, baseline cần thắng là hệ thống hiện tại của dự án: Qwen3.6 với prompt và post-processing, vốn đã làm zero-shot schema following. Một thí nghiệm field-holdout trên CORD v2 (nhãn do người gán, 18 nhãn đang chưa dùng) đo được cả hai điều này trước khi reference config bị đóng băng.

## MLM và NSP dạy mô hình đọc trang, không dạy nó tìm trường mới

Các mục tiêu self-supervised trong document AI đều nhắm vào ba kỹ năng: đọc chữ, hiểu bố cục và căn chỉnh chữ với ảnh. Họ encoder dùng masked language modeling giữ tọa độ 2-D (LayoutLM), căn chỉnh text–image (LayoutLMv2, LayoutLMv3) hoặc quan hệ hình học (GeoLayoutLM). Họ sinh dùng "đọc toàn bộ trang theo thứ tự" (Donut), "parse screenshot thành HTML" (Pix2Struct) hoặc infilling khối chữ (DocLLM) ([survey Ding et al.](https://arxiv.org/abs/2408.01287)). Không mục tiêu nào dạy ánh xạ từ một tên trường như `tax_code` sang giá trị của nó. Ánh xạ đó chỉ được học khi fine-tune có nhãn. Vì vậy mọi mô hình SSL vẫn giải KIE bằng một tập nhãn cố định. LayoutLM đặt lớp linear + softmax trên từng token cho các key "định nghĩa trước" ([LayoutLM](https://arxiv.org/abs/1912.13318)). Donut thêm special token riêng cho từng trường lúc fine-tune ([Donut](https://arxiv.org/abs/2111.15664)). Trên CORD, khi không có tài liệu huấn luyện nào, LayoutLMv3 và Donut đạt **0.00 micro-F1** ([LMDX](https://arxiv.org/abs/2309.10952)). Ngay cả trong phân phối, phần đóng góp của các mục tiêu SSL bổ sung cho KIE cũng rất nhỏ: thêm MIM và WPA vào LayoutLMv3 chỉ đưa CORD từ **96.27 lên 96.49 F1** ([LayoutLMv3](https://arxiv.org/abs/2204.08387)). Mục tiêu dạng parse cho mức tăng lớn hơn nhiều: bỏ screenshot parsing làm DocVQA của Pix2Struct rơi từ 67.8 xuống 12.2 ([Pix2Struct](https://arxiv.org/abs/2210.03347)). Nhưng các phép đo đó là trên QA và phân loại, với mô hình học đọc gần như từ đầu, không phải trên trường chưa gán nhãn.

Hai thành phần cụ thể của đề xuất còn yếu hơn. **NSP đã bị bỏ từ 2019.** RoBERTa thấy bỏ NSP "khớp hoặc nhỉnh hơn" ở downstream; ví dụ SQuAD 2.0 đạt 78.7 F1 khi có NSP và 79.7 khi không có ([RoBERTa](https://arxiv.org/abs/1907.11692)). ALBERT chỉ ra NSP chủ yếu học nhận biết đổi chủ đề, và mô hình huấn luyện bằng NSP chỉ ngang mức ngẫu nhiên trên bài sentence-order ([ALBERT](https://arxiv.org/abs/1909.11942)). Với một VLM đọc từng tài liệu riêng lẻ, NSP không có dạng tự nhiên nào. **MLM không phải mục tiêu gốc của Gemma 4 hay Qwen3.5.** Dạng tương đương cho decoder là continued pretraining next-token hoặc infilling trên OCR text, hoặc image → transcription. Các nghiên cứu về DAPT/TAPT cho thấy bước này luôn đi *trước* fine-tune có nhãn chứ không thay thế nó, và lợi ích của nó tùy vào khoảng cách domain. ACL-ARC tăng từ 63.0 lên 75.4, còn AGNews đứng yên ở 93.9 ([Gururangan et al.](https://arxiv.org/abs/2004.10964)). Nguy hiểm hơn, với decoder LLM, continued pretraining trên văn bản thô "làm tổn hại nghiêm trọng" khả năng làm theo prompt ([AdaptLLM](https://arxiv.org/abs/2309.09530)). Đó lại đúng là năng lực cần để đọc một schema mới trong prompt. Nếu chạy bằng LoRA, bước này học được ít ([Biderman et al.](https://arxiv.org/abs/2405.09673)). Mỗi lần đổi phân phối còn cần replay và LR re-warming để không quên ([Ibrahim et al.](https://arxiv.org/abs/2403.08763)), và sau đó phải quantize lại W4A16.

Lý do thứ ba là base model đã đọc tài liệu ở quy mô mà một pass SSL nội bộ không thể so. Qwen2.5-VL pretrain trên 4.1 nghìn tỷ token. Dữ liệu này gồm "document omni-parsing" dạng HTML kèm box và thứ tự đọc, và một tập OCR đa ngôn ngữ có tiếng Việt ([Qwen2.5-VL](https://arxiv.org/abs/2502.13923)). Qwen3-VL thêm 30 triệu mẫu OCR nội bộ, cùng dữ liệu parsing từ 3 triệu PDF Common Crawl và 4 triệu tài liệu nội bộ ([Qwen3-VL](https://arxiv.org/abs/2511.21631)). Gemma 4 không công bố dữ liệu tài liệu, nhưng model card liệt kê document/PDF parsing và OCR đa ngôn ngữ ([Gemma 4 model card](https://gemma.ai/models/gemma-4/model-card/)). Chỗ thiếu nằm ở giao diện output, không ở kỹ năng đọc. Trên 1,771 schema chưa thấy của VAREX, NuExtract 2.0 (2B), một mô hình chỉ được tune cho trích xuất theo template, đạt **90.8% exact match**; base Qwen2-VL của chính nó chỉ đạt **9.7%** ([VAREX](https://arxiv.org/abs/2603.15118)). Trong các nguồn khảo sát, kết quả zero-shot trên trường mới luôn đến từ cùng một công thức: generation có điều kiện theo key hoặc schema, cộng huấn luyện có giám sát trên nhiều schema (LMDX, DocLLM, NuExtract).

Trực giác đằng sau đề xuất là tạo tín hiệu học mà không cần nhãn mới, và trực giác đó vẫn đúng; nó chỉ cần đặt đúng chỗ. Với bài toán này, "nhiệm vụ tự giám sát" hiệu quả nhất là **ngẫu nhiên hóa schema trên nhãn sẵn có**. Một tài liệu có 12 trường đã gán nhãn cho ra 4,096 tập con trường (2^12). Mỗi tập con là một cặp (schema, target) hợp lệ, chưa kể các biến thể đổi tên và mô tả, và không tốn nhãn mới nào. Nguồn tín hiệu thứ hai là pseudo-label từ teacher. NuExtract v1 cho LLM sinh template từ 300k đoạn C4, Llama-3-70B điền giá trị, rồi lọc còn 50k mẫu ([NuMind](https://numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction)). UniversalNER cho thấy pre-tune trên dữ liệu tổng hợp rồi fine-tune có giám sát đạt 61.8 F1 out-of-domain, cao hơn chỉ dùng dữ liệu giám sát (57.2) hay chỉ dùng dữ liệu tổng hợp (53.4) ([UniversalNER](https://arxiv.org/html/2308.03279v2)). Dự án đã có nhãn teacher; giới hạn của chúng được bàn ở phần áp dụng bên dưới. Nguồn thứ ba, gần tinh thần SSL nhất, là auxiliary "parse everything": trang → text giữ bố cục hoặc danh sách key-value. LayoutLLM tăng 3.06 điểm trên FUNSD zero-shot nhờ layout-aware pretraining dạng instruction ([LayoutLLM](https://arxiv.org/abs/2404.05225)). DocOwl 1.5 tăng DocVQA từ 72.8 lên 77.7 nhờ parsing ([DocOwl 1.5](https://arxiv.org/abs/2403.12895)). Nhưng chưa ai chứng minh parse-everything giúp trích xuất trường chưa gán nhãn. Vì vậy nó thuộc về một nhánh của thí nghiệm field-holdout, không thuộc kế hoạch chính.

## Trích xuất có điều kiện theo schema là lựa chọn chung của paper và sản phẩm

Các hệ IE tổng quát đều đưa danh sách nhãn hoặc trường vào input; chúng chỉ khác nhau ở cách mã hóa. UIE và InstructUIE dùng danh sách nhãn hoặc "options" ([UIE](https://arxiv.org/abs/2203.12277); [InstructUIE](https://arxiv.org/abs/2304.08085)). GoLLIE viết mỗi nhãn thành một class Python, docstring là guideline ([GoLLIE](https://arxiv.org/html/2310.03668)). UniversalNER hỏi lần lượt từng loại ([UniversalNER](https://arxiv.org/html/2308.03279v2)). GLiNER2 dùng cú pháp `field::type::description` trong encoder ([GLiNER2](https://arxiv.org/html/2507.18546)).

Với output JSON lồng nhau từ ảnh hoặc text giữ bố cục, hai thiết kế gần nhất là LMDX và NuExtract. Trong LMDX, prompt chứa OCR có token tọa độ cùng một JSON schema như `{"foo": "", "bar": [{"baz": []}]}`. Completion giữ đúng key và đúng thứ tự, và ghi rõ `null` hoặc `[]` cho trường vắng ([LMDX](https://arxiv.org/abs/2309.10952)). NuExtract dùng template JSON có kiểu (`verbatim-string`, `string`, `number`, `date-time`, `enum`...). Trường vắng trả `null`/`[]`, và ví dụ in-context là tùy chọn ([NuExtract-2.0-8B](https://huggingface.co/numind/NuExtract-2.0-8B)). NuExtract3 là bản fine-tune của Qwen3.5-4B với đúng giao diện này ([NuExtract3](https://huggingface.co/numind/NuExtract3)), và đã có trong ma trận thí nghiệm dưới tên E17.

Các công thức khác kém phù hợp với tài liệu thương mại. QA từng trường phá vỡ sự gắn kết giữa các trường của một dòng hàng và tốn một lần gọi cho mỗi trường. Open key-value extraction như KVP10k chỉ đạt khoảng **0.61 F1** với baseline LLM fine-tune ([KVP10k](https://arxiv.org/abs/2405.00505)). Còn "parse toàn bộ rồi chọn" chỉ khái quát được cho những trường đã có trong tập được parse.

### Công thức SFT trong kế hoạch dạy mô hình bỏ qua schema

Trong kế hoạch, mỗi doc type có một system prompt cố định, và SFT học (prompt, tài liệu) → JSON cố định. Đó là đủ điều kiện cho **format specialization**. ProMoT fine-tune mT5-XXL trên RTE rồi đưa vào prompt TriviaQA 1-shot. Sau 300 bước, **90% output là "True"/"False"**: format được học trước và nhanh nhất, vì mọi mẫu dùng chung một format ([ProMoT](https://arxiv.org/abs/2211.00635)). Trong dialogue state tracking, mô hình huấn luyện không có schema augmentation "overfit vào tên slot đã thấy" ([arXiv 2502.13310](https://arxiv.org/abs/2502.13310)). Với VLM Llama-3.2-Vision-11B, SFT làm hiệu năng out-of-distribution dưới rule shift rơi từ 80.8% xuống 1.3% trên V-IRL-L ([Chu et al.](https://arxiv.org/abs/2501.17161)).

Huấn luyện một mô hình chung cho mọi doc type giúp được một chút. Nhưng K doc type chỉ cho K schema, nên system prompt hoạt động như mã doc type để tra bảng chứ không như đặc tả cần đọc. So với 13,020 loại thực thể của UniversalNER hay 1,771 schema của VAREX, độ đa dạng đó gần bằng không.

Khi thêm key vào prompt, có ba hệ quả dự kiến. Đây là suy luận từ các nghiên cứu tương tự; chưa ai đo trực tiếp trên VLM KIE:

- mô hình bỏ qua key mới và xuất lại schema cũ;
- mô hình xuất key mới nhưng chép giá trị từ một trường cũ gần nghĩa;
- mô hình vẫn xuất các key đã bị xóa khỏi schema.

### Công thức huấn luyện có ablation hậu thuẫn

Bằng chứng ablation rõ nhất đến từ GoLLIE, UniversalNER, GLiNER và IEPile. Hai đòn bẩy mạnh nhất là **trường vắng mặt với target rỗng** và **giá trị ví dụ trong schema**. Field dropout có lợi ích nhỏ nhưng có ý nghĩa thống kê. Đảo thứ tự, paraphrase guideline và che tên nhãn không có ý nghĩa thống kê trong ablation của GoLLIE. Tỷ lệ trong bảng dưới là phán đoán, cần tune trên validation.

| Kỹ thuật | Bằng chứng | Cài đặt đề xuất |
|---|---|---|
| Trường vắng mặt, target `null`/`[]` (negative sampling) | UniversalNER: zero-shot F1 31.5 khi không có negative, 53.4 khi negative lấy theo tần suất ([UniversalNER](https://arxiv.org/html/2308.03279v2)). GLiNER lấy negative từ nhãn của các mẫu khác trong batch: 53.3 → 60.9 với 50% negative; ở 75% thì recall giảm ([GLiNER](https://arxiv.org/html/2311.08526)) | Mỗi mẫu có 1–3 trường vắng chắc chắn; tỷ lệ negative không quá khoảng 50%. Nguồn negative an toàn được bàn ở phần áp dụng |
| Hard negative: trường anh em dễ nhầm | IEPile dùng từ điển hard negative; bỏ nó làm giảm rõ trên RE/EE ([IEPile](https://arxiv.org/html/2402.14710)) | Ghép các cặp dễ nhầm, kèm mô tả "đừng nhầm với X" |
| Giá trị ví dụ trong schema | GoLLIE: bỏ "representative candidates" làm mất 5.4 F1, mức giảm lớn nhất; bản đầy đủ đạt 55.3 so với baseline 42.3 ([GoLLIE](https://arxiv.org/html/2310.03668)). NuMind chọn ví dụ thay cho mô tả ([NuMind](https://numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction)) | 1–3 giá trị ví dụ mỗi trường, lấy từ field registry |
| Mô tả trường | Guideline giúp GoLLIE sửa các nhãn đa nghĩa. Mô hình UniversalNER học bằng định nghĩa thua 11.8% khi bị hỏi bằng tên | Mô tả 1–2 câu có vị trí và mẫu giá trị; huấn luyện cả dạng có mô tả lẫn dạng chỉ có tên |
| Field dropout: xóa trường khỏi cả schema lẫn target | GoLLIE: −1.3 F1 khi bỏ (p = 0.004). GLiNER: +1.4 điểm out-of-domain | Bỏ 10–50% trường mỗi mẫu |
| Thay đổi số trường mỗi mẫu | IEPile: train với khoảng 20 schema rồi test với 10 hoặc 30 thì giảm "đáng kể" | Phủ toàn dải từ 1 trường đến toàn bộ registry của doc type |
| Đổi tên bằng từ đồng nghĩa hoặc mã vô nghĩa | Schema augmentation trong DST: gain tới 2× trên domain chưa thấy, theo abstract ([arXiv 2411.00150](https://arxiv.org/abs/2411.00150)) | 10–20% mẫu, luôn giữ mô tả |
| Đảo thứ tự, paraphrase system prompt | GoLLIE: không có ý nghĩa thống kê. Cách diễn đạt mới làm giảm hiệu năng của mô hình instruction-tuned ([Sun et al.](https://arxiv.org/abs/2306.11270)) | Rẻ; 5–10 bản paraphrase |
| Tag doc type khi cùng tên trường mang nghĩa khác | Tag tên dataset giải xung đột định nghĩa, +22.0% trên nhãn facility (UniversalNER) | Thêm tag doc type vào prompt |

Một lo ngại hợp lý là ngẫu nhiên hóa schema có thể làm giảm accuracy trên các trường đã huấn luyện. Bằng chứng hiện có cho thấy chi phí nhỏ: GoLLIE đạt 73.3 F1 in-domain, so với 73.0 của baseline không dùng các kỹ thuật này ([GoLLIE](https://arxiv.org/html/2310.03668)). Nhưng đó là IE trên văn bản, ở mức thấp hơn nhiều so với mục tiêu 95%. Vì vậy điều kiện test "chỉ trường đã thấy" trong thí nghiệm ở phần cuối là bắt buộc.

Ví dụ in-context gần như không khả thi với context 21–32K token mỗi tài liệu, và bằng chứng về chúng lẫn lộn. LayIE-LLM thấy F1 trung bình cao nhất khi không có ví dụ: 0.650, so với 0.580–0.595 khi có ví dụ ([LayIE-LLM](https://arxiv.org/abs/2502.18179)). Ngược lại, few-shot prompting nâng F1 trên hóa đơn điện Tây Ban Nha từ 82.92% lên 96.11% ([arXiv 2604.25927](https://arxiv.org/pdf/2604.25927)). Giá trị ví dụ ngắn trong registry, đòn bẩy mạnh nhất của GoLLIE, tốn ít token hơn nhiều.

Hai chi tiết triển khai quan trọng với mô hình dưới 4B. Thứ nhất, VAREX thấy lỗi chính của mô hình nhỏ là "schema echo": mô hình xuất lại cấu trúc schema thay vì giá trị. Lỗi này làm mất 45–65 điểm phần trăm và do `$defs` trong JSON Schema gây ra. Inline các định nghĩa đưa Qwen3-VL-2B từ 27.4% lên 91.8% trên tập bị ảnh hưởng ([VAREX](https://arxiv.org/abs/2603.15118)). Vì vậy template trong prompt nên là JSON có kiểu và mô tả inline, không phải JSON Schema có tham chiếu. Thứ hai, NuMind khuyên tắt thinking mode cho mô hình nhỏ, vì chúng hay rơi vào vòng lặp và sinh JSON không parse được ([NuExtract3](https://huggingface.co/numind/NuExtract3)).

### Sản phẩm IDP đã biến cách làm này thành quy trình

Các sản phẩm IDP thương mại đi tới cùng kết luận và vận hành nó thành quy trình.

Google Document AI Custom Extractor ghi rõ một thang ba bậc. Zero-shot chỉ cần schema, 0 tài liệu, độ chính xác "medium". Few-shot cần 5–10 tài liệu, mức "medium-high". Fine-tune cần 10–50+ tài liệu, mức "high". Theo Google, tên và mô tả trường "ảnh hưởng mạnh tới độ chính xác". Mỗi trường được đánh dấu Extract hoặc Derive, và mỗi processor version giữ một bản chụp schema ([Google Document AI](https://docs.cloud.google.com/document-ai/docs/ce-with-genai)).

Azure Content Understanding khuyên "ưu tiên tinh chỉnh mô tả trường trước khi thêm ví dụ có nhãn". Azure cũng cảnh báo rằng tên và mô tả trường khác ngôn ngữ với tài liệu "có thể làm giảm đáng kể độ chính xác" ([Azure CU best practices](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/concepts/best-practices)).

AWS có hai đường. Textract adapter là fine-tune: cần ít nhất 5 tài liệu train và 5 tài liệu test, và mất 2–30 giờ ([AWS Textract](https://docs.aws.amazon.com/textract/latest/dg/textract-using-adapters.md)). Bedrock Data Automation dùng tối đa 10 tài liệu có nhãn để viết lại mô tả trường mà không đổi trọng số ([AWS What's New](https://aws.amazon.com/about-aws/whats-new/2025/12/bedrock-data-automation-optimization-document-blueprints)).

Các nhà cung cấp này bảo vệ độ chính xác bằng citation, confidence và hàng đợi review. Con số đo được về chi phí của cách đó đáng chú ý. Trên ExtractBench, pipeline mạnh nhất của LlamaIndex đạt precision 94.69% khi không lọc gì. Muốn đạt 97%, phải chuyển khoảng 27% giá trị sang review, và recall rơi xuống 66.48% ([LlamaIndex](https://www.llamaindex.ai/blog/what-makes-an-extraction-confidence-score-useful)). Zero-shot là bước khởi động. Phần còn lại do nhãn được sửa tay, retrain và review đảm nhận.

## Ground truth phải ghi rõ trường nào đã gán nhãn, và một trường mới tốn vài chục đến vài trăm tài liệu

Ground truth cho training theo schema phải thỏa một điều kiện mà ground truth cố định không cần: **mọi trường được yêu cầu phải được gán nhãn đầy đủ trên tài liệu đó**, và trường vắng phải ghi `null`/`[]` tường minh. Nếu không, field sampling sẽ yêu cầu một trường có mặt nhưng chưa được gán nhãn, và mô hình học null giả.

Từ đó có ba quyết định thiết kế.

**Thứ nhất, phân biệt ba trạng thái:** có giá trị; `null` nghĩa là "đã kiểm tra, không có trên tài liệu"; và "chưa gán nhãn". Lưu `annotated_fields` cho từng tài liệu và không bao giờ yêu cầu trường nằm ngoài danh sách đó. Null tường minh có giá trị đo được: trong LMDX, để completion bỏ qua trường vắng thay vì ghi `null` làm mất **6.77 micro-F1** ([LMDX](https://arxiv.org/abs/2309.10952)).

**Thứ hai, một field registry có version cho mỗi doc type.** Mỗi mục trong registry gồm:

- tên và alias;
- mô tả 1–2 câu có gợi ý vị trí và mẫu giá trị, viết bằng ngôn ngữ của tài liệu theo khuyến nghị của Azure;
- kiểu và cardinality;
- chế độ: extract (fact in sẵn) hay compute (tính bằng rule);
- các doc type có trường đó;
- quy tắc chuẩn hóa;
- 1–3 giá trị ví dụ.

Registry vừa là văn bản đưa vào prompt, vừa là guideline cho người kiểm tra nhãn. Như vậy tránh được lỗi GoLLIE gặp khi guideline lệch với cách gán nhãn thực tế ([GoLLIE](https://arxiv.org/html/2310.03668)). Kiểu và quy tắc chuẩn hóa đi vào scorer, giống type-aware matching của VRDU ([VRDU](https://arxiv.org/abs/2211.15421)).

**Thứ ba, provenance.** Mỗi giá trị cần gắn với vị trí trên tài liệu. LMDX xuất định danh đoạn cho từng giá trị, nhờ đó loại được giá trị không có trong tài liệu, với độ chính xác định vị dòng 88–94% ở zero-shot và 98–99% sau fine-tune ([LMDX](https://arxiv.org/abs/2309.10952)). VRDU giải thích vì sao vị trí phải ở mức token: một "total" ghi là "10" không thể khôi phục đúng chỗ nếu "10" còn xuất hiện ở nơi khác ([VRDU](https://arxiv.org/abs/2211.15421)). Các box hiện có của dự án do VLM sinh nên cần được sửa; phần áp dụng bên dưới bàn cách làm.

### Câu trả lời cho ví dụ 5 → 7 trường

Theo công thức SFT cố định: có, phải gán nhãn lại và fine-tune lại, và chi phí lớn hơn vẻ ngoài. Bằng chứng trực tiếp gần nhất là GenKIE. Khi giữ một trong bốn trường SROIE khỏi huấn luyện, mô hình sinh học trên tập trường cố định chỉ đạt **3–5 F1 ở 0-shot** trên trường đó. Sau 5 mẫu là 12–77 F1, sau 10 mẫu là 42–78 F1, so với khoảng 97–98 F1 khi đủ dữ liệu ([GenKIE](https://arxiv.org/abs/2310.16131)). Tệ hơn, target cố định phải đầy đủ. Mọi tài liệu cũ thiếu nhãn của hai trường mới đều không dùng được với schema 7 trường mà không dạy null giả, nên nhóm phải mở lại toàn bộ khoảng 2,000 tài liệu của doc type đó. Với training theo schema và `annotated_fields`, tài liệu cũ vẫn dùng nguyên vẹn với tập 5 trường của chúng; chỉ tập nhỏ mới gán nhãn mang 7 trường.

Sau khi đổi công thức, chi phí của một trường mới tùy thuộc vào trường hợp của nó. Các đường cong LMDX trong bảng dưới là cho toàn bộ schema, đo trên mô hình năm 2023 (PaLM 2-S, Gemini Pro). Áp chúng cho một trường mới trên Gemma 4 hay Qwen3.5 là ngoại suy.

| Trường hợp của trường mới | Nhãn huấn luyện mới | Mức kỳ vọng | Bằng chứng |
|---|---|---|---|
| Đã có trong registry, đã được gán nhãn trên một phần tài liệu và từng xuất hiện trong schema ngẫu nhiên | 0 | Gần mức của trường đã huấn luyện | Hệ quả trực tiếp của thiết kế |
| Trường suy diễn mới hoặc rule đổi, nhưng các fact in sẵn cần dùng đã được trích | 0, chỉ viết và test rule | Bằng accuracy của các fact đầu vào | Phần áp dụng bên dưới |
| Fact in sẵn mới hoàn toàn, chỉ sửa schema và mô tả (zero-shot) | 0, nhưng cần tập test do người kiểm tra | Thấp hơn 95% nhiều: khoảng 66–67 micro-F1 trên CORD; line item trên VRDU chỉ 21–23 F1 | [LMDX](https://arxiv.org/abs/2309.10952) |
| Top-up ở đợt retrain | 5–10 tài liệu | 90.02 micro-F1 trên CORD sau 10 tài liệu; Google xếp "medium-high" | [LMDX](https://arxiv.org/abs/2309.10952); [Google](https://docs.cloud.google.com/document-ai/docs/ce-with-genai) |
| Retrain đủ để nhắm 95% | 50–200+ tài liệu | 91.4 (50), 93.4–93.8 (200), 94.5–95.6 (800 tài liệu) | [LMDX](https://arxiv.org/abs/2309.10952) |
| Mô hình SFT prompt cố định | Gần như gán nhãn lại toàn bộ | 3–5 F1 ở 0-shot, 42–78 F1 sau 10 mẫu | [GenKIE](https://arxiv.org/abs/2310.16131) |

Khi đếm chi phí, hãy **đếm mẫu dương, không đếm tài liệu**. Trong bản CORD 12 trường của repo, `subtotal.discount_price` chỉ có mặt ở 74/1,000 tài liệu và `subtotal.service_price` ở 123/1,000. Vì vậy "50 tài liệu" chỉ cho 4–6 ví dụ dương. Hãy chọn tài liệu có chủ đích, lọc bằng dự đoán zero-shot hoặc từ khóa, và gán nhãn cả những trường hợp vắng thật để đo null precision.

### Exact match làm mục tiêu 95% khắc nghiệt hơn nhiều

Theo định nghĩa của dự án, một mẫu chỉ đúng exact match khi mọi trường đúng và không có trường thừa. Nếu lỗi độc lập với nhau, EM xấp xỉ tích accuracy của các trường. Ví dụ:

- Thêm hai trường, mỗi trường đúng 99%, kéo EM từ 95.1% (0.99^5) xuống 93.2% (0.99^7).
- Trên CORD, trường được làm phẳng thành leaf. Một hóa đơn 5 món có 28 leaf (4 × 5 + 8), nên EM 95% đòi khoảng **99.8% accuracy cho mỗi leaf**.

Lỗi thực tế tương quan theo độ khó tài liệu, nên EM thật cao hơn tích này. Nhưng hướng kết luận không đổi. Một bài báo ICDM 2026 về Qwen3.5-4B cho thấy khoảng cách đó: sau LoRA, mô hình đạt 96.60% F1 nhưng chỉ 91.67% exact match cấp bản ghi; ở zero-shot là 0.758 F1 và 0.473 EM ([arXiv 2606.08051](https://arxiv.org/pdf/2606.08051)).

Định nghĩa EM này cũng ảnh hưởng tới việc đo. Để khẳng định một trường đạt 99% với sai số khoảng ±1 điểm, cần cỡ 300 mẫu dương được người kiểm tra trong tập test (khoảng tin cậy nhị thức). Con số này lớn hơn số mẫu dương mà một tập test 100–200 tài liệu thường có cho một trường hiếm.

## Áp dụng vào dự án: fact học vào trọng số, logic nghiệp vụ nằm trong rule có version

Năm sự kiện về dự án thay đổi khuyến nghị:

1. Thay đổi đến dưới ba dạng: thêm hoặc bớt trường, doc type mới, và **trường suy diễn gắn với logic nghiệp vụ** thay đổi theo khách hàng và theo thời gian.
2. Trong giai đoạn go-live và hot-fix, nhiều khách hàng cùng lúc đổi yêu cầu hàng tuần hoặc hàng tháng. Khi đã ổn định, có thể chỉ đổi hàng năm.
3. Đây là dự án thực nghiệm. Kết quả chỉ lên production khi đã chứng minh, nên ≥95% *sau* fine-tune là đủ.
4. Ground truth chỉ phủ các trường đang bắt buộc. Nhãn và box phần lớn do Qwen3.6-35B-A3B-FP8 sinh ra, với tọa độ đã rescale về 0–1000.
5. Hệ thống hiện tại của dự án là Qwen3.6-35B-A3B-FP8. Hệ thống này xử lý thay đổi chỉ bằng cách sửa prompt và post-processing.

Phần này áp các bằng chứng ở trên vào từng sự kiện.

### Trường suy diễn thuộc về rule layer, không thuộc về trọng số

Lấy chuỗi "3 Pallet, 5 Carton". Trên invoice, trường cần lấy là 3 carton. Trên phiếu xuất kho, cùng chuỗi đó có nghĩa là tổng 8 đơn vị. Ở khách hàng khác, trường cần lấy lại là số pallet. Nếu giá trị cuối cùng được dạy thẳng vào trọng số, cùng một đầu vào sẽ có nhiều target mâu thuẫn nhau theo khách hàng và theo thời gian.

Tài liệu IE cho thấy xung đột định nghĩa như vậy gây hại. IEPile phải bỏ ba trong bốn dataset "person" vì định nghĩa mâu thuẫn ([IEPile](https://arxiv.org/html/2402.14710)). UniversalNER cần tag tên dataset mới gỡ được xung đột, và tag đó tăng 22.0% trên nhãn facility ([UniversalNER](https://arxiv.org/html/2308.03279v2)). Với rule đổi hàng tuần, ngay cả tag theo khách hàng cũng không cứu được: nhãn của tháng trước sẽ sai theo rule của tháng này.

Các sản phẩm thương mại đều tách hai loại trường này:

- Google tách Extract khỏi Derive ([Google](https://docs.cloud.google.com/document-ai/docs/ce-with-genai)).
- Azure Content Understanding tách Extract, Classify và Generate. Training của Azure "hiện không hỗ trợ trường dùng phương thức generate" ([Azure CU](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/analyzer-improvement)).
- Citation của Reducto có thể rỗng khi giá trị được suy ra ([Reducto](https://docs.reducto.ai/configs/extract/citations)).
- NuMind phải tách `verbatim-string` khỏi `string` sau khi bản chỉ-chép-nguyên-văn thất bại với giá trị cần suy luận ([NuExtract-2.0-8B](https://huggingface.co/numind/NuExtract-2.0-8B)).

Từ đó, khuyến nghị của báo cáo là chia làm hai lớp:

- **Mô hình** chỉ học fact in sẵn, giữ nguyên đơn vị và quan hệ như in: `[{qty: 3, unit: "Pallet"}, {qty: 5, unit: "Carton"}]`. Nếu tài liệu ghi rõ quan hệ đóng gói, quan hệ đó cũng là fact cần trích.
- **Lớp rule** trong post-processing tính giá trị nghiệp vụ. Rule nhận đầu vào là fact, doc type và khách hàng, có version, và có unit test.

Cách này có ba lợi ích. Fact in sẵn không đổi theo thời gian, nên nhãn của chúng không bao giờ hết hạn. Ground truth của trường suy diễn được *tính* lại từ fact mỗi khi rule đổi, không phải gán nhãn lại. Còn độ đúng của rule được kiểm bằng test tất định, không phải bằng fine-tune.

Hai trường hợp cần chú ý. Thứ nhất, nhãn hiện tại có thể đã chứa giá trị suy diễn theo logic cũ. Những trường đó phải được chuyển thành fact cộng rule, hoặc loại khỏi target SFT. Thứ hai, một trường suy diễn có thể cần nhìn ảnh mới hiểu được và không tính được từ fact in sẵn. Đó là dấu hiệu registry còn thiếu một fact trung gian; cần thêm fact đó thay vì dạy mô hình giá trị cuối cùng.

### Nhịp thay đổi hàng tuần quyết định kiến trúc release

Mỗi lần retrain là một chu kỳ đầy đủ: gán nhãn và kiểm tra → train → merge → W4A16 → regression eval. Chạy chu kỳ đó cho mỗi thay đổi hàng tuần là không thực tế. Kiến trúc phù hợp có ba thành phần:

- một mô hình theo schema, ổn định giữa các đợt release;
- thay đổi được hấp thụ bằng schema, rule và fallback;
- các đợt **retrain định kỳ để gộp** nhãn đã được sửa: hàng tháng trong go-live, thưa dần khi ổn định.

Vendor cũng đi theo hướng này. Bản preview của Azure Content Understanding "chưng cất" các mẫu có nhãn vào analyzer thay vì giữ chúng làm ví dụ lúc chạy ([Azure CU](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/analyzer-improvement)). Extend cho draft, test, publish, pin và rollback cấu hình theo version, với regression trên evaluation set trước mỗi lần deploy ([Extend](https://www.extend.ai/resources/schema-versioning-tools-document-ai-pipelines)). Google giữ bản chụp schema cho từng processor version ([Google](https://docs.cloud.google.com/document-ai/docs/ce-with-genai)). Với dự án, mỗi bản release là một bộ ba (model version, schema/registry version, rule version), và bộ test field-holdout là cổng chặn trước mỗi lần promote.

| Loại thay đổi | Giữa các đợt release | Ở đợt retrain gộp |
|---|---|---|
| Bỏ trường | Xóa khỏi schema; post-processing bỏ key nếu cần. Không đụng tới mô hình | Không cần gì |
| Đổi logic trường suy diễn theo khách hàng | Sửa rule có version, chạy unit test. Không đụng tới mô hình, miễn là đã trích đủ fact đầu vào | Thêm fact mới nếu rule mới cần |
| Thêm fact in sẵn | Thêm vào registry kèm mô tả. Student trích zero-shot qua schema, hoặc chuyển riêng trường đó cho Qwen3.6 với prompt. Đồng thời tích lũy nhãn đã được người sửa | Gộp nhãn đã sửa vào SFT, có replay dữ liệu cũ, rồi quantize lại và chạy regression |
| Doc type mới | Qwen3.6 với prompt xử lý, như hiện nay, trong lúc tích lũy nhãn đã sửa | Thêm doc type vào SFT |

Doc type mới khó hơn trường mới. Đó là tình huống "schema mới, bố cục mới" của LMDX. VRDU cho thấy khi template chưa thấy, micro-F1 thấp hơn 13–17 điểm so với khi template đã thấy, ngay cả ở 200 tài liệu huấn luyện trên Registration Forms ([VRDU](https://arxiv.org/abs/2211.15421)).

Việc chuyển trường mới cho Qwen3.6 tốn thêm một lần gọi mô hình lớn cho phần thay đổi. Đổi lại, các trường ổn định vẫn chạy trên mô hình nhỏ đã được chứng minh. Đây là phương án của báo cáo này, chưa được đo.

### Không có danh mục trường đầy đủ: registry tăng dần, negative phải an toàn

Với House B/L và các tài liệu phức tạp tương tự, không ai liệt kê được mọi trường nếu thiếu kiến thức nghiệp vụ. Vì vậy ý tưởng "gán nhãn dày mọi trường một lần" phải đổi thành một **registry tăng dần có version**. Chuyên gia nghiệp vụ viết từng mục khi khách hàng cần. Mỗi tài liệu ghi lại *trường nào* đã được gán nhãn, để trạng thái "chưa gán nhãn" không bao giờ bị coi là `null`. Để gợi ý trường mới cho chuyên gia duyệt, có thể dùng teacher hoặc open key-value extraction. Nhưng không nên dùng open KV làm đường trích xuất, vì KVP10k chỉ đạt khoảng 0.61 F1 ([KVP10k](https://arxiv.org/abs/2405.00505)). Lưu lại ảnh và OCR của mọi tài liệu để trường mới có thể được gán nhãn ngược về sau.

Thiếu danh mục đầy đủ cũng làm việc tạo negative cho training null khó hơn. Một trường chưa gán nhãn có thể đang có mặt trên tài liệu. Có hai nguồn negative an toàn:

1. **Trường đã được kiểm tra là vắng** (`null` đã xác minh).
2. **Trường mồi mượn từ doc type khác.** Ví dụ, hỏi số container trên một hóa đơn bán lẻ. Cách này tương tự negative lấy từ nhãn của các mẫu khác trong batch của GLiNER, vốn đã nâng F1 từ 53.3 lên 60.9 ([GLiNER](https://arxiv.org/html/2311.08526)).

Trường mồi chỉ an toàn khi chắc chắn không thể xuất hiện trên doc type đó. "Số hóa đơn" trên phiếu xuất kho chẳng hạn không phải trường mồi an toàn. Trường mồi từ doc type khác là negative dễ. Hard negative, tức trường anh em dễ nhầm, vẫn phải lấy từ các trường đã xác minh là vắng.

### Nhãn do Qwen3.6 sinh đặt trần chất lượng cho student

Nhãn hiện có phần lớn do Qwen3.6-35B-A3B-FP8 sinh, nên SFT trên chúng là chưng cất (distillation). Chưng cất từ nhãn LLM là cách UniversalNER (45,889 mẫu do gpt-3.5 gán nhãn) và GLiNER2 (254,334 mẫu do GPT-4o gán nhãn) học được khả năng theo schema ([UniversalNER](https://arxiv.org/html/2308.03279v2); [GLiNER2](https://arxiv.org/html/2507.18546)). Nhưng các mô hình đó hoạt động ở mức F1 50–60 zero-shot, không phải 95%.

Theo phân tích của báo cáo này, lỗi có hệ thống của teacher sẽ được student học như sự thật. Hệ quả là trần chất lượng của student xấp xỉ chất lượng teacher trên từng trường, trừ khi người sửa nhãn. Đo student bằng nhãn teacher còn cho một con số gây hiểu nhầm: "95%" khi đó chỉ có nghĩa là đồng ý với Qwen3.6 tới 95%. So Qwen3.6 với chính nhãn của nó thì gần như chắc chắn được điểm tối đa.

Khuyến nghị về nhãn:

- **Test set phải do người kiểm tra toàn bộ.** Với trường mới, cần khoảng 300 mẫu dương để khẳng định 99% với sai số ±1 điểm, như đã tính ở trên.
- **Mọi trường mới phải được người kiểm tra.**
- **Nhãn huấn luyện**: kiểm tra một mẫu ngẫu nhiên theo từng trường để ước lượng tỷ lệ lỗi của teacher. Ưu tiên sửa nơi student và teacher bất đồng, hoặc nơi giá trị không tìm thấy trong OCR.

Đây cũng là quy trình "auto-label rồi người sửa" của Textract, Google và Azure ([AWS](https://docs.aws.amazon.com/textract/latest/dg/textract-using-adapters.md); [Google](https://docs.cloud.google.com/document-ai/docs/ce-with-genai); [Azure CU](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/analyzer-improvement)).

Box có thể được sửa rẻ nhờ OCR mà luồng spatial text đã có. Các bước:

1. Đổi tọa độ 0–1000 của VLM về pixel.
2. Tìm các token OCR có text khớp với giá trị sau chuẩn hóa. KVP10k dùng normalized edit distance < 0.2 và IoU > 0.3 làm tiêu chí khớp ([KVP10k](https://arxiv.org/abs/2405.00505)), có thể dùng làm ngưỡng khởi điểm.
3. Snap box về token OCR khớp. Nếu giá trị xuất hiện nhiều lần, tức đúng trường hợp "10" mà VRDU nêu, chọn token gần box của VLM nhất.
4. Giá trị không tìm thấy trong OCR được gắn cờ. Nguyên nhân có thể là lỗi OCR, giá trị suy diễn, hoặc teacher bịa. Các nhãn này được người xem trước tiên.

Bước 4 giống cách NuExtract v1 loại mẫu có giá trị không tìm thấy trong text ([NuMind](https://numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction)). Nhớ rằng nhãn không sửa được lỗi OCR ([Azure CU](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/analyzer-improvement)).

### Baseline cần thắng là Qwen3.6 với prompt và post-processing

Hệ thống hiện tại của dự án đã làm đúng việc mà báo cáo đề xuất dạy cho mô hình nhỏ: đọc một schema trong prompt và trích xuất zero-shot. Nếu thay nó bằng một mô hình nhỏ SFT trên prompt cố định, dự án sẽ đánh đổi khả năng thích ứng mà hệ thống hiện tại đang có lấy chi phí suy luận thấp hơn. Với nhịp thay đổi hàng tuần, đó là một bước lùi.

Tiêu chí thành công vì vậy có hai vế:

- Student đạt ≥95% trên các trường đã huấn luyện, đo trên test set do người kiểm tra.
- Trên các phép sửa schema (trường bị giữ lại, đổi tên, trường vắng), student không tệ hơn Qwen3.6 quá một ngưỡng mà nhóm chốt trước.

Nếu student chỉ đạt vế thứ nhất, nó chỉ dùng được cho các doc type đã ổn định, và Qwen3.6 vẫn phải gánh mọi thay đổi.

## SFT prompt cố định, calibration W4A16 và GRPO có thể nới rộng hoặc thu hẹp khoảng cách

### LoRA không miễn nhiễm với format specialization

LoRA giảm quên nhưng không loại bỏ được nó. Biderman et al. thấy LoRA "học ít hơn và quên ít hơn" full fine-tuning ([Biderman et al.](https://arxiv.org/abs/2405.09673)). Tuy vậy, mức quên vẫn tăng theo power law với số bước và số tham số được huấn luyện, và early stopping không tránh được ([Kalajdzievski](https://arxiv.org/abs/2401.05605)). LoRA còn tạo ra "intruder dimensions" gắn với việc quên phân phối pretraining ([Shuttleworth et al.](https://arxiv.org/abs/2410.21228)). Mô hình nhỏ rủi ro hơn: khoảng cách giữa nhãn đã thấy và chưa thấy của GoLLIE giảm từ 7.1 F1 ở 7B xuống 3.3 ở 34B ([GoLLIE](https://arxiv.org/html/2310.03668)). Vì vậy E2B và Qwen3.5-2B là các điểm yếu nhất.

Các biện pháp rẻ:

- **Learning rate nhỏ hơn**: giảm đáng kể suy giảm năng lực chung mà vẫn giữ được hiệu năng domain ([Lin et al. 2025](https://arxiv.org/abs/2509.20758)).
- **Trộn khoảng 1K mẫu instruction chung**, vì năng lực chung bão hòa quanh mức đó ([Dong et al.](https://arxiv.org/abs/2310.05492)).
- **Giữ checkpoint instruction-tuned làm base** ([Luo et al.](https://arxiv.org/abs/2308.08747)).
- **Early-stop theo accuracy trên trường bị giữ lại**, thay vì theo loss trong phân phối.
- **Quét hệ số λ khi merge** (W₀ + λ·BA, λ ∈ {0.5, 0.75, 1.0}). Đây là một dạng weight interpolation; model averaging đã cho Pareto tốt nhất giữa chuyên môn và năng lực chung ([Lin et al. 2023](https://arxiv.org/abs/2309.06256)). Riêng với LoRA thì chưa có bằng chứng trực tiếp.

Với các đợt retrain gộp, replay dữ liệu của các đợt trước là bắt buộc. Nếu không, mỗi đợt sẽ quên một phần những gì đợt trước đã học.

### Calibration trên prompt cố định thu hẹp mô hình lần thứ hai

W4A16 với group size 128 gần như không mất mát trên mô hình 7B+ và benchmark tiếng Anh ngắn ([Kurtic et al.](https://arxiv.org/abs/2411.02355)). Nhưng dự án nằm đúng ở những vùng mà kết luận đó yếu đi:

- **Mô hình nhỏ:** mô hình nhỏ sụt mạnh hơn ở 4-bit, và AWQ tốt hơn GPTQ khi chỉ lượng tử trọng số ([Lee et al.](https://arxiv.org/abs/2409.11055)).
- **GPTQ overfit tập calibration** vì dựa vào tái tạo theo Hessian. Khi calibration lệch domain, perplexity của AWQ chỉ tăng 0.5–0.6, so với 2.3–4.9 của GPTQ; AWQ cũng chỉ cần 16 chuỗi calibration so với 192 ([AWQ](https://arxiv.org/abs/2306.00978)). Đổi ngôn ngữ calibration làm GPTQ dao động tới 3.52 perplexity, so với khoảng 0.35 của AWQ ([Chimoto et al.](https://arxiv.org/abs/2601.18306)).
- **Ngôn ngữ không phải tiếng Anh:** metric tự động đánh giá thấp thiệt hại. Ở tiếng Nhật, mức sụt 1.7% trên task tự động tương ứng 16.0% trong đánh giá của người ([Marchisio et al.](https://arxiv.org/abs/2407.03211)).
- **Context dài:** 4-bit mất tới 59% trên task context dài hơn 64K token, nặng hơn với input không phải tiếng Anh ([Mekala et al.](https://arxiv.org/abs/2505.20276)). Context 21–32K của dự án ngắn hơn ngưỡng đó, nhưng chưa ai đo ở độ dài này.

Rủi ro riêng của dự án là sự cộng dồn. SFT đã thu hẹp mô hình về một prompt cố định. Sau đó GPTQ được calibration trên chính các prompt cố định đó, tức split `calibration` hiện tại, và khớp phần bù lỗi vào cùng phân phối hẹp. Chưa có paper nào đo liệu PTQ có nới rộng khoảng cách giữa trường đã thấy và trường mới hay không. Mỗi đợt retrain gộp cũng phải quantize lại, nên phép đo này cần chạy ở mọi đợt.

Khuyến nghị:

- Dùng AWQ làm mặc định, hoặc GPTQ với calibration trộn.
- Dùng 256–512 mẫu "đại diện cho use case" như LLM Compressor gợi ý ([LLM Compressor](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-dataset/)), ở độ dài thật và có chat template.
- Tỷ lệ khởi điểm: khoảng một nửa là mẫu task với schema ngẫu nhiên theo registry version mới nhất, một phần tư là instruction tiếng Việt chung, một phần tư là chat đa ngôn ngữ. Đây là phán đoán, không phải công thức đã được kiểm chứng.
- Tạo 2–3 tập calibration để đo phương sai, vì kết quả từng task có thể dao động lớn giữa các tập calibration trong khi perplexity gần như đứng yên ([Williams & Aletras](https://arxiv.org/abs/2311.09755)).
- Dùng exact match có dấu trên các trường tên và địa chỉ làm canary cho tiếng Việt.

Checkpoint QAT của Gemma 4 ([Google](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)) không giữ được lợi ích QAT sau khi merge LoRA, trừ khi QAT được làm lại sau SFT. Đây là suy luận, liên quan trực tiếp tới E5/E6.

### GRPO khái quát tốt nhất khi SFT trước nó chưa overfit

GRPO là giai đoạn phù hợp nhất để giữ khả năng khái quát, với một điều kiện. Trên Llama-3.2-Vision-11B dưới rule shift, RL cải thiện out-of-distribution (+11.0% trên V-IRL-L), trong khi SFT sụp (−79.5%) ([Chu et al.](https://arxiv.org/abs/2501.17161)). RL's Razor cho thấy GRPO đạt mức tăng trên task mới mà gần như không quên. Forward KL giữa base và mô hình đã tune dự đoán được mức quên, với R² = 0.71 trên LLM ([RL's Razor](https://arxiv.org/abs/2509.04259)). Chen et al. thấy cùng xu hướng trên Llama và Qwen ([Retaining by Doing](https://arxiv.org/abs/2510.18874)). Điều kiện là SFT phải đi trước để cố định format: không có SFT, mọi run RL end-to-end trong Chu et al. đều thất bại. Nhưng **RL không cứu được một checkpoint SFT đã overfit** ([Chu et al.](https://arxiv.org/abs/2501.17161)).

Với Workflow 6, điều này có nghĩa:

- SFT nhẹ trên schema ngẫu nhiên, early-stop theo trường bị giữ lại.
- GRPO lấy một schema ngẫu nhiên cho mỗi prompt.
- Scorer hiện có chỉ chấm các *fact* được yêu cầu, không chấm giá trị sau rule.
- LoRA là đủ cho GRPO, kể cả ở rank 1, vì RL chỉ hấp thụ khoảng 1 bit mỗi episode ([Thinking Machines](https://thinkingmachines.ai/blog/lora/)).

Hai lưu ý về reward (suy luận của báo cáo này). Thứ nhất, nhiều trường hiếm, ví dụ `discount_price` chỉ có ở 7.4% tài liệu CORD. Reward dạng field accuracy vì thế sẽ thưởng cho việc luôn trả `null`, nên cần phạt riêng null sai và phạt riêng giá trị bịa. Thứ hai, GRPO không loại bỏ nhu cầu nhãn: reward kiểm chứng được vẫn cần ground truth, và nếu ground truth là nhãn teacher chưa sửa thì GRPO tối ưu sự đồng ý với Qwen3.6.

Bằng chứng "RL quên ít" đến từ full fine-tuning trên LLM văn bản. Với LoRA trên VLM thì chưa được kiểm chứng, nên E18/E19 cần thêm các điều kiện test field-holdout.

## Thí nghiệm field-holdout trên CORD v2 trả lời các câu hỏi trước khi có dữ liệu riêng

Chưa có benchmark công khai nào đo trường bị giữ lại trong cùng một doc type cho VLM. VRDU và DocILE giữ lại template, LMDX giữ lại cả schema, còn SVRD không nói rõ zero-shot nghĩa là gì ([VRDU](https://arxiv.org/abs/2211.15421); [DocILE](https://arxiv.org/abs/2302.05658); [SVRD](https://arxiv.org/abs/2306.03287)). Cũng chưa ai fine-tune một VLM instruction-tuned hiện đại trên một schema cố định rồi đo trên trường mới thêm vào prompt.

CORD v2 có hai ưu điểm cho thí nghiệm này. Thứ nhất, nhãn do người gán, nên tránh được vấn đề nhãn teacher. Thứ hai, CORD có 30 nhãn ngữ nghĩa ([LayoutLMv3](https://arxiv.org/abs/2204.08387)) nhưng dự án chỉ dùng 12. Khoảng 18 nhãn còn lại (`menu.sub_nm`, `menu.discountprice`, `total.menuqty_cnt`, `total.emoneyprice`, `sub_total.othersvc_price`...) là những "trường mới" đã có sẵn nhãn.

Cách dựng dữ liệu:

1. Dựng lại target từ `gt_parse` gốc; manifest đã trỏ về `naver-clova-ix/cord-v2`.
2. Giữ nguyên split tài liệu hiện có (seed 42), theo quy tắc không chia lại.
3. Việc đầu tiên là đếm tần suất của 18 nhãn chưa dùng, vì một số có thể quá hiếm.
4. Mỗi kết quả kèm khoảng tin cậy bootstrap theo tài liệu, vì test chỉ có 100 tài liệu.

| Vòng xoay | Trường bị giữ lại (F_new) | Kiểu khó | Mẫu dương ước tính trong 100 tài liệu test |
|---|---|---|---|
| R1 | `total.change_price`, `subtotal.tax_price` | Fact đơn có key in sẵn. Các trường "hàng xóm" dễ nhầm (`cash_price`, `service_price`) vẫn nằm trong tập đã thấy | Khoảng 62 và 45 |
| R2 | `menu.unit_price` | Trường con trong line item | Khoảng 36 tài liệu, hơn 70 instance |
| R3 | `total.menuqty_cnt`, `menu.sub_nm` hoặc nhãn khác trong 18 nhãn chưa dùng | Trường dạng tổng hợp hoặc lồng sâu, mới hoàn toàn với dự án | Cần đếm trước |

Thiết kế huấn luyện là ma trận 2 × 2: công thức SFT (prompt cố định hay schema ngẫu nhiên) nhân với có hay không bước SSL. Bên cạnh đó có ba baseline zero-shot. Mọi nhánh chỉ thấy F_seen khi huấn luyện và dùng chung reference config. Bước SSL được hiểu ở dạng decoder: continued pretraining next-token trên spatial text, hoặc image → transcription, trên tài liệu train. Text của giá trị bị giữ lại được phép xuất hiện ở đây, vì production cũng thấy text đó. Điều bị cấm là bất kỳ target nào ghép giá trị đó với tên hoặc mô tả của trường bị giữ lại.

| Nhánh | Huấn luyện | Câu hỏi được trả lời |
|---|---|---|
| Z0 | Qwen3.6-35B-A3B-FP8 với prompt theo schema (R0) | Hệ thống hiện tại làm được tới đâu: đây là mức khả năng thích ứng cần giữ |
| Z1 | Base zero-shot với prompt theo schema: Gemma 4 E2B/E4B, Qwen3.5-4B | Mức sàn của mô hình nhỏ |
| Z2 | NuExtract3 zero-shot (E17) | Một mô hình chuyên trích xuất theo template làm được gì mà không cần dữ liệu của nhóm |
| A | SFT prompt cố định trên F_seen (kế hoạch hiện tại) | Mô hình có bỏ qua key mới không |
| B | SFT schema ngẫu nhiên trên F_seen | Công thức đề xuất có thắng không |
| A+S | SSL rồi A | Đề xuất MLM/NSP, ở dạng decoder, có tạo ra khả năng trích trường mới không |
| B+S | SSL rồi B | SSL có cộng thêm gì trên nền schema ngẫu nhiên không |
| B+P | B + auxiliary "parse everything" | Phiên bản mạnh nhất của ý tưởng self-supervised |

Cần chạy ít nhất cả Gemma 4 E2B và Qwen3.5-4B, vì xếp hạng base model đảo chiều giữa các benchmark. Trên benchmark nội bộ của NuMind (tự báo cáo), gemma-4-E4B-it đạt 0.538, cao hơn Qwen3.5-4B với 0.417 ([NuExtract3](https://huggingface.co/numind/NuExtract3)). Trên IDP Leaderboard thì ngược lại: Qwen3.5-4B 74.5, Gemma-4-E4B-it 55.0 ([IDP Leaderboard](https://benchmarking.nanonets.com/)).

Mỗi checkpoint được test trên cùng các tài liệu, với sáu cách yêu cầu schema:

| Điều kiện | Schema yêu cầu | Đo gì |
|---|---|---|
| T1 | Chỉ F_seen | Regression trên trường đã thấy; điều kiện tiên quyết để chấp nhận công thức B |
| T2 | Chỉ F_new | Khả năng trích trường mới |
| T3 | Trộn, ví dụ 5 trường đã thấy + 2 trường mới | Đúng kịch bản 5 → 7 |
| T4 | Trường đã thấy nhưng đổi tên hoặc paraphrase, kèm mô tả | Tách độ nhạy với tên khỏi độ mới của khái niệm |
| T5 | Trường có ground truth là null, cộng trường mồi mượn từ doc type khác | Tỷ lệ hallucination |
| T6 | Trường suy diễn tính bằng rule từ fact, ví dụ tổng số lượng từ `menu.count` | Tách accuracy của fact khỏi độ đúng của rule |

Metric dùng scorer và postprocess hiện có:

- accuracy theo trường;
- tỷ lệ key mới có xuất hiện, và tỷ lệ key cũ hoặc key không được yêu cầu vẫn xuất hiện;
- false positive trên trường vắng;
- group-level F1 cho menu item theo kiểu KIEval ([KIEval](https://arxiv.org/abs/2503.05488)), vì so list theo vị trí vẫn là câu hỏi mở trong T-L06;
- EM trên tập trường được yêu cầu;
- parse rate;
- khoảng cách giữa trường đã thấy và trường mới.

Từ nhánh tốt nhất, chạy thêm hai thứ. Thứ nhất là đường cong retrain gộp: 0, 5, 10, 25, 50, 100 tài liệu có nhãn F_new, có replay, mỗi mức 3 seed như VRDU. Đường cong này đo cả trường mới lẫn regression trên trường cũ. Thứ hai, chạy mọi điều kiện test qua các giai đoạn:

1. adapter BF16;
2. merge với λ ∈ {0.5, 0.75, 1.0};
3. GRPO;
4. W4A16 với AWQ;
5. W4A16 với GPTQ calibration chỉ trên task;
6. W4A16 với GPTQ calibration trộn.

Metric quyết định là mức sụt trên trường bị giữ lại từ BF16 sang W4A16, so với mức sụt trên trường đã thấy.

Luật quyết định nên chốt trước khi chạy:

- B ngang A trên T1 trong phạm vi nhiễu, và vượt rõ A trên T2/T3: chuyển reference config sang schema ngẫu nhiên trước khi đóng băng M3.
- A+S không hơn A trên T2: bỏ đề xuất MLM/NSP.
- Ở W4A16, B tụt quá xa Z0 trên T2–T5: student chỉ dùng cho doc type đã ổn định, còn thay đổi vẫn đi qua Qwen3.6.

CORD là hóa đơn tiếng Indonesia với 100 tài liệu test, nên thí nghiệm này cho bằng chứng về cơ chế, không cho con số cho tài liệu tiếng Việt. Bản lặp lại trên dữ liệu riêng làm được ngay với ground truth hiện có. Giữ 2 trong số các trường đang bắt buộc của một doc type khỏi huấn luyện, và dùng một test set do người kiểm tra.

### Các câu hỏi còn mở mà nhóm cần chốt

| Câu hỏi | Vì sao nó quyết định thiết kế | Mặc định đề xuất |
|---|---|---|
| Ai sở hữu lớp rule theo khách hàng, rule được version và test thế nào, và gắn với version schema ra sao? | Logic nghiệp vụ rời khỏi trọng số thì phải có chủ, có test, có version | Nằm trong phạm vi T-L05, có unit test cho từng khách hàng |
| Có trường suy diễn nào không tính được từ fact in sẵn không? | Nếu có, registry đang thiếu một fact trung gian | Rà toàn bộ trường suy diễn hiện có trước khi viết T-L02 |
| Ngân sách người kiểm tra là bao nhiêu? | Không có test do người kiểm tra thì mọi con số 95% chỉ là mức đồng ý với teacher | Test kiểm tra toàn bộ; mọi trường mới đều kiểm tra; huấn luyện kiểm tra theo mẫu |
| Giữa các đợt release, trường mới được phục vụ bằng gì? | Quyết định chi phí suy luận và độ trễ của thay đổi | Student zero-shot nếu T2 đạt ngưỡng, nếu không thì chuyển cho Qwen3.6 |
| Nhịp retrain gộp là bao lâu? | Mỗi đợt là một chu kỳ train → merge → W4A16 → regression | Hàng tháng trong go-live, thưa dần khi ổn định |
| 95% tính trên fact hay trên giá trị sau rule, theo doc type hay theo khách hàng? | EM là tích theo trường; rule có lỗi riêng của nó | Báo cáo riêng fact, giá trị sau rule, trường cũ và trường mới |
| Mô tả trường viết bằng tiếng Việt hay tiếng Anh? | Azure cảnh báo lệch ngôn ngữ làm giảm accuracy | A/B test trên validation |
| Một adapter cho mọi doc type hay mỗi doc type một adapter? | Adapter riêng là cấu hình dễ format specialization nhất | Giữ một adapter chung như quyết định hiện tại |

### Thay đổi cụ thể trong kế hoạch

| Hạng mục | Thay đổi | Lý do |
|---|---|---|
| T-L02 (guideline ground truth) | Thêm registry tăng dần có version, chế độ extract/compute, cách biểu diễn fact in sẵn (số lượng kèm đơn vị như in), null ba trạng thái, `annotated_fields`, cờ nguồn nhãn (teacher hay người đã kiểm tra), snap box về OCR | Phải xong trước M2; sửa sau thì phải gán nhãn lại |
| T-L05 (postprocess) | Mở rộng thành lớp rule theo khách hàng có version cho trường suy diễn | Logic đổi hàng tuần không được nằm trong trọng số |
| T-L03 (dữ liệu riêng) | Test set do người kiểm tra; kiểm tra lỗi teacher theo từng trường; loại giá trị suy diễn cũ khỏi target SFT | Nhãn teacher đặt trần chất lượng và làm phép đo vòng tròn |
| `docs/formats.md` và scorer | Prompt = instruction chung + template có kiểu; scorer chỉ chấm trường được yêu cầu, báo riêng fact, giá trị sau rule, trường cũ và trường mới | Cùng một định dạng cho SFT, GRPO và đánh giá |
| Reference config (M3) | Quyết định công thức SFT trước khi đóng băng | Sau khi đóng băng, đổi config buộc chạy lại mọi experiment kế thừa (quy tắc 2) |
| Ma trận thí nghiệm | Thêm hàng riêng cho field-holdout (ví dụ E20), có Z0 = Qwen3.6 làm baseline, không dùng ID DEV | Quy tắc 7: run DEV không bao giờ được tính là kết quả |
| E17 (NuExtract3) | Nâng từ P2 lên P1, làm baseline zero-shot | Là mô hình công khai có đúng giao diện đề xuất, trên base Qwen3.5-4B |
| E4/T-D04 và E19 | Thêm điều kiện field-holdout; so AWQ với GPTQ calibration task và GPTQ calibration trộn | Đo xem PTQ có nới rộng khoảng cách trường mới không |
| E18 (GRPO) | Schema ngẫu nhiên cho mỗi prompt; reward trên fact được yêu cầu; phạt riêng null sai và giá trị bịa | RL giữ khái quát tốt hơn SFT nhưng có thể bị khai thác reward bằng null |
| Quy trình release | Bộ ba (model, registry, rule) có version; bộ test field-holdout là cổng chặn; retrain gộp định kỳ có replay | Thay đổi hàng tuần không thể kéo theo retrain mỗi lần |
| Đề xuất MLM/NSP | Bỏ NSP; giữ SSL dưới dạng các nhánh A+S, B+S, B+P | Kiểm chứng giả thuyết với chi phí nhỏ thay vì bác bỏ bằng lập luận |

## Kết luận

Bằng chứng cho thấy vấn đề nằm ở chỗ khác với giả định ban đầu, và các sự kiện về dự án dời nó thêm một lần nữa. Khả năng trích xuất trường mới không nằm trong biểu diễn do pretraining tạo ra. Nó nằm trong giao diện mà fine-tune dạy: mô hình chỉ đọc schema nếu, trong lúc huấn luyện, đọc schema là cách duy nhất để trả lời đúng. Nhưng phần thay đổi nhiều nhất trong dự án không phải là trường mới, mà là logic nghiệp vụ đổi theo khách hàng và theo tuần. Phần đó không nên vào trọng số. Chia hệ thống thành hai lớp là quyết định quan trọng nhất: mô hình trích fact in sẵn, rule có version tính giá trị nghiệp vụ. Phần lớn các thay đổi hàng tuần khi đó trở thành sửa schema hoặc rule, chỉ các fact mới thật sự mới cần nhãn, và nhãn đó chỉ phải chờ tới đợt retrain gộp tiếp theo.

Mục tiêu "không bao giờ phải gán nhãn lại" nên được thay bằng ba cam kết đo được. Fact được gán nhãn một lần và được người kiểm tra, vì nhãn của chúng không hết hạn. Mỗi fact mới tốn vài chục đến vài trăm tài liệu đã sửa, và được gộp ở đợt retrain kế tiếp. Mỗi bản release phải không tệ hơn Qwen3.6 trên bộ test field-holdout, ở chính độ chính xác W4A16 sẽ triển khai. Ẩn số lớn nhất là lượng tử hóa và GRPO sẽ nới rộng hay thu hẹp khoảng cách giữa trường đã thấy và trường mới; chưa tài liệu nào trả lời câu hỏi này. Thí nghiệm CORD v2, với nhãn do người gán, sẽ là bằng chứng đầu tiên, và nó đủ rẻ để chạy trước khi reference config bị đóng băng.

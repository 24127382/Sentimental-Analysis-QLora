# Financial Sentiment Analyzer (LoRA / QLoRA)

Refactor notebook thành các module nhỏ, giữ bài toán Financial PhraseBank: `negative`, `neutral`, `positive`.

## Cấu trúc

- `config.py`: model/dataset ID, label mapping và cấu hình chung.
- `data.py`: load dataset, split 80/10/10, tokenize.
- `model.py`: `load_model_from_huggingface(...)` để load model/tokenizer từ Hub hoặc local path; `load_akiya_vyre_model(...)` để load checkpoint `Akiya-Vyre/QLora-LLama-Sentimental-Analysis`; hỗ trợ 4-bit và LoRA adapter.
- `train.py`: `train_qlora(...)`, `train_full_finetune(...)`, metric Accuracy + Macro F1.
- `inference.py`: hàm dự đoán trả nhãn và xác suất.
- `app.py`: Streamlit demo.
- `requirements.txt`: dependencies.

## Cài đặt

Python 3.10+ khuyến nghị. Tạo virtual environment trước khi cài.

```bash
pip install -r requirements.txt
```

LLaMA-2-7B trên Hugging Face là model gated: cần được cấp quyền truy cập và đăng nhập/token hợp lệ. **Không hardcode token trong source code.** Có thể đặt biến môi trường:

```bash
# macOS/Linux
export HF_TOKEN="hf_..."
# PowerShell
$env:HF_TOKEN="hf_..."
```

Bạn cũng có thể nhập token trong ô **Hugging Face token** ở sidebar của app.
Token phải thuộc tài khoản đã được chấp thuận quyền truy cập
`meta-llama/Llama-2-7b-hf`; chỉ tạo token là chưa đủ để vượt qua gated repo.

## Train QLoRA

QLoRA yêu cầu GPU CUDA tương thích. Trên Colab, bật GPU rồi chạy:

```python
from train import train_qlora
trainer, test_metrics = train_qlora(
    model_id="meta-llama/Llama-2-7b-hf",
    output_dir="./artifacts/financial-llama-lora",
    epochs=3,
)
print(test_metrics)
```

Đầu ra LoRA thường là adapter + cấu hình, không phải bản sao đầy đủ của base model. Khi inference, load base model rồi truyền `adapter_id` trỏ tới thư mục adapter hoặc repo adapter.

## Train full fine-tuning

```python
from train import train_full_finetune
trainer, test_metrics = train_full_finetune(
    model_id="meta-llama/Llama-2-7b-hf",
    output_dir="./artifacts/financial-llama-full",
)
print(test_metrics)
```

**Lưu ý:** full fine-tuning LLaMA-2-7B rất tốn VRAM/RAM; chỉ chạy nếu GPU đủ mạnh. Để demo nhanh, nên dùng checkpoint đã fine-tune hoặc một model nhỏ hơn.

## Chạy Streamlit

Sau khi đã có model fine-tuned:

```bash
streamlit run app.py
```

App inference load base `meta-llama/Llama-2-7b-hf` rồi gắn LoRA adapter
`Akiya-Vyre/QLora-LLama-Sentimental-Analysis`, đúng với cách checkpoint được
export từ notebook QLoRA. Cần `HF_TOKEN` có quyền truy cập base model gated.

Trong sidebar:
- `Model ID / local path`: đường dẫn checkpoint local hoặc model repo đã fine-tune hoàn chỉnh.
- `Optional LoRA adapter ID/path`: nếu dùng adapter LoRA, đặt `Model ID` là base model và điền adapter path/ID.
- Bật 4-bit nếu máy có CUDA và muốn tiết kiệm VRAM.

Có thể đổi model demo bằng biến môi trường `INFERENCE_MODEL_ID`. Model gated cần `HF_TOKEN`.

## Một số lưu ý kỹ thuật

1. Notebook ban đầu có token Hugging Face viết trực tiếp trong code. Token đã xuất hiện trong file nên hãy **revoke token đó trên Hugging Face và tạo token mới**, rồi dùng biến môi trường/Secrets.
2. Financial PhraseBank `sentences_allagree` là tập con theo mức đồng thuận cao; split 80/10/10 được tạo từ split `train` mà dataset trả về.
3. Label mapping được khai báo rõ `negative=0, neutral=1, positive=2`. Hãy xác nhận thứ tự label của dataset và checkpoint trước khi so sánh metric hoặc dùng model đã fine-tune.
4. Streamlit chỉ tải model khi bấm Analyze lần đầu và cache tài nguyên model trong phiên chạy.
5. Với model lớn, không nên load base LLaMA-2 7B trực tiếp chỉ để thử giao diện nếu máy không có GPU phù hợp.

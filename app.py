import os
import streamlit as st
from inference import predict_sentiment
from model import load_akiya_vyre_model

st.set_page_config(page_title="Financial Sentiment Analyzer", page_icon="📈", layout="centered")
st.title("📈 Financial Sentiment Analyzer")
st.caption("Phân loại cảm xúc văn bản tài chính bằng mô hình sequence-classification trên Hugging Face.")

with st.sidebar:
    st.header("Model settings")
    use_4bit = st.checkbox("Load in 4-bit (CUDA only)", value=False)
    hf_token = st.text_input(
        "Hugging Face token",
        value=os.getenv("HF_TOKEN", ""),
        type="password",
        help="Token cần quyền truy cập meta-llama/Llama-2-7b-hf.",
    )
    st.caption("App dùng base LLaMA-2 và adapter QLoRA Akiya-Vyre.")

@st.cache_resource(show_spinner="Đang tải model...")
def get_model(use_4bit, token):
    return load_akiya_vyre_model(use_4bit=use_4bit, token=token or None)

text = st.text_area(
    "Financial text",
    value="The company reported strong earnings and increased its revenue forecast.",
    height=150,
    placeholder="Paste a financial news sentence or market comment..."
)
if st.button("Analyze sentiment", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Vui lòng nhập văn bản trước.")
    else:
        try:
            with st.spinner("Đang phân tích..."):
                tokenizer, model = get_model(use_4bit, hf_token)
                result = predict_sentiment(text, tokenizer, model)
            st.subheader(f"Prediction: {result['label'].upper()}")
            st.metric("Confidence", f"{result['confidence']:.1%}")
            st.write("Class probabilities")
            st.bar_chart(result["probabilities"])
            st.caption("Confidence là xác suất softmax của lớp được chọn, không đồng nghĩa mô hình đã được hiệu chuẩn.")
        except Exception as exc:
            st.error(f"Không thể chạy inference: {exc}")
            if "gated repo" in str(exc).lower() or "401" in str(exc):
                st.info(
                    "Hãy xin quyền truy cập Llama-2 trên Hugging Face, tạo User Access Token "
                    "có quyền đọc (read), rồi nhập token ở sidebar hoặc đặt biến môi trường HF_TOKEN."
                )
            else:
                st.info("Kiểm tra HF_TOKEN và cấu hình GPU/quantization.")

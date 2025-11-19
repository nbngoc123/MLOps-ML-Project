import gradio as gr
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv
import httpx # Thêm thư viện httpx

# Tải biến môi trường (nếu có)
load_dotenv()

# Cấu hình URL của backend API
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api") # Mặc định là localhost

# ========================
# 1. PHÂN LOẠI CẢM XÚC (Gọi API Backend)
# ========================
async def sentiment_classification(text):
    if not text or not text.strip():
        return "Vui lòng nhập bình luận."
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/sentiment/predict",
                json={"text": text}
            )
            response.raise_for_status() 
            result = response.json()
            return result.get("sentiment", "Không xác định")
    except httpx.RequestError as e:
        return f"Lỗi kết nối đến backend: {e}"
    except httpx.HTTPStatusError as e:
        return f"Lỗi từ backend: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Lỗi không xác định: {e}"


# ========================
# 2. PHÂN LOẠI EMAIL
# ========================
def email_classification(email_text):
    if not email_text or not email_text.strip():
        return "Vui lòng nhập nội dung email."
    text = email_text.lower()
    if any(kw in text for kw in ["spam", "quảng cáo", "khuyến mãi", "giảm giá"]):
        return "Thư rác"
    elif any(kw in text for kw in ["hỗ trợ", "support", "giúp", "trợ giúp"]):
        return "Hỗ trợ khách hàng"
    elif any(kw in text for kw in ["đơn hàng", "order", "giao hàng", "mã đơn"]):
        return "Đơn hàng"
    else:
        return "Khác"


# ========================
# 3. OCR HÓA ĐƠN (MOCK - KHÔNG DÙNG TESSERACT)
# ========================
def ocr_invoice(image):
    if image is None:
        return "Vui lòng tải lên hình ảnh hóa đơn."
    
    # MOCK: Trả về văn bản mẫu khi có ảnh
    mock_invoice_text = """
CỬA HÀNG ĐIỆN MÁY XYZ
Địa chỉ: 123 Đường ABC, Quận 1, TP.HCM
Hotline: 1900 1234

HÓA ĐƠN BÁN HÀNG
Mã hóa đơn: HD20251116001
Ngày: 16/11/2025

STT | Mô tả             | SL | Đơn giá    | Thành tiền
----|-------------------|----|------------|------------
1   | Điện thoại XYZ    | 1  | 12.990.000 | 12.990.000
2   | Ốp lưng silicon   | 1  | 99.000     | 99.000

Tổng cộng: 13.089.000 VNĐ
Giảm giá: 0 VNĐ
Khách thanh toán: 13.089.000 VNĐ

Cảm ơn quý khách!
    """.strip()
    
    return mock_invoice_text


# ========================
# 4. TÓM TẮT VĂN BẢN (Mock đơn giản)
# ========================
def text_summarization(text):
    if not text or not text.strip():
        return "Vui lòng nhập văn bản."
    if len(text) < 100:
        return "Văn bản quá ngắn để tóm tắt."
    sentences = text.split('. ')
    summary = '. '.join(sentences[:2])
    return summary + ("..." if len(sentences) > 2 else ".")


# ========================
# 5. CHATBOT HỖ TRỢ (Rule-based)
# ========================
def customer_chatbot(message, history):
    msg = message.lower().strip()
    if not msg:
        return "Vui lòng nhập tin nhắn."
    if any(g in msg for g in ["xin chào", "chào", "hi"]):
        return "Chào bạn! Tôi là trợ lý ảo. Bạn cần hỗ trợ gì ạ?"
    elif "sản phẩm" in msg:
        return "Bạn muốn hỏi về sản phẩm nào? (điện thoại, laptop, sách...)"
    elif any(k in msg for k in ["đơn hàng", "order", "mã đơn"]):
        return "Vui lòng cung cấp mã đơn hàng (VD: DH12345) để tôi kiểm tra."
    elif "bảo hành" in msg:
        return "Sản phẩm được bảo hành 12 tháng. Vui lòng giữ hóa đơn."
    elif any(t in msg for t in ["cảm ơn", "thanks", "ok"]):
        return "Rất vui được hỗ trợ bạn! Chúc một ngày tốt lành!"
    else:
        return "Xin lỗi, tôi chưa hiểu. Bạn có thể hỏi về: sản phẩm, đơn hàng, bảo hành..."


# ========================
# 6. GỢI Ý SẢN PHẨM (Rule-based)
# ========================
def product_recommendation(product_name):
    if not product_name or not product_name.strip():
        return "Vui lòng nhập tên sản phẩm."
    p = product_name.lower()
    suggestions = {
        "điện thoại": "Ốp lưng, sạc dự phòng, tai nghe Bluetooth",
        "laptop": "Chuột không dây, túi chống sốc, bàn phím cơ",
        "sách": "Đèn đọc sách, đánh dấu trang, kệ sách nhỏ",
        "quần áo": "Tất, dây lưng, túi xách thời trang",
        "giày": "Tất thể thao, xi đánh giày, lót giày êm"
    }
    for key, rec in suggestions.items():
        if key in p:
            return f"Gợi ý: {rec}"
    return "Không có gợi ý cụ thể. Xem thêm sản phẩm bán chạy!"


# ========================
# 7. PHÂN TÍCH XU HƯỚNG (Keyword count - Mock)
# ========================
def product_trend_analysis(keywords):
    if not keywords or not keywords.strip():
        return "Vui lòng nhập từ khóa."
    kw_list = [k.strip().lower() for k in keywords.split(",") if k.strip()]
    if not kw_list:
        return "Danh sách từ khóa trống."

    trend_db = {
        "điện thoại": "Tăng trưởng 25% so với tháng trước",
        "laptop": "Ổn định, tăng nhẹ 5%",
        "sách": "Tăng cao vào mùa tựu trường",
        "quần áo": "Thay đổi theo mùa, hiện tại: áo khoác",
        "giày": "Xu hướng giày thể thao đang hot",
        "tai nghe": "True wireless dẫn đầu thị trường"
    }

    result = []
    for kw in kw_list:
        trend = trend_db.get(kw, "Không có dữ liệu xu hướng")
        result.append(f"• **{kw.capitalize()}**: {trend}")
    return "\n".join(result)


# ========================
# 8. DỊCH PHẢN HỒI (Mock cố định)
# ========================
def translate_feedback(text, target_lang="Vietnamese"):
    if not text or not text.strip():
        return "Vui lòng nhập phản hồi."
    text = text.lower().strip()
    en_to_vi = {
        "hello": "Xin chào",
        "thank you": "Cảm ơn bạn",
        "good product": "Sản phẩm tốt",
        "fast delivery": "Giao hàng nhanh",
        "bad quality": "Chất lượng kém",
        "recommend": "Khuyên dùng",
        "expensive": "Đắt",
        "cheap": "Rẻ"
    }
    if target_lang == "Vietnamese":
        for eng, vie in en_to_vi.items():
            if eng in text:
                return vie
        return f"[Dịch] {text.capitalize()}"
    else:
        return "Chỉ hỗ trợ dịch sang tiếng Việt."


# ========================
# 9. TẠO NỘI DUNG QUẢNG CÁO (Template)
# ========================
def generate_ad_content(product, discount, cta):
    if not all([product, discount, cta]):
        return "Vui lòng điền đầy đủ thông tin."
    return f"""
**SIÊU KHUYẾN MÃI HÔM NAY**
Mua **{product}** – **{discount}**!
{cta}
Số lượng có hạn – Đặt ngay!
    """.strip()


# ========================
# 10. TẠO BÁO CÁO CSV (Template + Mock Data)
# ========================
def generate_csv_report(report_name, extra_data):
    if not report_name.strip():
        return "Vui lòng nhập tên báo cáo."
    
    header = "ID,Tên sản phẩm,Số lượng,Giá (VNĐ)"
    mock_data = [
        "1,Điện thoại XYZ,12,12990000",
        "2,Laptop Pro,8,28990000",
        "3,Sách Python,20,149000",
        "4,Ốp lưng,50,99000"
    ]
    
    extra = ""
    if extra_data.strip():
        extra = f"\n--- Dữ liệu bổ sung ---\n{extra_data.strip()}"
    
    return f"**BÁO CÁO: {report_name}**\n\n{header}\n" + "\n".join(mock_data) + extra


# ========================
# GRADIO INTERFACE – 10 USE CASES
# ========================

with gr.Blocks(title="NexusML - 10 AI Use Cases", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # NexusML Platform
        ### **10 AI Use Cases cho Doanh nghiệp – Đơn giản, Hiệu quả, Triển khai nhanh**
        """
    )

    # === 1. Phân loại cảm xúc ===
    with gr.Tab("1. Phân loại cảm xúc bình luận"):
        with gr.Row():
            with gr.Column(): inp1 = gr.Textbox(label="Bình luận", lines=3, placeholder="Sản phẩm rất tốt, giao nhanh...")
            with gr.Column(): out1 = gr.Textbox(label="Kết quả", interactive=False)
        gr.Button("Phân loại").click(sentiment_classification, inp1, out1)

    # === 2. Phân loại email ===
    with gr.Tab("2. Phân loại Email"):
        with gr.Row():
            with gr.Column(): inp2 = gr.Textbox(label="Nội dung Email", lines=5)
            with gr.Column(): out2 = gr.Textbox(label="Loại Email", interactive=False)
        gr.Button("Phân loại").click(email_classification, inp2, out2)

    # === 3. OCR Hóa đơn (MOCK) ===
    with gr.Tab("3. Nhận dạng văn bản hóa đơn (Mock)"):
        with gr.Row():
            with gr.Column(): inp3 = gr.Image(type="pil", label="Tải lên hóa đơn (bất kỳ ảnh nào)")
            with gr.Column(): out3 = gr.Textbox(label="Văn bản trích xuất (Mock)", lines=8, interactive=False)
        gr.Button("Trích xuất").click(ocr_invoice, inp3, out3)

    # === 4. Tóm tắt văn bản ===
    with gr.Tab("4. Tự động tóm tắt văn bản"):
        with gr.Row():
            with gr.Column(): inp4 = gr.Textbox(label="Văn bản dài", lines=7)
            with gr.Column(): out4 = gr.Textbox(label="Tóm tắt", interactive=False)
        gr.Button("Tóm tắt").click(text_summarization, inp4, out4)

    # === 5. Chatbot ===
    with gr.Tab("5. Chatbot hỗ trợ khách hàng"):
        gr.ChatInterface(
            customer_chatbot,
            examples=["Xin chào", "Đơn hàng của tôi đâu?", "Sản phẩm có bảo hành không?"],
            title="Hỗ trợ 24/7",
            type="messages"
        )

    # === 6. Gợi ý sản phẩm ===
    with gr.Tab("6. Gợi ý sản phẩm"):
        with gr.Row():
            with gr.Column(): inp6 = gr.Textbox(label="Sản phẩm đã xem", placeholder="VD: điện thoại")
            with gr.Column(): out6 = gr.Textbox(label="Sản phẩm gợi ý", interactive=False)
        gr.Button("Gợi ý").click(product_recommendation, inp6, out6)

    # === 7. Phân tích xu hướng ===
    with gr.Tab("7. Phân tích xu hướng"):
        with gr.Row():
            with gr.Column(): inp7 = gr.Textbox(label="Từ khóa (cách nhau bởi dấu phẩy)", placeholder="điện thoại, laptop")
            with gr.Column(): out7 = gr.Textbox(label="Xu hướng", interactive=False)
        gr.Button("Phân tích").click(product_trend_analysis, inp7, out7)

    # === 8. Dịch phản hồi ===
    with gr.Tab("8. Dịch phản hồi khách hàng"):
        with gr.Row():
            with gr.Column(): 
                inp8 = gr.Textbox(label="Phản hồi (tiếng Anh)", lines=2, placeholder="good product, fast delivery")
                lang8 = gr.Dropdown(["Vietnamese"], value="Vietnamese", label="Ngôn ngữ đích")
            with gr.Column(): out8 = gr.Textbox(label="Kết quả dịch", interactive=False)
        gr.Button("Dịch").click(translate_feedback, [inp8, lang8], out8)

    # === 9. Tạo quảng cáo ===
    with gr.Tab("9. Tạo nội dung quảng cáo"):
        with gr.Row():
            ad_p = gr.Textbox(label="Sản phẩm", value="Điện thoại XYZ")
            ad_d = gr.Textbox(label="Ưu đãi", value="giảm 20%")
            ad_c = gr.Textbox(label="Kêu gọi hành động", value="Mua ngay hôm nay!")
            ad_out = gr.Textbox(label="Nội dung quảng cáo", interactive=False)
        gr.Button("Tạo").click(generate_ad_content, [ad_p, ad_d, ad_c], ad_out)

    # === 10. Báo cáo CSV ===
    with gr.Tab("10. Tạo báo cáo CSV"):
        with gr.Row():
            with gr.Column():
                r_name = gr.Textbox(label="Tên báo cáo", value="Báo cáo bán hàng tháng 11")
                r_data = gr.Textbox(label="Dữ liệu bổ sung (CSV)", lines=3, placeholder="5,Tai nghe,30,490000")
            with gr.Column(): r_out = gr.Textbox(label="Nội dung báo cáo", interactive=False)
        gr.Button("Tạo báo cáo").click(generate_csv_report, [r_name, r_data], r_out)


# ========================
# KHỞI CHẠY
# ========================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  # Đặt share=True để tạo link công khai (nếu dùng Colab hoặc local có ngrok)
    )

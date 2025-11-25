import gradio as gr
import os
import pandas as pd
import numpy as np
import httpx
import asyncio
import time 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from dotenv import load_dotenv
from utils import analyze_advanced_dashboard

import matplotlib
matplotlib.use('Agg')

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Thiết lập thư mục tạm cho Gradio
GRADIO_TEMP_DIR = "/tmp/gradio"
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR
os.makedirs(GRADIO_TEMP_DIR, exist_ok=True) # Tạo thư mục nếu chưa có

# ========================
# Utils: Call API
# ========================

async def call_api_post(endpoint, json_data):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(f"{BACKEND_URL}{endpoint}", json=json_data)
            res.raise_for_status()
            return res.json()
    except Exception as e:
        return {"error": str(e)}


async def send_csv_batch(endpoint, file_obj):
    if file_obj is None:
        return pd.DataFrame({"Lỗi": ["Chưa chọn file"]})

    try:
        file_path = file_obj.name if hasattr(file_obj, "name") else file_obj

        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, "text/csv")}
                res = await client.post(f"{BACKEND_URL}{endpoint}", files=files)

        res.raise_for_status()
        result = res.json()
        return pd.DataFrame(result.get("data", []))

    except Exception as e:
        return pd.DataFrame({"Lỗi": [str(e)]})


# ========================
# LOGIC DASHBOARD
# ========================


    
# ========================
# Wrappers & Helpers
# ========================

def run_async(func, *args):
    """Chạy hàm async đơn lẻ"""
    return asyncio.run(func(*args))

def run_batch_and_export(async_func, file_obj):
    """
    MỚI: Wrapper xử lý Batch + Xuất file CSV
    1. Chạy hàm async để lấy DataFrame
    2. Lưu DataFrame ra file CSV tạm
    3. Trả về (DataFrame, FilePath)
    """
    # 1. Chạy logic AI (Async -> Sync)
    df = asyncio.run(async_func(file_obj))
    
    # 2. Lưu ra file CSV
    if df is not None and not df.empty:
        # Tạo tên file dựa trên thời gian để không bị trùng
        filename = f"export_{int(time.time())}.csv"
        filepath = os.path.join(GRADIO_TEMP_DIR, filename)
        
        # Lưu file (utf-8-sig để Excel hiển thị đúng tiếng Việt)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return df, filepath
    
    return df, None


# ========================
# Sentiment
# ========================

async def sentiment_single(text):
    if not text.strip(): return "⚠️ Vui lòng nhập nội dung."
    result = await call_api_post("/sentiment/predict", {"text": text})

    if "error" in result:
        return f"❌ {result['error']}"

    return f"Cảm xúc: {result['sentiment'].upper()} (Conf: {result['confidence']:.2%})"


async def sentiment_batch(file):
    if file is None:
        return pd.DataFrame({"Lỗi": ["Chưa chọn file"]})
    return await send_csv_batch("/sentiment/predict_batch", file)


# ========================
# Email
# ========================

async def email_single(text):
    if not text.strip(): return "⚠️ Nội dung trống."

    result = await call_api_post("/email/predict", {"text": text})

    if "error" in result:
        return f"❌ {result['error']}"

    status = "🚨 SPAM" if result["is_spam"] else "✅ Hợp lệ"
    return f"{status} (Conf: {result['confidence']:.2%})"


async def email_batch(file):
    return await send_csv_batch("/email/predict_batch", file)


# ========================
# Topic
# ========================

async def topic_single(text):
    if not text.strip():
        return "⚠️ Nhập văn bản."

    result = await call_api_post("/topic/predict", {"text": text})

    if "error" in result:
        return f"❌ {result['error']}"

    return f"Chủ đề: {result['topic']}"


async def topic_batch(file):
    return await send_csv_batch("/topic/predict_batch", file)


# ========================
# RecSys
# ========================

async def recsys_single(uid, pid, desc):
    payload = {"user_id": uid, "product_id": pid, "description": desc or ""}
    result = await call_api_post("/recsys/predict", payload)

    if "error" in result:
        return f"❌ {result['error']}"

    note = "(Cold Start)" if result.get("is_cold_start") else ""
    return f"⭐ Rating: {result['predicted_rating']}/5 {note}"


async def recsys_batch(file):
    return await send_csv_batch("/recsys/predict_batch", file)


# ========================
# Trend
# ========================

async def trend_analysis(file):
    return await send_csv_batch("/trend/predict", file)


# ========================
# GIAO DIỆN (UPDATED)
# ========================

with gr.Blocks(title="NexusML AI Platform", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🚀 NexusML AI Platform")
    gr.Markdown("Hệ thống phân tích dữ liệu AI đa mô hình.")

    with gr.Tabs():

        # ==========================
        # TAB 1: SENTIMENT
        # ==========================
        with gr.Tab("1. Cảm xúc (Sentiment)"):
            gr.Markdown("### 😐 Phân tích sắc thái văn bản (Tích cực/Tiêu cực)")
            
            s_in = gr.Textbox(label="Nhập văn bản kiểm tra nhanh")
            s_out = gr.Textbox(label="Kết quả", interactive=False)
            gr.Button("Phân tích ngay").click(
                lambda x: run_async(sentiment_single, x),
                inputs=s_in, outputs=s_out
            )

            gr.Markdown("---")
            gr.Markdown("### 📂 Xử lý hàng loạt (Batch Processing)")
            
            # --- HƯỚNG DẪN SENTIMENT ---
            with gr.Accordion("📝 Hướng dẫn định dạng file CSV (Xem chi tiết)", open=False):
                gr.Markdown("""
                **Yêu cầu file CSV:**
                - Encoding: `UTF-8`
                - Cần có **1 trong các cột sau**: `text`, `comment`, hoặc `content`.
                
                **Ví dụ nội dung file:**
                ```csv
                text,id
                "Sản phẩm dùng rất tốt",1
                "Giao hàng chậm quá",2
                ```
                """)
            # ---------------------------

            with gr.Row():
                s_file = gr.File(label="Tải lên CSV")
                s_download = gr.File(label="Tải xuống kết quả")
            
            s_df = gr.DataFrame(label="Xem trước dữ liệu")
            
            gr.Button("Chạy Batch & Export").click(
                lambda f: run_batch_and_export(sentiment_batch, f),
                inputs=s_file, outputs=[s_df, s_download]
            )

        # ==========================
        # TAB 2: EMAIL
        # ==========================
        with gr.Tab("2. Email Spam"):
            gr.Markdown("### 📧 Phát hiện Email Rác (Spam/Ham)")
            
            e_in = gr.Textbox(label="Nội dung Email")
            e_out = gr.Textbox(interactive=False, label="Kết quả")
            gr.Button("Kiểm tra").click(
                lambda x: run_async(email_single, x),
                inputs=e_in, outputs=e_out
            )

            gr.Markdown("---")
            gr.Markdown("### 📂 Xử lý hàng loạt")

            # --- HƯỚNG DẪN EMAIL ---
            with gr.Accordion("📝 Hướng dẫn định dạng file CSV", open=False):
                gr.Markdown("""
                **Yêu cầu file CSV:**
                - Encoding: `UTF-8`
                - Cần có **1 trong các cột sau**: `text`, `content`, `body`, hoặc `email`.
                
                **Ví dụ nội dung file:**
                ```csv
                text,subject
                "Chúc mừng bạn trúng thưởng iPhone...", "Quà tặng"
                "Lịch họp dự án vào ngày mai", "Công việc"
                ```
                """)
            # -----------------------

            with gr.Row():
                e_file = gr.File(label="Tải lên CSV")
                e_download = gr.File(label="Tải xuống kết quả")
            
            e_df = gr.DataFrame()
            
            gr.Button("Chạy Batch & Export").click(
                lambda f: run_batch_and_export(email_batch, f),
                inputs=e_file, outputs=[e_df, e_download]
            )

        # ==========================
        # TAB 3: TOPIC
        # ==========================
        with gr.Tab("3. Chủ đề (Topic)"):
            gr.Markdown("### 🏷️ Phân loại chủ đề văn bản")
            
            t_in = gr.Textbox(label="Văn bản")
            t_out = gr.Textbox(interactive=False, label="Chủ đề dự đoán")
            gr.Button("Phân loại").click(
                lambda x: run_async(topic_single, x),
                inputs=t_in, outputs=t_out
            )

            gr.Markdown("---")
            gr.Markdown("### 📂 Xử lý hàng loạt")

            # --- HƯỚNG DẪN TOPIC ---
            with gr.Accordion("📝 Hướng dẫn định dạng file CSV", open=False):
                gr.Markdown("""
                **Yêu cầu file CSV:**
                - Encoding: `UTF-8`
                - Cần có **1 trong các cột sau**: `text`, `comment`, hoặc `content`.
                
                **Ví dụ nội dung file:**
                ```csv
                content
                "Thủ tướng chính phủ vừa ban hành chỉ thị mới..."
                "Đội tuyển bóng đá nam giành huy chương vàng..."
                ```
                """)
            # -----------------------

            with gr.Row():
                t_file = gr.File(label="Tải lên CSV")
                t_download = gr.File(label="Tải xuống kết quả")
            
            t_df = gr.DataFrame()
            
            gr.Button("Chạy Batch & Export").click(
                lambda f: run_batch_and_export(topic_batch, f),
                inputs=t_file, outputs=[t_df, t_download]
            )

        # ==========================
        # TAB 4: RECSYS
        # ==========================
        with gr.Tab("4. Gợi ý (RecSys)"):
            gr.Markdown("### ⭐ Dự đoán đánh giá người dùng (Rating Prediction)")
            
            

            gr.Markdown("---")
            gr.Markdown("### 📂 Xử lý hàng loạt")

            # --- HƯỚNG DẪN RECSYS (QUAN TRỌNG) ---
            with gr.Accordion("📝 Hướng dẫn định dạng file CSV (Bắt buộc đọc)", open=True):
                gr.Markdown("""
                **Cấu trúc file CSV bắt buộc:**
                Hệ thống sẽ tự động map các tên cột phổ biến về chuẩn:
                
                | Dữ liệu cần | Tên cột chấp nhận trong CSV | Bắt buộc? |
                |-------------|-----------------------------|-----------|
                | **Người dùng** | `user_id`, `username`, `reviews.username` | ✅ **CÓ** |
                | **Sản phẩm** | `product_id`, `item_id`, `asins` | ✅ **CÓ** |
                | **Tiêu đề** | `title`, `reviews.title` | ❌ (Tùy chọn) |
                | **Nội dung** | `text`, `reviews.text` | ❌ (Tùy chọn) |
                
                **Ví dụ mẫu:**
                ```csv
                username,product_id,title
                "john_doe","IPHONE15","Điện thoại rất mượt"
                "jane_smith","SAMSUNG_S24","Pin hơi yếu"
                ```
                """)
            # -------------------------------------

            with gr.Row():
                r_file = gr.File(label="Tải lên CSV")
                r_download = gr.File(label="Tải xuống kết quả")
            
            r_df = gr.DataFrame()
            
            gr.Button("Chạy Batch & Export").click(
                lambda f: run_batch_and_export(recsys_batch, f),
                inputs=r_file, outputs=[r_df, r_download]
            )

        # ==========================
        # TAB 5: TREND
        # ==========================
        with gr.Tab("5. Xu hướng (Trend)"):
            gr.Markdown("### 📈 Phân tích xu hướng")
            
            # --- HƯỚNG DẪN TREND ---
            with gr.Accordion("📝 Hướng dẫn định dạng file CSV", open=False):
                gr.Markdown("""
                **Lưu ý:** Tính năng đang thử nghiệm.
                **Cấu trúc mong đợi:**
                ```csv
                date,reviews.text
                "2024-01-01","Bình luận về sản phẩm A"
                "2024-01-02","Bình luận về sản phẩm B"
                ```
                """)
            # -----------------------

            with gr.Row():
                tr_file = gr.File(label="Tải lên CSV")
                tr_download = gr.File(label="Tải xuống kết quả")
            
            tr_df = gr.DataFrame()
            
            gr.Button("Phân tích & Export").click(
                lambda f: run_batch_and_export(trend_analysis, f),
                inputs=tr_file, outputs=[tr_df, tr_download]
            )

        # ==========================
        # TAB 6: DASHBOARD BÁO CÁO 
        # ==========================

        with gr.Tab("6. 📊 Dashboard Báo Cáo"):
            gr.Markdown("## Tổng hợp dữ liệu từ 3 nguồn: RecSys, Trend & Email Classification")
            
            # 1. Upload
            with gr.Accordion("📝 Hướng dẫn Upload Dataset", open=True):
                gr.Markdown("""
                • **File Rating:** từ Tab 4 — cần: `product_id`, `predicted_rating`, `date` (optional). 
                • **File Trend:** từ Tab 5 — cần: `date`, `topic`, `negative`, `positive`, `total_volume`. 
                • **File Email:** từ Tab 2 — cần: `label`, `is_spam`, `date`. 
                
                """)

            with gr.Row():
                d_file_rate = gr.File(label="1. File Rating (Required)")
                d_file_trend = gr.File(label="2. File Trend (Required)")
                d_file_email = gr.File(label="3. File Email (Optional)")

            btn_analyze = gr.Button("🚀 Chạy Phân Tích & Tìm Insight", variant="primary")
            gr.Markdown("---")

            # ======================
            # INSIGHT ZONE
            # ======================
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 💡 Insight Tự Động Theo Rule Engine")
                    txt_insights_view = gr.Textbox(label="Insight Summary", lines=12, interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Tải Báo Cáo")
                    # CẬP NHẬT: Thêm dl_pdf và dl_html để nhận các file trả về mới
                    dl_txt = gr.File(label="Báo cáo chi tiết (.txt)")
                    dl_csv = gr.File(label="Dữ liệu tổng hợp (.csv)")
                    dl_pdf = gr.File(label="Báo cáo PDF (Mới)") # <-- MỚI
                    dl_html = gr.File(label="Báo cáo HTML (Mới)") # <-- MỚI
                    dl_xlsx = gr.File(label="Báo cáo Excel (.xlsx, nhiều sheet)") # Giữ nguyên nếu bạn có ý định dùng component này sau

            # ======================
            # VISUALIZATION ZONE
            # ======================
            with gr.Tabs():
                # -------- TREND TAB ----------
                with gr.TabItem("📈 Trend & Anomaly Detection"):
                    plot_trend_view = gr.Plot(label="Xu hướng thảo luận & điểm bất thường")
                    # CẬP NHẬT: Vị trí 3: plot_topic_view (Top Negative Topics) thay thế plot_heatmap_view 
                    plot_topic_view = gr.Plot(label="Top 10 Chủ đề Tiêu cực") # <-- Sửa tên biến để dễ quản lý outputs
                    
                    gr.Markdown("#### 🔥 Các ngày có khách nói nhiều bất thường (Spike Detection)")
                    df_spike_view = gr.DataFrame(label="Danh sách ngày bất thường")

                # -------- RATING TAB ----------
                with gr.TabItem("⭐ Rating Performance"):
                    with gr.Row():
                        # CẬP NHẬT: Vị trí 2: plot_rating_view
                        plot_rating_view = gr.Plot(label="Phân bố điểm Rating") 
                        # plot_rating_time không được trả về, giữ nguyên hoặc xóa
                        plot_rating_time = gr.Plot(label="Rating theo thời gian") 
                    gr.Markdown("#### 🚨 Sản phẩm rủi ro (Rating thấp / Rating giảm)")
                    # CẬP NHẬT: df_risky_view
                    df_risky_view = gr.DataFrame(label="Danh sách sản phẩm rủi ro")

                # -------- EMAIL TAB ----------
                with gr.TabItem("📬 Email Analysis"):
                    # CẬP NHẬT: Vị trí 4: Biểu đồ phân bố Label mới (fig_email)
                    plot_email_label_view = gr.Plot(label="Phân bố Chủ đề/Label Email (%)") 
                    
                    # Các components không được trả về, giữ nguyên hoặc xóa
                    plot_email_spam = gr.Plot(label="Spam theo thời gian")
                    df_email_spike = gr.DataFrame(label="Ngày có spam tăng đột biến")
                    df_email_topics = gr.DataFrame(label="Nhóm chủ đề email nổi bật")

            # ======================
            # ACTION: RUN ANALYSIS
            # ======================
            btn_analyze.click(
                fn=analyze_advanced_dashboard, 
                inputs=[d_file_rate, d_file_trend, d_file_email],
                outputs=[
                    # 1. fig_trend (Trend & Anomaly)
                    plot_trend_view, 
                    # 2. fig_rating (Rating Distribution)
                    plot_rating_view, 
                    # 3. fig_topic (Top Negative Topics)
                    plot_topic_view, 
                    # 4. fig_email (Email Label Distribution) 
                    plot_email_label_view, 
                    plot_email_spam,
                    # 5. df_spikes (Spike Detection)
                    df_spike_view,
                    # 6. risky_products (Risky Products)
                    df_risky_view,

                    df_email_spike,

                    # 7. insights_text (Insight Summary)
                    txt_insights_view,

                    # 8. txt_path (Insights TXT)
                    dl_txt, 
                    # 9. csv_path (Summary CSV)
                    dl_csv, 
                    # 10. pdf_path (Report PDF)
                    dl_pdf, # <--- MỚI, VỊ TRÍ 10
                    # 11. html_path (Report HTML)
                    dl_html # <--- MỚI, VỊ TRÍ 11
                ] # TỔNG CỘNG 11 OUTPUTS
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
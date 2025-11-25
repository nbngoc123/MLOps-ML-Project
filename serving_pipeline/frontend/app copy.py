import gradio as gr
import os
import pandas as pd
import numpy as np
import httpx
import asyncio
import time 
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

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

def analyze_advanced_dashboard(file_rating, file_trend, file_email):
    """
    Xử lý phân tích chuyên sâu: Anomaly Detection, Cross-Analysis & Insight Generation
    """
    if not file_rating or not file_trend:
        return [None] * 9 # Trả về rỗng nếu thiếu file quan trọng

    # 1. Load Data
    try:
        df_rating = pd.read_csv(file_rating.name)
        df_trend = pd.read_csv(file_trend.name)
        # Email là optional, nếu có thì đọc
        df_email = pd.read_csv(file_email.name) if file_email else pd.DataFrame()
        
        # Pre-processing
        if 'date' in df_trend.columns:
            df_trend['date'] = pd.to_datetime(df_trend['date'], errors='coerce')
        
        # Chuẩn hóa tên cột (tránh lỗi case sensitive)
        df_rating.columns = [c.lower() for c in df_rating.columns]
        df_trend.columns = [c.lower() for c in df_trend.columns]

    except Exception as e:
        return [f"Lỗi đọc file: {str(e)}"] + [None]*8

    # ==========================
    # A. TREND ANALYSIS (KPIs & Anomaly)
    # ==========================
    
    # 1. Volume theo ngày
    daily_vol = df_trend.groupby('date')['total_volume'].sum().reset_index()
    
    # 2. Anomaly Detection (Phát hiện Spike)
    # Logic: Ngày nào volume > Mean + 1.5 * StdDev là bất thường
    vol_mean = daily_vol['total_volume'].mean()
    vol_std = daily_vol['total_volume'].std()
    threshold = vol_mean + 1.5 * vol_std
    
    df_spikes = daily_vol[daily_vol['total_volume'] > threshold].copy()
    df_spikes['note'] = '🔥 High Volume Spike'
    df_spikes = df_spikes.sort_values('total_volume', ascending=False)

    # 3. Top Negative Topics
    # Tính negative rate: sum(Negative) / sum(Total)
    topic_stats = df_trend.groupby('topic')[['negative', 'total_volume']].sum().reset_index()
    topic_stats['neg_rate'] = (topic_stats['negative'] / topic_stats['total_volume']) * 100
    top_neg_topics = topic_stats.sort_values('neg_rate', ascending=False).head(5)

    # ==========================
    # B. RATING ANALYSIS
    # ==========================
    
    # 1. Avg Rating per Product
    prod_stats = df_rating.groupby('product_id')['predicted_rating'].agg(['mean', 'count']).reset_index()
    prod_stats.columns = ['product_id', 'avg_rating', 'review_count']
    
    # 2. Risky Products (Rating < 3.5)
    risky_products = prod_stats[prod_stats['avg_rating'] < 3.5].sort_values('avg_rating')
    
    # 3. User Variance (Optional): Độ lệch chuẩn rating của từng user
    # (Để xem user nào khó tính hay dễ tính)
    if 'user_id' in df_rating.columns:
        user_var = df_rating.groupby('user_id')['predicted_rating'].std().mean()
    else:
        user_var = 0

    # ==========================
    # C. CROSS-INSIGHTS (TỔNG HỢP)
    # ==========================
    
    insights = []
    insights.append("=== BÁO CÁO PHÂN TÍCH TỔNG HỢP ===")
    
    # Insight 1: Tình hình chung
    avg_r = df_rating['predicted_rating'].mean()
    avg_n_rate = (df_trend['negative'].sum() / df_trend['total_volume'].sum()) * 100
    insights.append(f"1. Tổng quan:\n   - Rating trung bình toàn sàn: {avg_r:.2f}/5.0\n   - Tỷ lệ tiêu cực trên mxh/trend: {avg_n_rate:.1f}%")
    
    # Insight 2: Mâu thuẫn (Polarization)
    if avg_r > 4.0 and avg_n_rate > 30:
        insights.append("⚠️ CẢNH BÁO: Sản phẩm có Rating cao nhưng Thảo luận tiêu cực nhiều -> Có thể là 'Seeding' ảo hoặc Sản phẩm gây tranh cãi.")
    elif avg_r < 3.0 and avg_n_rate > 50:
        insights.append("🚨 KHỦNG HOẢNG: Rating thấp và Dư luận rất tiêu cực -> Cần dừng bán hoặc cải tổ sản phẩm ngay.")
    else:
        insights.append("✅ Ổn định: Chỉ số Rating và Sentiment tương đồng.")

    # Insight 3: Spike Analysis
    if not df_spikes.empty:
        spike_dates = df_spikes['date'].dt.strftime('%Y-%m-%d').tolist()
        insights.append(f"🔥 Phát hiện bất thường: Có {len(df_spikes)} ngày lượng thảo luận tăng vọt: {', '.join(spike_dates)}. Cần kiểm tra xem là khủng hoảng hay viral tốt.")

    # Insight 4: Vấn đề cụ thể
    bad_topic = top_neg_topics.iloc[0]
    insights.append(f"❌ Chủ đề bị phàn nàn nhiều nhất: '{bad_topic['topic']}' (Tỷ lệ tiêu cực: {bad_topic['neg_rate']:.1f}%).")

    # ==========================
    # D. VISUALIZATION
    # ==========================
    
    # Plot 1: Volume & Spikes (Line Chart)
    fig_trend = plt.figure(figsize=(10, 5))
    sns.lineplot(data=daily_vol, x='date', y='total_volume', marker='o', label='Daily Volume')
    # Vẽ điểm spike
    if not df_spikes.empty:
        plt.scatter(df_spikes['date'], df_spikes['total_volume'], color='red', s=100, zorder=5, label='Anomaly (Spike)')
    plt.title('Xu hướng thảo luận & Điểm bất thường')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Plot 2: Rating Distribution
    fig_rating = plt.figure(figsize=(6, 5))
    sns.histplot(df_rating['predicted_rating'], bins=20, kde=True, color='green')
    plt.axvline(3.5, color='red', linestyle='--', label='Ngưỡng rủi ro (3.5)')
    plt.title('Phân bố điểm đánh giá')
    plt.legend()
    plt.tight_layout()

    # Plot 3: Negative Rate by Topic (Bar Chart)
    fig_topic = plt.figure(figsize=(8, 5))
    sns.barplot(data=top_neg_topics, x='neg_rate', y='topic', palette='Reds_r')
    plt.title('Top Chủ đề có tỷ lệ tiêu cực cao nhất')
    plt.xlabel('Tỷ lệ tiêu cực (%)')
    plt.tight_layout()

    # ==========================
    # E. EXPORT FILES
    # ==========================
    
    # 1. File Insights Text
    txt_path = os.path.join(GRADIO_TEMP_DIR, f"insights_{int(time.time())}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(insights))
        f.write("\n\n--- CHI TIẾT SẢN PHẨM RỦI RO ---\n")
        f.write(risky_products.head(20).to_string())

    # 2. File Summary CSV (Gộp Risky Product + Spikes)
    csv_path = os.path.join(GRADIO_TEMP_DIR, f"summary_analysis_{int(time.time())}.csv")
    # Chúng ta lưu danh sách sản phẩm rủi ro làm chính
    risky_products.to_csv(csv_path, index=False, encoding='utf-8-sig')

    return (
        fig_trend, fig_rating, fig_topic,   # 3 Biểu đồ
        df_spikes[['date', 'total_volume', 'note']], # Dataframe Spike
        risky_products.head(10),            # Dataframe Risky Products
        "\n".join(insights),                # Text Insights hiển thị
        txt_path,                           # File txt download
        csv_path                            # File csv download
    )
    
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
            gr.Markdown("### Tổng hợp dữ liệu từ 3 nguồn: RecSys, Trend & Email Classification")
            
            # 1. Khu vực Upload
            with gr.Accordion("📝 Hướng dẫn Upload 3 file dữ liệu nguồn", open=True):
                gr.Markdown("""
                Để tạo Dashboard, vui lòng upload các file kết quả (`export_...csv`) từ các Tab trước:
                1. **File Rating:** Kết quả từ Tab 4 (RecSys). Cần cột: `product_id`, `predicted_rating`.
                2. **File Trend:** Kết quả từ Tab 5 (Trend). Cần cột: `date`, `topic`, `Negative`, `Positive`.
                3. **File Email:** Kết quả từ Tab 2 (Email). Cần cột: `label`, `is_spam`.
                """)
            
            with gr.Row():
                d_file_rate = gr.File(label="1. File Rating (Required)")
                d_file_trend = gr.File(label="2. File Trend (Required)")
                d_file_email = gr.File(label="3. File Email (Optional)")
            
            btn_analyze = gr.Button("🚀 Chạy Phân Tích & Tìm Insight", variant="primary")
        
            gr.Markdown("---")
            
            # Khu vực Insights Text
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 💡 Tự động phát hiện Insights")
                    txt_insights_view = gr.Textbox(label="AI Summary", lines=10, interactive=False)
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Tải báo cáo")
                    dl_txt = gr.File(label="Báo cáo chi tiết (.txt)")
                    dl_csv = gr.File(label="Dữ liệu tổng hợp (.csv)")

            # Khu vực Biểu đồ
            with gr.Tabs():
                with gr.TabItem("📈 Trend & Anomalies"):
                    plot_trend_view = gr.Plot(label="Biểu đồ xu hướng")
                    gr.Markdown("#### ⚠️ Các ngày có lượng thảo luận tăng đột biến (Spikes)")
                    df_spike_view = gr.DataFrame(label="Danh sách ngày bất thường")
                
                with gr.TabItem("⭐ Rating Performance"):
                    with gr.Row():
                        plot_rating_view = gr.Plot(label="Phân bố điểm")
                        plot_topic_view = gr.Plot(label="Vấn đề theo Topic")
                    
                    gr.Markdown("#### 🚨 Danh sách sản phẩm rủi ro (Rating < 3.5)")
                    df_risky_view = gr.DataFrame(label="Cần kiểm tra gấp")

            # Wiring
            btn_analyze.click(
                fn=analyze_advanced_dashboard,
                inputs=[d_file_rate, d_file_trend, d_file_email],
                outputs=[
                    plot_trend_view, plot_rating_view, plot_topic_view, # Plots
                    df_spike_view, df_risky_view,                       # Dataframes
                    txt_insights_view,                                  # Text Insight
                    dl_txt, dl_csv                                      # Download files
                ]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
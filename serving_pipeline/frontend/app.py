import gradio as gr
import os
import pandas as pd
import httpx
import asyncio
import os
os.environ["GRADIO_TEMP_DIR"] = "/tmp/gradio"
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


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
# Wrappers để chạy async
# ========================

def run_async(func, *args):
    return asyncio.run(func(*args))


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

    return f"Chủ đề: {result['topic']} (Conf: {result['confidence']:.2%})"


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
# GIAO DIỆN
# ========================

with gr.Blocks(title="NexusML AI Platform", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🚀 NexusML AI Platform")

    with gr.Tabs():

        # Sentiment
        with gr.Tab("1. Cảm xúc"):
            s_in = gr.Textbox(label="Nhập bình luận")
            s_out = gr.Textbox(label="Kết quả", interactive=False)
            gr.Button("Phân tích").click(
                lambda x: run_async(sentiment_single, x),
                inputs=s_in, outputs=s_out
            )

            s_file = gr.File()
            s_df = gr.DataFrame()
            gr.Button("Batch CSV").click(
                # lambda f: run_async(sentiment_batch, f),
                sentiment_batch,
                inputs=s_file, outputs=s_df
            )

        # Email
        with gr.Tab("2. Email"):
            e_in = gr.Textbox(label="Email Content")
            e_out = gr.Textbox(interactive=False)
            gr.Button("Check Spam").click(
                lambda x: run_async(email_single, x),
                inputs=e_in, outputs=e_out
            )

            e_file = gr.File()
            e_df = gr.DataFrame()
            gr.Button("Batch CSV").click(
                # lambda f: run_async(email_batch, f),
                email_batch,
                inputs=e_file, outputs=e_df
            )

        # Topic
        with gr.Tab("3. Chủ đề"):
            t_in = gr.Textbox(label="Văn bản")
            t_out = gr.Textbox(interactive=False)
            gr.Button("Phân loại").click(
                lambda x: run_async(topic_single, x),
                inputs=t_in, outputs=t_out
            )

            t_file = gr.File()
            t_df = gr.DataFrame()
            gr.Button("Batch CSV").click(
                lambda f: run_async(topic_batch, f),
                inputs=t_file, outputs=t_df
            )

        # RecSys
        with gr.Tab("4. RecSys"):
            uid = gr.Textbox(label="User ID")
            pid = gr.Textbox(label="Product ID")
            desc = gr.Textbox(label="Mô tả SP")
            r_out = gr.Textbox(interactive=False)

            gr.Button("Dự đoán").click(
                lambda u, p, d: run_async(recsys_single, u, p, d),
                inputs=[uid, pid, desc], outputs=r_out
            )

            r_file = gr.File()
            r_df = gr.DataFrame()
            gr.Button("Batch CSV").click(
                lambda f: run_async(recsys_batch, f),
                inputs=r_file, outputs=r_df
            )

        # Trend
        with gr.Tab("5. Xu hướng"):
            tr_file = gr.File()
            tr_df = gr.DataFrame()
            gr.Button("Phân tích").click(
                lambda f: run_async(trend_analysis, f),
                inputs=tr_file, outputs=tr_df
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

import gradio as gr
import requests
import time
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")
API_KEY = os.getenv("API_KEY", "AIO2024")

HEADERS = {
    "X-API-Key": API_KEY
}

# Sentiment classification function
def classify_sentiment(text):
    """Classify sentiment of product review"""
    if not text or text.strip() == "":
        return "No text provided", 0.0, ""
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/sentiment/predict",
            headers=HEADERS,
            json={"text": text}
        )
        response.raise_for_status()
        result = response.json()
        
        label = result.get("label", "Unknown")
        confidence = result.get("confidence", 0.0)
        emoji = "😊" if label == "positive" else "😢" if label == "negative" else "😐"
        
        return f"{emoji} {label.upper()}", confidence, f"Prediction successful"
    except Exception as e:
        return "Error", 0.0, f"Error: {str(e)}"

# Email classification function
def classify_email(subject, body):
    """Classify email as spam/support/order"""
    if not subject and not body:
        return "No content provided", 0.0, ""
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/email/classify",
            headers=HEADERS,
            json={"subject": subject, "body": body}
        )
        response.raise_for_status()
        result = response.json()
        
        category = result.get("category", "Unknown")
        confidence = result.get("confidence", 0.0)
        
        emoji_map = {"spam": "🚫", "support": "🎧", "order": "📦"}
        emoji = emoji_map.get(category.lower(), "📧")
        
        return f"{emoji} {category.upper()}", confidence, "Classification successful"
    except Exception as e:
        return "Error", 0.0, f"Error: {str(e)}"

# Batch CSV processing
def process_csv(file):
    """Process CSV file with sentiment classification"""
    if file is None:
        return None, "No file uploaded"
    
    try:
        # Read CSV
        df = pd.read_csv(file.name)
        
        if 'text' not in df.columns and 'comment' not in df.columns:
            return None, "CSV must contain 'text' or 'comment' column"
        
        text_col = 'text' if 'text' in df.columns else 'comment'
        
        # Send to batch API
        texts = df[text_col].tolist()
        response = requests.post(
            f"{API_BASE_URL}/sentiment/batch",
            headers=HEADERS,
            json={"texts": texts}
        )
        response.raise_for_status()
        results = response.json()
        
        # Add results to dataframe
        df['sentiment'] = [r.get('label', 'unknown') for r in results]
        df['confidence'] = [r.get('confidence', 0.0) for r in results]
        
        # Create visualization
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
        )
        
        # Save results
        output_path = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_path, index=False)
        
        return fig, f"Processed {len(df)} rows. Results saved to {output_path}"
    except Exception as e:
        return None, f"Error: {str(e)}"

# Dashboard stats
def get_dashboard_stats(date_from=None, date_to=None, product=None):
    """Get sentiment statistics for dashboard"""
    try:
        params = {}
        if date_from:
            params['date_from'] = date_from
        if date_to:
            params['date_to'] = date_to
        if product:
            params['product'] = product
            
        response = requests.get(
            f"{API_BASE_URL}/sentiment/stats",
            headers=HEADERS,
            params=params
        )
        response.raise_for_status()
        data = response.json()
        
        # Create visualizations
        # Sentiment distribution
        fig1 = go.Figure(data=[
            go.Bar(
                x=list(data.get('sentiment_counts', {}).keys()),
                y=list(data.get('sentiment_counts', {}).values()),
                marker_color=['green', 'red', 'gray']
            )
        ])
        fig1.update_layout(title="Sentiment Distribution", xaxis_title="Sentiment", yaxis_title="Count")
        
        # Top negative reviews
        negative_reviews = data.get('top_negative', [])
        negative_df = pd.DataFrame(negative_reviews)
        
        stats_text = f"""
        📊 **Statistics Summary**
        - Total Reviews: {data.get('total', 0)}
        - Positive: {data.get('sentiment_counts', {}).get('positive', 0)}
        - Negative: {data.get('sentiment_counts', {}).get('negative', 0)}
        - Neutral: {data.get('sentiment_counts', {}).get('neutral', 0)}
        - Average Confidence: {data.get('avg_confidence', 0):.2%}
        """
        
        return fig1, negative_df, stats_text
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# Build Gradio Interface
with gr.Blocks(title="MLOps Sentiment Analysis Platform", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧠 MLOps Sentiment Analysis Platform
        
        **Features:**
        - 💬 Real-time sentiment classification for product reviews
        - 📧 Email classification (Spam/Support/Order)
        - 📊 Batch processing with CSV upload
        - 📈 Analytics dashboard with visualizations
        """
    )
    
    with gr.Tabs():
        # Tab 1: Sentiment Classification
        with gr.Tab("💬 Sentiment Analysis"):
            gr.Markdown("### Classify product review sentiment")
            with gr.Row():
                with gr.Column():
                    sentiment_input = gr.Textbox(
                        label="Product Review",
                        placeholder="Enter customer review here...",
                        lines=5
                    )
                    sentiment_button = gr.Button("Classify Sentiment", variant="primary")
                
                with gr.Column():
                    sentiment_label = gr.Textbox(label="Sentiment", interactive=False)
                    sentiment_confidence = gr.Number(label="Confidence", interactive=False)
                    sentiment_status = gr.Textbox(label="Status", interactive=False)
        
        # Tab 2: Email Classification
        with gr.Tab("📧 Email Classification"):
            gr.Markdown("### Classify customer emails")
            with gr.Row():
                with gr.Column():
                    email_subject = gr.Textbox(
                        label="Email Subject",
                        placeholder="Enter email subject...",
                        lines=2
                    )
                    email_body = gr.Textbox(
                        label="Email Body",
                        placeholder="Enter email content...",
                        lines=8
                    )
                    email_button = gr.Button("Classify Email", variant="primary")
                
                with gr.Column():
                    email_category = gr.Textbox(label="Category", interactive=False)
                    email_confidence = gr.Number(label="Confidence", interactive=False)
                    email_status = gr.Textbox(label="Status", interactive=False)
        
        # Tab 3: Batch Processing
        with gr.Tab("📊 Batch Processing"):
            gr.Markdown("""
            ### Upload CSV for batch sentiment analysis
            
            **CSV Format Requirements:**
            - Must contain a column named `text` or `comment`
            - Each row will be classified separately
            - Results will be saved with sentiment labels and confidence scores
            """)
            with gr.Row():
                with gr.Column():
                    csv_file = gr.File(
                        label="Upload CSV File",
                        file_types=[".csv"]
                    )
                    csv_button = gr.Button("Process CSV", variant="primary")
                
                with gr.Column():
                    csv_chart = gr.Plot(label="Sentiment Distribution")
                    csv_status = gr.Textbox(label="Processing Status", interactive=False)
        
        # Tab 4: Dashboard
        with gr.Tab("📈 Analytics Dashboard"):
            gr.Markdown("### View sentiment analysis statistics")
            with gr.Row():
                date_from = gr.Textbox(label="From Date (YYYY-MM-DD)", placeholder="2024-01-01")
                date_to = gr.Textbox(label="To Date (YYYY-MM-DD)", placeholder="2024-12-31")
                product_filter = gr.Textbox(label="Product Name (optional)", placeholder="Leave empty for all")
            
            dashboard_button = gr.Button("Load Dashboard", variant="primary")
            
            with gr.Row():
                dashboard_chart = gr.Plot(label="Sentiment Distribution")
            
            with gr.Row():
                with gr.Column():
                    dashboard_stats = gr.Markdown(label="Statistics")
                with gr.Column():
                    dashboard_negatives = gr.Dataframe(
                        label="Top Negative Reviews",
                        headers=["text", "sentiment", "confidence"]
                    )
    
    # Connect buttons to functions
    sentiment_button.click(
        classify_sentiment,
        inputs=[sentiment_input],
        outputs=[sentiment_label, sentiment_confidence, sentiment_status]
    )
    
    email_button.click(
        classify_email,
        inputs=[email_subject, email_body],
        outputs=[email_category, email_confidence, email_status]
    )
    
    csv_button.click(
        process_csv,
        inputs=[csv_file],
        outputs=[csv_chart, csv_status]
    )
    
    dashboard_button.click(
        get_dashboard_stats,
        inputs=[date_from, date_to, product_filter],
        outputs=[dashboard_chart, dashboard_negatives, dashboard_stats]
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)


import logging
import pandas as pd
import numpy as np
from io import BytesIO
from fastapi import HTTPException, UploadFile, File
from app.services.model_manager import model_manager

logger = logging.getLogger("trend-controller")
logging.basicConfig(level=logging.INFO)

# ========================
# API Endpoints
# ========================

async def startup_event():
    # ModelManager sẽ tự động đăng ký model classes khi ứng dụng khởi động
    pass

async def analyze_trend_csv(file: UploadFile = File(...)):
    """
    Input: File CSV chứa cột 'date' và 'comment' (hoặc 'text').
    Output: JSON (dạng bảng CSV) chứa thống kê xu hướng theo ngày.
    """
    try:
        # 1. Lấy models thông qua ModelManager
        sentiment_model = await model_manager.get_model(
            model_type="sentiment",
            model_name="sentiment_classifier_Model",
            model_version="production"
        )
        
        topic_model = await model_manager.get_model(
            model_type="topic",
            model_name="Nexus_Topic_Classifier",
            model_version="production"
        )

        # 2. Đọc File
        content = await file.read()
        try:
            df = pd.read_csv(BytesIO(content), encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(BytesIO(content), encoding='latin1')
        
        # 3. Validate Columns
        text_col = next((col for col in ['comment', 'reviews.text', 'content', 'feedback'] if col in df.columns), None)
        date_col = next((col for col in ['date', 'time', 'created_at'] if col in df.columns), None)
        
        if not text_col or not date_col:
            raise HTTPException(400, "CSV phải chứa cột thời gian ('date', 'time') và nội dung ('comment', 'reviews.text', 'content')")

        logger.info(f"Analyzing trends for {len(df)} records...")

        # 4. Preprocessing
        df['date'] = pd.to_datetime(df[date_col], utc=True, errors='coerce')
        df = df.dropna(subset=['date'])
        
        clean_text_series = df[text_col].astype(str).str.lower().fillna("")
        clean_text_series = clean_text_series.str.replace(r'[^\w\s]', '', regex=True) 
        clean_text_series = clean_text_series.str.replace(r'\s+', ' ', regex=True).str.strip()
        
        df_inference = pd.DataFrame({'clean_text': clean_text_series})

        # 5. Inference (Chạy 2 model)
        # Dự đoán Topic
        df['topic'] = topic_model.model.predict(df_inference['clean_text'])
        
        # Dự đoán Sentiment
        sent_preds = sentiment_model.model.predict(df_inference['clean_text'])
        
        def map_sentiment(val):
            s_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
            if isinstance(val, (int, np.integer)):
                return s_map.get(val, 'Neutral')
            return str(val).capitalize() 

        df['sentiment'] = [map_sentiment(x) for x in sent_preds]

        # 6. Aggregation
        trend_df = df.groupby([
            pd.Grouper(key='date', freq='D'), 
            'topic', 
            'sentiment'
        ]).size().reset_index(name='volume')

        pivot_trend = trend_df.pivot_table(
            index=['date', 'topic'], 
            columns='sentiment', 
            values='volume', 
            fill_value=0
        ).reset_index()

        for col in ['Negative', 'Positive', 'Neutral']:
            if col not in pivot_trend.columns:
                pivot_trend[col] = 0

        pivot_trend['total_volume'] = pivot_trend['Negative'] + pivot_trend['Positive'] + pivot_trend['Neutral']
        pivot_trend['negative_rate'] = pivot_trend.apply(
            lambda x: round(x['Negative'] / x['total_volume'], 2) if x['total_volume'] > 0 else 0.0, 
            axis=1
        )

        pivot_trend['date'] = pivot_trend['date'].dt.strftime('%Y-%m-%d')

        # 7. Return Result
        result_data = pivot_trend.to_dict(orient="records")

        return {
            "filename": file.filename,
            "total_records": len(df),
            "total_days": len(pivot_trend['date'].unique()),
            "data": result_data,
            "columns": list(pivot_trend.columns) 
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trend Analysis Error: {e}")
        raise HTTPException(500, f"Lỗi phân tích xu hướng: Không thể xử lý dữ liệu. Chi tiết: {str(e)}")

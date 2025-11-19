# MLOps Sentiment Analysis Frontend

This is a Gradio-based web interface for the MLOps Sentiment Analysis Platform. It provides user-friendly interfaces for sentiment classification of product reviews and email classification.

## Features

- 💬 **Real-time Sentiment Analysis**: Classify product reviews as positive, negative, or neutral
- 📧 **Email Classification**: Categorize customer emails as spam, support, or order-related
- 📊 **Batch Processing**: Upload CSV files for bulk sentiment analysis
- 📈 **Analytics Dashboard**: View sentiment statistics with interactive visualizations
- 🔍 **Top Negative Reviews**: Identify and address critical customer feedback

## Setup

### 1. Create Virtual Environment (WSL)

```bash
cd /mnt/c/Users/tuan2/coding/MLOps-ML-Project/serving_pipeline/frontend

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in this directory:

```bash
cp .env.example .env
```

Edit the `.env` file:

```env
API_BASE_URL=http://localhost:8000/api
API_KEY=AIO2024
PORT=7860
```

### 4. Run the Application

```bash
python app.py
```

The interface will be available at `http://localhost:7860`

## Usage

### Sentiment Analysis

1. Navigate to the **💬 Sentiment Analysis** tab
2. Enter a product review in the text box
3. Click **Classify Sentiment**
4. View the sentiment label, confidence score, and emoji indicator

### Email Classification

1. Go to the **📧 Email Classification** tab
2. Enter the email subject and body
3. Click **Classify Email**
4. See the category (Spam/Support/Order) with confidence score

### Batch Processing

1. Navigate to the **📊 Batch Processing** tab
2. Upload a CSV file with a `text` or `comment` column
3. Click **Process CSV**
4. View the sentiment distribution chart
5. Results are saved to a timestamped CSV file

**CSV Format Example:**
```csv
text,product_id,date
"Great product! Highly recommend",101,2024-01-15
"Terrible quality, very disappointed",102,2024-01-16
```

### Analytics Dashboard

1. Go to the **📈 Analytics Dashboard** tab
2. Optional: Set date range and product filters
3. Click **Load Dashboard**
4. View:
   - Sentiment distribution bar chart
   - Statistics summary (total, positive, negative, neutral counts)
   - Table of top negative reviews for action

## API Endpoints Expected

The frontend expects the following backend API endpoints:

### Sentiment Analysis
- `POST /api/sentiment/predict` - Single text classification
  ```json
  {
    "text": "Your review text here"
  }
  ```

- `POST /api/sentiment/batch` - Batch classification
  ```json
  {
    "texts": ["review1", "review2", "..."]
  }
  ```

- `GET /api/sentiment/stats` - Get statistics
  Query params: `date_from`, `date_to`, `product`

### Email Classification
- `POST /api/email/classify` - Classify email
  ```json
  {
    "subject": "Email subject",
    "body": "Email content"
  }
  ```

## Architecture

This frontend is part of the larger MLOps pipeline:

```
Frontend (Gradio) → FastAPI Backend → MLflow Model Registry → PostgreSQL
                         ↓
                   Prometheus/Grafana
                   (Monitoring)
```

## Development

To modify the interface:

1. Edit `app.py` to add new features or tabs
2. Update `requirements.txt` if adding new dependencies
3. Test locally before deploying
4. Ensure backend API endpoints match the expected contract

## Deployment

For production deployment, consider:

- Running behind a reverse proxy (Nginx)
- Using Docker for containerization
- Setting up proper authentication
- Configuring CORS for API calls
- Enabling SSL/TLS

## Troubleshooting

**Issue**: Cannot connect to backend API
- Check that `API_BASE_URL` in `.env` is correct
- Ensure backend service is running
- Verify network connectivity

**Issue**: Import errors
- Activate virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

**Issue**: CSV upload fails
- Ensure CSV has `text` or `comment` column
- Check file encoding (should be UTF-8)
- Verify CSV format is valid

## License

Part of the MLOps-ML-Project. See main repository LICENSE.

5. Wait for the generation process to complete

## Connection to Backend

This frontend connects to the NexusML backend API. Make sure the backend is running and properly configured in the `.env` file.

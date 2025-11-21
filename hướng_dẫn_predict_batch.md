# Hướng Dẫn Sử Dụng Predict Batch cho Các Model

Tài liệu này mô tả chi tiết các trường dữ liệu cần thiết khi sử dụng tính năng `predict_batch` cho từng model trong hệ thống.

## 1. TopicModel - Phân Loại Chủ Đề

### Mục đích
Model phân loại văn bản thành các chủ đề khác nhau.

### Trường dữ liệu đầu vào (CSV)
```csv
text,comment,content
"Văn bản cần phân loại chủ đề", "",""
```

**Cột bắt buộc:**
- `text` - Nội dung văn bản cần phân loại
- `comment` - Thay thế cho `text`
- `content` - Thay thế cho `text`

**Lưu ý:** Chỉ cần 1 trong 3 cột trên.

### Kết quả trả về
```json
{
  "filename": "ten_file.csv",
  "total_rows": 100,
  "data": [
    {
      "text": "Nội dung gốc",
      "topic": "chủ đề_dự_đoán",
      "run_source": "mlflow_production_batch"
    }
  ],
  "run_source": "mlflow_production_batch"
}
```

---

## 2. SentimentModel - Phân Tích Cảm Xúc

### Mục đích
Model phân tích cảm xúc (tích cực/tiêu cực/trung tính) của văn bản.

### Trường dữ liệu đầu vào (CSV)
```csv
text,comment,content
"Tôi rất hài lòng với sản phẩm", "",""
```

**Cột bắt buộc:**
- `text` - Nội dung văn bản cần phân tích
- `comment` - Thay thế cho `text`
- `content` - Thay thế cho `text`

**Lưu ý:** Chỉ cần 1 trong 3 cột trên.

### Kết quả trả về
```json
{
  "filename": "ten_file.csv",
  "total_rows": 100,
  "data": [
    {
      "text": "Nội dung gốc",
      "sentiment": "tích cực", // tiêu cực, tích cực, trung tính
      "confidence": 0.95,
      "run_source": "mlflow_production_batch"
    }
  ],
  "run_source": "mlflow_production_batch"
}
```

---

## 3. EmailModel - Phân Loại Email Spam

### Mục đích
Model phát hiện email spam/ham (không phải spam).

### Trường dữ liệu đầu vào (CSV)
```csv
text,content,body,email
"Nội dung email cần kiểm tra", "","",""
```

**Cột bắt buộc:**
- `text` - Nội dung email
- `content` - Thay thế cho `text`
- `body` - Thay thế cho `text`
- `email` - Thay thế cho `text`

**Lưu ý:** Chỉ cần 1 trong 4 cột trên.

### Kết quả trả về
```json
{
  "filename": "ten_file.csv",
  "total_rows": 100,
  "data": [
    {
      "text": "Nội dung gốc",
      "is_spam": true,
      "label": "spam", // spam, ham
      "confidence": 0.88,
      "run_source": "mlflow_production_batch"
    }
  ],
  "run_source": "mlflow_production_batch"
}
```

---

## 4. RecSysModel - Hệ Thống Gợi Ý Sản Phẩm

### Mục đích
Model dự đoán rating mà người dùng có thể cho sản phẩm.

### Trường dữ liệu đầu vào (CSV)
```csv
reviews.username,asins,reviews.title,reviews.text
"user123","B001ABC123","Tên sản phẩm","Nội dung đánh giá"
```

**Cột bắt buộc:**
- `reviews.username` hoặc `username` → Map thành `user_id`
- `asins` hoặc `product_id` hoặc `item_id` → Map thành `product_id`

**Cột tùy chọn (để cải thiện độ chính xác):**
- `reviews.title` hoặc `title` → Map thành `title`
- `reviews.text` hoặc `text` → Map thành `text`

### Ánh xạ cột (Column Mapping)
| Cột đầu vào | Cột sau khi map | Bắt buộc |
|-------------|----------------|----------|
| `reviews.username` | `user_id` | ✓ |
| `username` | `user_id` | ✓ |
| `asins` | `product_id` | ✓ |
| `product_id` | `product_id` | ✓ |
| `item_id` | `product_id` | ✓ |
| `reviews.title` | `title` | ✗ |
| `title` | `title` | ✗ |
| `reviews.text` | `text` | ✗ |
| `text` | `text` | ✗ |

### Kết quả trả về
```json
{
  "filename": "ten_file.csv",
  "total_rows": 100,
  "data": [
    {
      "user_id": "user123",
      "product_id": "B001ABC123",
      "predicted_rating": 4.2
    }
  ],
  "run_source": "mlflow_production_batch"
}
```

### Ví dụ CSV hoàn chỉnh cho RecSysModel
```csv
reviews.username,asins,reviews.title,reviews.text
"john_doe","B001ABC123","iPhone 15 Pro","Excellent phone with great camera"
"jane_smith","B001DEF456","Samsung Galaxy S24","Good but battery could be better"
"bob_wilson","B001GHI789","MacBook Air M3","Amazing performance for work"
```

---

## 5. TrendModel - Phân Tích Xu Hướng

### Mục đích
Model phân tích xu hướng thời gian (đang trong giai đoạn phát triển).

### Trường dữ liệu đầu vào (CSV)
```csv
date,reviews.text (comment về sp)
"2024-01-01",100,"sales"
"2024-01-02",120,"sales"
```

**Lưu ý:** Hiện tại model này chỉ là placeholder và chưa có logic phân tích time series thực tế.

### Kết quả trả về
```json
{
  "filename": "ten_file.csv",
  "total_rows": 100,
  "data": [
    {
      "date": "2024-01-01",
      "value": 100,
      "category": "sales",
      "trend_analysis": "Time series analysis required"
    }
  ],
  "run_source": "mlflow_production_batch"
}
```

---

## Tổng Kết Các Yêu Cầu Chung

### Định dạng file
- **Định dạng:** CSV (Comma Separated Values)
- **Encoding:** UTF-8
- **Phân tách:** Dấu phẩy (,)

### Xử lý lỗi thường gặp

1. **Thiếu cột bắt buộc:**
   ```
   HTTPException 400: CSV must contain 'text', 'comment' or 'content' column
   ```

2. **Tên cột sai (RecSysModel):**
   ```
   HTTPException 400: CSV thiếu cột (hoặc sai tên): ['user_id', 'product_id']
   ```

3. **Dữ liệu null/empty:**
   - Các model tự động xử lý giá trị null
   - Text rỗng sẽ được thay thế bằng chuỗi trống

### Gợi ý sử dụng

1. **Topic/Sentiment/Email Models:** Chỉ cần cột text đơn giản
2. **RecSysModel:** Cần user_id và product_id chính xác, có thể bổ sung title/text để tăng độ chính xác
3. **TrendModel:** Chưa sẵn sàng sử dụng trong production

### Lưu ý về Performance
- **Kích thước file:** Không quá 50MB mỗi lần upload
- **Số dòng:** Tối đa 10,000 dòng mỗi batch
- **Xử lý bất đồng bộ:** Các request được xử lý song song

---


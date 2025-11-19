# Chọn image Python nhẹ
FROM python:3.11-slim-buster
# Thư mục làm việc
WORKDIR /app

# Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code
COPY . .

# Biến môi trường cho Gradio và Backend
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
ENV BACKEND_URL="http://backend:8000/api"

# Mở cổng Gradio
EXPOSE 7860

# Chạy ứng dụng Gradio
CMD ["python", "app.py"]

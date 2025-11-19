from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from mlflow.tracking import MlflowClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import s3fs
import boto3
import os
import mlflow
import logging
import shutil

log = logging.getLogger(__name__)

# --- CONFIG ---
DAG_ID = "product_recsys_dag"
MINIO_BUCKET = "nexusml"
USE_CASE_PREFIX = "product_recsys"
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ['MLFLOW_S3_ENDPOINT_URL'] = MINIO_ENDPOINT
os.environ['AWS_ACCESS_KEY_ID'] = MINIO_ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = MINIO_SECRET_KEY

default_args = {'owner': 'ml-team', 'start_date': datetime(2025, 11, 16), 'retries': 1}
dag = DAG(DAG_ID, default_args=default_args, schedule='0 11 * * *', catchup=False, tags=['recsys', 'nexusml'])

def get_storage_options():
    return {"key": MINIO_ACCESS_KEY, "secret": MINIO_SECRET_KEY, "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}}

@task(dag=dag)
def ensure_bucket_exists():
    s3 = boto3.client('s3', endpoint_url=MINIO_ENDPOINT, aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY)
    try: s3.create_bucket(Bucket=MINIO_BUCKET)
    except: pass
    for folder in ["raw", "processed", "models"]:
        try: s3.put_object(Bucket=MINIO_BUCKET, Key=f"{USE_CASE_PREFIX}/{folder}/.keep", Body=b'')
        except: pass

@task(dag=dag)
def extract_data():
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': MINIO_ENDPOINT}, key=MINIO_ACCESS_KEY, secret=MINIO_SECRET_KEY)
    files = fs.glob(f"{MINIO_BUCKET}/{USE_CASE_PREFIX}/raw/*.csv")
    
    if not files:
        # Mock Data: User - Product Interaction
        df = pd.DataFrame({
            "user_id": [1, 1, 2, 2, 3, 3, 4],
            "product_id": [101, 102, 101, 103, 102, 104, 101],
            "rating": [5, 3, 4, 5, 2, 5, 4] # Hoặc implicit feedback = 1
        })
        path = "/tmp/demo_recsys.csv"
        df.to_csv(path, index=False)
        boto3.client('s3', endpoint_url=MINIO_ENDPOINT, aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY)\
             .upload_file(path, MINIO_BUCKET, f"{USE_CASE_PREFIX}/raw/demo_recsys.csv")
        files = [f"{MINIO_BUCKET}/{USE_CASE_PREFIX}/raw/demo_recsys.csv"]

    dfs = [pd.read_csv(f"s3://{f}", storage_options=get_storage_options()) for f in files]
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv(f"s3://{MINIO_BUCKET}/{USE_CASE_PREFIX}/processed/combined.csv", index=False, storage_options=get_storage_options())
    return f"{USE_CASE_PREFIX}/processed/combined.csv"

@task(dag=dag)
def preprocess_recsys(file_key: str):
    """
    Xử lý dữ liệu: Chọn cột, điền khuyết thiếu, Label Encode ID, gộp Text.
    """
    # 1. Đọc các cột quan trọng
    usecols = [
        'reviews.username', 'asins', 'reviews.rating', 
        'reviews.text', 'reviews.title', 'name', 'manufacturer', 'categories'
    ]
    
    # Xử lý trường hợp file thực tế có thể thiếu cột
    df_raw = pd.read_csv(f"s3://{MINIO_BUCKET}/{file_key}", storage_options=get_storage_options())
    existing_cols = [c for c in usecols if c in df_raw.columns]
    df = df_raw[existing_cols].copy()
    
    # 2. Đổi tên cột cho dễ làm việc
    rename_map = {
        'reviews.username': 'user_id',
        'asins': 'item_id',
        'reviews.rating': 'rating',
        'reviews.text': 'text',
        'reviews.title': 'title'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # 3. Xử lý Missing Data
    df.dropna(subset=['user_id', 'item_id', 'rating'], inplace=True)
    df.fillna('', inplace=True) # Điền chuỗi rỗng cho các cột text thiếu
    
    # 4. Feature Engineering: Gộp thông tin Text & Metadata
    # Ý tưởng: Gom tất cả đặc điểm mô tả sản phẩm và review vào 1 câu dài để model hiểu ngữ cảnh
    df['full_description'] = (
        df['title'] + " " + 
        df['text'] + " " + 
        df['name'] + " " + 
        df['manufacturer'] + " " + 
        df['categories']
    ).str.lower()
    
    # 5. Label Encoding cho User và Item
    # Cần lưu lại Encoder để sau này khi dự đoán thực tế (Inference) còn map lại được
    le_user = LabelEncoder()
    le_item = LabelEncoder()
    
    df['user_idx'] = le_user.fit_transform(df['user_id'].astype(str))
    df['item_idx'] = le_item.fit_transform(df['item_id'].astype(str))
    
    # Lưu Encoders
    local_enc_dir = "/tmp/encoders"
    os.makedirs(local_enc_dir, exist_ok=True)
    joblib.dump(le_user, f"{local_enc_dir}/user_le.pkl")
    joblib.dump(le_item, f"{local_enc_dir}/item_le.pkl")
    
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': MINIO_ENDPOINT}, key=MINIO_ACCESS_KEY, secret=MINIO_SECRET_KEY)
    fs.put(f"{local_enc_dir}/user_le.pkl", f"{MINIO_BUCKET}/{USE_CASE_PREFIX}/encoders/user_le.pkl")
    fs.put(f"{local_enc_dir}/item_le.pkl", f"{MINIO_BUCKET}/{USE_CASE_PREFIX}/encoders/item_le.pkl")
    
    # 6. Lưu dữ liệu sạch
    # Giữ lại: user_idx, item_idx, full_description (Text Feature), rating (Target)
    clean_key = f"{USE_CASE_PREFIX}/processed/clean_hybrid.csv"
    df[['user_idx', 'item_idx', 'full_description', 'rating']].to_csv(
        f"s3://{MINIO_BUCKET}/{clean_key}", 
        index=False, 
        storage_options=get_storage_options()
    )
    
    return clean_key

@task(dag=dag)
def train_recsys_model(clean_key: str):
    """
    Huấn luyện Hybrid Model:
    - Input: UserID (Int), ItemID (Int), Full Description (Text)
    - Model: Gradient Boosting Regressor
    """
    df = pd.read_csv(f"s3://{MINIO_BUCKET}/{clean_key}", storage_options=get_storage_options())
    
    # Tách Feature và Target
    X = df[['user_idx', 'item_idx', 'full_description']]
    y = df['rating']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- XÂY DỰNG PIPELINE PHỨC HỢP ---
    
    # 1. Xử lý Text: Biến đổi text thành vector số (TF-IDF)
    # Chỉ lấy max_features=500 để demo nhẹ nhàng, thực tế có thể tăng lên
    text_transformer = TfidfVectorizer(stop_words='english', max_features=500)
    
    # 2. ColumnTransformer: Áp dụng xử lý riêng cho từng cột
    # - Cột 'full_description': đi qua TfidfVectorizer
    # - Cột 'user_idx', 'item_idx': giữ nguyên (passthrough) vì Tree-based model xử lý tốt số nguyên
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, 'full_description'),
            ('ids', 'passthrough', ['user_idx', 'item_idx'])
        ]
    )
    
    # 3. Model: Gradient Boosting Regressor
    # Đây là model mạnh mẽ, có thể học phi tuyến tính từ cả ID và Text Vector
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42
        ))
    ])
    
    # --- TRAINING & LOGGING ---
    
    mlflow.set_experiment(f"nexus_{USE_CASE_PREFIX}")
    with mlflow.start_run() as run:
        log.info("Training Gradient Boosting Hybrid Model...")
        model.fit(X_train, y_train)
        
        # Evaluation
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        
        log.info(f"RMSE: {rmse}")
        
        # Log Metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        # Log Params
        mlflow.log_param("model_type", "GradientBoostingRegressor")
        mlflow.log_param("features", "User_Item_ID_plus_TFIDF_Metadata")
        
        # Log Model
        mlflow.sklearn.log_model(model, "model")
        
    return {"rmse": rmse, "run_id": run.info.run_id}

@task.branch(dag=dag)
def evaluate(metrics: dict):
    # Với RMSE, càng thấp càng tốt. Ví dụ: sai số dưới 2 điểm rating là chấp nhận
    return "register_model" if metrics['rmse'] < 2.0 else "notify_fail"

@task(dag=dag)
def register_model(metrics: dict):
    """
    FIX lỗi backup: Tải Artifact từ MLflow về trước, sau đó backup thủ công lên MinIO.
    """
    run_id = metrics.get('run_id')
    if not run_id:
        log.warning("No run_id provided; skipping register_model")
        return

    # 1. Đăng ký và gán alias (Giữ nguyên)
    client = MlflowClient(MLFLOW_TRACKING_URI)
    model_name = "Nexus_RecSys_SVD"
    reg = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    client.set_registered_model_alias(model_name, "production", reg.version)
    log.info("Registered and aliased version %s", reg.version)

    # 2. Thiết lập kết nối S3/MinIO và thư mục tạm
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': MINIO_ENDPOINT}, key=MINIO_ACCESS_KEY, secret=MINIO_SECRET_KEY)
    version_path = f"{MINIO_BUCKET}/{USE_CASE_PREFIX}/models/v{reg.version}"
    temp_download_dir = f"/tmp/download_artifacts_v{reg.version}"
    os.makedirs(temp_download_dir, exist_ok=True)
    
    # 3. TẢI ARTIFACT TỪ MLFLOW VỀ LOCAL (Sửa lỗi phụ thuộc worker)
    try:
        log.info(f"Downloading artifacts from run {run_id}...")
        # Tải folder 'model' (chứa model.pkl)
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model", dst_path=temp_download_dir)
        # Tải folder 'artifacts' (chứa vectorizer.pkl)
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="artifacts", dst_path=temp_download_dir)
    except Exception as e:
        log.warning(f"Could not download artifacts: {e}")
        shutil.rmtree(temp_download_dir)
        return

    # 4. BACKUP thủ công (Dùng file vừa tải về)
    local_model_path = f"{temp_download_dir}/model/model.pkl"
    local_vectorizer_path = f"{temp_download_dir}/artifacts/vectorizer.pkl"
    
    try: fs.touch(f"{version_path}/.keep") 
    except: pass
        
    if os.path.exists(local_model_path):
        fs.put(local_model_path, f"{version_path}/model.pkl")
        fs.put(local_vectorizer_path, f"{version_path}/vectorizer.pkl")
        log.info(f"Đã backup model thủ công vào: {version_path}")
        
    # 5. Dọn dẹp
    if os.path.exists(temp_download_dir):
        shutil.rmtree(temp_download_dir)



# def register_model_simplified(metrics: dict):
#     """
#     Chỉ đăng ký và gán alias trong MLflow. Loại bỏ logic backup thủ công.
#     """
#     run_id = metrics.get('run_id')
#     if not run_id:
#         log.warning("No run_id provided; skipping register_model")
#         return

#     client = MlflowClient(MLFLOW_TRACKING_URI)
#     model_name = "Nexus_RecSys_SVD"
    
#     # 1. Đăng ký mô hình
#     reg = mlflow.register_model(f"runs:/{run_id}/model", model_name)
#     log.info("Registered model name=%s version=%s", model_name, reg.version)

#     # 2. Gán alias 'production'
#     client.set_registered_model_alias(
#         name=model_name, 
#         alias="production", 
#         version=reg.version
#     )
#     log.info("Set alias 'production' for version %s", reg.version)

# Flow
init = ensure_bucket_exists()
raw = extract_data()
clean = preprocess_recsys(raw)
trained = train_recsys_model(clean)
check = evaluate(trained)
reg = register_model(trained)
fail = BashOperator(task_id='notify_fail', bash_command='echo "RecSys RMSE too high"', dag=dag)
success = BashOperator(task_id='notify_success', bash_command='echo "RecSys Deployed"', trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS, dag=dag)

init >> raw >> clean >> trained >> check
check >> reg >> success
check >> fail
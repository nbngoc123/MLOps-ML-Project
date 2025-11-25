import os
import time
import base64
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# ---------------------
# Ensure temporary directory
# ---------------------
GRADIO_TEMP_DIR = os.environ.get("GRADIO_TEMP_DIR", "/tmp/gradio")
os.makedirs(GRADIO_TEMP_DIR, exist_ok=True)

# ---------------------
# Helper: Validate inputs
# ---------------------
def _validate_inputs(df_rating, df_trend, df_email):
    """Return list of error messages (empty if ok)."""
    errs = []

    # Rating required columns
    if df_rating is None or df_rating.empty:
        errs.append("File Rating missing.")
    else:
        rcols = set(df_rating.columns.str.lower())
        need_r = {"product_id", "predicted_rating"}
        if not need_r.issubset(rcols):
            errs.append(f"Rating file thiếu cột: {sorted(list(need_r - rcols))}")

    # Trend required columns
    if df_trend is None or df_trend.empty:
        errs.append("File Trend missing.")
    else:
        tcols = set(df_trend.columns.str.lower())
        need_t = {"date", "topic", "total_volume", "negative", "positive"}
        if not need_t.issubset(tcols):
            errs.append(f"Trend file thiếu cột: {sorted(list(need_t - tcols))}")

    # Email optional but validate if present
    if df_email is not None and not df_email.empty:
        ecols = set(df_email.columns.str.lower())
        need_e = {"label", "is_spam"}  # adjust depending on schema
        if not need_e.issubset(ecols):
            errs.append(f"Email file thiếu cột: {sorted(list(need_e - ecols))}")

    return errs

# ---------------------
# Utility: Save figure as PNG
# ---------------------
def _save_fig(fig, name_prefix="chart"):
    fname = os.path.join(GRADIO_TEMP_DIR, f"{name_prefix}_{int(time.time()*1000)}.png")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname

# ---------------------
# Utility: Create simple HTML report with embedded images
# ---------------------
def _create_html_report(image_paths, insights_text, csv_summary_path):
    html_path = os.path.join(GRADIO_TEMP_DIR, f"report_{int(time.time())}.html")
    parts = [
        "<html><head><meta charset='utf-8'><title>Dashboard Report</title></head><body>",
        f"<h1>Dashboard Report</h1><pre>{insights_text}</pre><hr/>"
    ]
    for p in image_paths:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            parts.append(f"<img src='data:image/png;base64,{b64}' style='max-width:900px;display:block;margin-bottom:20px;'/>")
    parts.append(f"<hr/><p>CSV summary: {os.path.basename(csv_summary_path)}</p>")
    parts.append("</body></html>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return html_path

# ---------------------
# Main Analysis Function
# ---------------------
def analyze_advanced_dashboard(file_rating, file_trend, file_email=None):
    """
    Inputs: gr.File objects (or objects with .name path)
    Returns: figures, DataFrames, insights text, txt/csv/pdf/html paths
    """
    # Helper to extract path from file object
    def _get_path(f):
        if f is None: return None
        return getattr(f, "name", str(f))

    p_rating = _get_path(file_rating)
    p_trend = _get_path(file_trend)
    p_email = _get_path(file_email)

    # Load CSVs safely
    try:
        df_rating = pd.read_csv(p_rating) if p_rating else pd.DataFrame()
        df_trend = pd.read_csv(p_trend) if p_trend else pd.DataFrame()
        df_email = pd.read_csv(p_email) if p_email else pd.DataFrame()
    except Exception as e:
        err_txt = f"Lỗi đọc file: {str(e)}"
        return (None, None, None, pd.DataFrame(), pd.DataFrame(), err_txt, None, None, None, None)

    # Normalize column names
    df_rating.columns = [c.lower() for c in df_rating.columns]
    df_trend.columns = [c.lower() for c in df_trend.columns]
    if not df_email.empty:
        df_email.columns = [c.lower() for c in df_email.columns]

    # Validate inputs
    errs = _validate_inputs(df_rating, df_trend, df_email)
    if errs:
        err_txt = " | ".join(errs)
        return (None, None, None, pd.DataFrame(), pd.DataFrame(), err_txt, None, None, None, None)

    # ---------------------
    # Preprocessing
    # ---------------------
    df_trend['date'] = pd.to_datetime(df_trend['date'], errors='coerce')
    for col in ['total_volume', 'negative', 'positive', 'neutral']:
        if col in df_trend.columns:
            df_trend[col] = pd.to_numeric(df_trend[col], errors='coerce').fillna(0)
    df_trend['topic'] = df_trend['topic'].astype(str)
    df_rating['predicted_rating'] = pd.to_numeric(df_rating['predicted_rating'], errors='coerce')
    if 'user_id' in df_rating.columns:
        df_rating['user_id'] = df_rating['user_id'].astype(str)

    # ---------------------
    # A. Trend KPIs
    # ---------------------
    daily_vol = df_trend.groupby('date', as_index=False)['total_volume'].sum().sort_values('date')
    vol_mean = daily_vol['total_volume'].mean()
    vol_std = daily_vol['total_volume'].std()
    threshold = vol_mean + 1.5 * (vol_std if not np.isnan(vol_std) else 0)
    df_spikes = daily_vol[daily_vol['total_volume'] > threshold].copy()
    df_spikes['note'] = '🔥 High Volume Spike'

    topic_stats = df_trend.groupby('topic', as_index=False).agg({
        'negative': 'sum',
        'total_volume': 'sum',
        'positive': 'sum'
    })
    topic_stats['neg_rate'] = np.where(topic_stats['total_volume'] > 0,
                                       topic_stats['negative'] / topic_stats['total_volume'] * 100, 0)
    topic_stats['severity'] = topic_stats['neg_rate'] * topic_stats['total_volume']
    top_neg_topics = topic_stats.sort_values('neg_rate', ascending=False).head(10)

    # Topic growth
    topic_growth = []
    try:
        recent_window_end = df_trend['date'].max()
        recent_start = recent_window_end - pd.Timedelta(days=6)
        prev_start = recent_start - pd.Timedelta(days=7)
        recent = df_trend[df_trend['date'] >= recent_start].groupby('topic')['total_volume'].sum()
        prev = df_trend[(df_trend['date'] >= prev_start) & (df_trend['date'] < recent_start)].groupby('topic')['total_volume'].sum()
        growth_df = (recent - prev).fillna(0).rename('growth').reset_index()
        growth_df['growth_pct'] = growth_df['growth'] / (prev.replace(0, np.nan).abs()) * 100
        growth_df['growth_pct'] = growth_df['growth_pct'].replace([np.inf, -np.inf], np.nan).fillna(0)
        topic_growth = growth_df.sort_values('growth_pct', ascending=False).head(5)
    except Exception:
        topic_growth = pd.DataFrame(columns=['topic','growth','growth_pct'])

    # ---------------------
    # B. Rating KPIs
    # ---------------------
    prod_stats = df_rating.groupby('product_id', as_index=False)['predicted_rating'].agg(
        avg_rating=('mean'),
        review_count=('count')
    )

    prod_stats['rating_std'] = df_rating.groupby('product_id')['predicted_rating'].std().values
    prod_stats['pct_ge_4'] = df_rating.groupby('product_id')['predicted_rating'].apply(lambda x: (x >= 4.0).mean()).values * 100

    risky_products = prod_stats[prod_stats['avg_rating'] < 3.5].sort_values('avg_rating')

    # ---------------------
    # C. Email KPIs
    # ---------------------
    email_summary = pd.DataFrame()
    email_spam_summary = pd.DataFrame()
    df_email_topics = pd.DataFrame(columns=['Topic', 'Count'])
    email_insights = []

    if not df_email.empty:

        # 1. Chuẩn hóa cột date/timestamp và is_spam
        date_col_name = next((c for c in df_email.columns if 'date' in c or 'timestamp' in c), None)
        if date_col_name:
            df_email['date'] = pd.to_datetime(df_email[date_col_name], errors='coerce')
            df_email = df_email.dropna(subset=['date'])

        # 1. Tóm tắt theo Label
        if 'label' in df_email.columns:
            email_summary = df_email['label'].value_counts(normalize=True).mul(100).round(1).reset_index()
            email_summary.columns = ['Label', 'Percentage (%)']
            
            # Thêm insight
            top_label = email_summary.iloc[0]['Label']
            top_pct = email_summary.iloc[0]['Percentage (%)']
            email_insights.append(f"📦 Nhãn phổ biến nhất: {top_label} ({top_pct:.1f}%)")

        # 2. Tóm tắt theo Spam
        if 'is_spam' in df_email.columns:
            df_email['is_spam'] = df_email['is_spam'].astype(str).str.lower()
            spam_counts = df_email['is_spam'].value_counts()
            total_emails = spam_counts.sum()
            spam_rate = spam_counts.get('true', 0) / total_emails * 100
            email_insights.append(f"📧 Tỷ lệ Spam: {spam_rate:.1f}%")
            
            email_spam_summary = df_email['is_spam'].value_counts().reset_index()
            email_spam_summary.columns = ['Is_Spam', 'Count']
            
        # 3. Phân tích Trend và Spike Spam
        daily_spam_vol = pd.DataFrame()
        if 'date' in df_email.columns and 'is_spam' in df_email.columns and not df_email['date'].isnull().all():
            daily_spam_vol = df_email[df_email['is_spam'] == True].groupby('date', as_index=False).size().rename(columns={'size':'spam_volume'})
            
            if not daily_spam_vol.empty:
                vol_mean = daily_spam_vol['spam_volume'].mean()
                vol_std = daily_spam_vol['spam_volume'].std()
                
                # Tính ngưỡng (ví dụ: trung bình + 1.5 độ lệch chuẩn)
                threshold = vol_mean + 1.5 * (vol_std if not np.isnan(vol_std) else 0)
                
                df_spam_spikes = daily_spam_vol[daily_spam_vol['spam_volume'] > threshold].copy()
                df_spam_spikes['note'] = '🔥 High Spam Volume Spike'
                
                if not df_spam_spikes.empty:
                    spike_dates = df_spam_spikes['date'].dt.strftime('%Y-%m-%d').tolist()
                    email_insights.append(f"🚨 Phát hiện spike Spam tại các ngày: {', '.join(spike_dates)}")



    # ---------------------
    # D. Generate insights text
    # ---------------------
    insights = []
    avg_r = df_rating['predicted_rating'].mean()
    total_neg = df_trend['negative'].sum()
    total_vol = max(df_trend['total_volume'].sum(), 1)
    avg_n_rate = (total_neg / total_vol) * 100

    insights.append(f"- Rating trung bình toàn hệ thống: {avg_r:.2f}/5")
    insights.append(f"- Tỷ lệ tiêu cực trung bình (trend): {avg_n_rate:.1f}%")
    insights.append(f"- Tổng volume (trend): {int(total_vol)}")
    if avg_r > 4.0 and avg_n_rate > 30:
        insights.append("⚠️ Cảnh báo POLARIZATION")
    if avg_r < 3.0 and avg_n_rate > 50:
        insights.append("🚨 KHỦNG HOẢNG")
    if not df_spikes.empty:
        spike_dates = df_spikes['date'].dt.strftime('%Y-%m-%d').tolist()
        insights.append(f"🔥 Phát hiện spike volume tại các ngày: {', '.join(spike_dates)}")
    if not top_neg_topics.empty:
        bad_topic_row = top_neg_topics.iloc[0]
        insights.append(f"❌ Chủ đề nhiều than phiền nhất: {bad_topic_row['topic']} ({bad_topic_row['neg_rate']:.1f}%)")
    if not risky_products.empty:
        sample = risky_products.head(5)
        insights.append(f"🚨 Sản phẩm cần kiểm tra: {', '.join(sample['product_id'].astype(str).tolist())}")
    if isinstance(topic_growth, pd.DataFrame) and not topic_growth.empty:
        top_growth = topic_growth.iloc[0]
        insights.append(f"📈 Topic tăng mạnh: {top_growth['topic']} (growth% ≈ {top_growth['growth_pct']:.1f}%)")
    if email_insights:
        insights.extend(["\n--- Email/Review Analysis ---"] + email_insights)

    insights_text = "\n".join(insights)

    # ---------------------
    # E. Visualizations
    # ---------------------
    fig_trend, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(data=daily_vol, x='date', y='total_volume', marker='o', ax=ax)
    if not df_spikes.empty:
        ax.scatter(df_spikes['date'], df_spikes['total_volume'], color='red', s=80)
    ax.set_title('Daily Discussion Volume')
    ax.set_ylabel('Total Volume')
    ax.grid(alpha=0.3)
    fig_trend.tight_layout()
    
    fig_rating, ax2 = plt.subplots(figsize=(6,4))
    sns.histplot(df_rating['predicted_rating'].dropna(), bins=20, kde=True, ax=ax2)
    ax2.axvline(3.5, color='red', linestyle='--', label='Risk threshold 3.5')
    ax2.set_title('Rating Distribution')
    ax2.set_xlabel('Predicted Rating')
    ax2.legend()
    fig_rating.tight_layout()

    fig_topic, ax3 = plt.subplots(figsize=(8,4))
    plot_df = top_neg_topics.head(10).sort_values('neg_rate')
    sns.barplot(data=plot_df, x='neg_rate', y='topic', ax=ax3)
    ax3.set_xlabel('Negative Rate (%)')
    ax3.set_title('Top Negative Topics')
    fig_topic.tight_layout()

    fig_email = None
    # email_summary phải được tính toán ở phần C. Email KPIs
    if not email_summary.empty and 'Label' in email_summary.columns and 'Percentage (%)' in email_summary.columns:
        try:
            fig_email, ax4 = plt.subplots(figsize=(6,4))
            plot_df = email_summary.set_index('Label')['Percentage (%)']
            
            # Gom nhóm các nhãn nhỏ (nếu có) để biểu đồ trực quan hơn
            top_n_labels = plot_df.nlargest(5)
            others_pct = plot_df[~plot_df.index.isin(top_n_labels.index)].sum()
            
            if others_pct > 0:
                top_n_labels['Others'] = others_pct
            
            ax4.pie(top_n_labels, labels=top_n_labels.index, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
            ax4.set_title('Email/Review Label Distribution (%)')
            ax4.axis('equal')
            fig_email.tight_layout()
        except Exception as e:
            print(f"Lỗi khi tạo biểu đồ Email: {e}")
            fig_email = None # Gán lại None nếu có lỗi

    fig_spam_trend = None
    if 'daily_spam_vol' in locals() and not daily_spam_vol.empty:
        fig_spam_trend, ax5 = plt.subplots(figsize=(10,4))
        sns.lineplot(data=daily_spam_vol, x='date', y='spam_volume', marker='o', ax=ax5)
        if not df_spam_spikes.empty:
            ax5.scatter(df_spam_spikes['date'], df_spam_spikes['spam_volume'], color='red', s=80)
        ax5.set_title('Daily Spam Volume Trend')
        ax5.set_ylabel('Spam Volume (Count)')
        ax5.grid(alpha=0.3)
        fig_spam_trend.tight_layout()

    # ---------------------
    # F. Export files
    # ---------------------
    txt_path = os.path.join(GRADIO_TEMP_DIR, f"insights_{int(time.time())}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(insights_text + "\n\n")
        f.write("--- Top risk products ---\n")
        f.write(risky_products.to_string(index=False) if not risky_products.empty else "None\n")
        f.write("\n\n--- Top negative topics ---\n")
        f.write(top_neg_topics.to_string(index=False))

    csv_path = os.path.join(GRADIO_TEMP_DIR, f"summary_{int(time.time())}.csv")
    risky_products.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # Chuẩn bị danh sách Figures và Image Paths
    figures_to_save = [fig_trend, fig_rating, fig_topic]
    image_paths = [
        _save_fig(fig_trend, "trend"),
        _save_fig(fig_rating, "rating"),
        _save_fig(fig_topic, "topic")
    ]

    # Thêm Figure và Path của Email nếu có
    if fig_email is not None:
        figures_to_save.append(fig_email)
        image_paths.append(_save_fig(fig_email, "email_label"))
    else:
        # Nếu fig_email không được tạo, tạo một figure trống để giữ vị trí trong tuple trả về
        fig_email = plt.figure() 
        plt.close(fig_email)

    # Thêm fig_spam_trend 
    if fig_spam_trend is not None:
        figures_to_save.append(fig_spam_trend)
        image_paths.append(_save_fig(fig_spam_trend, "spam_trend"))
    else:
        fig_spam_trend = plt.figure()
        plt.close(fig_spam_trend)

    pdf_path = os.path.join(GRADIO_TEMP_DIR, f"report_{int(time.time())}.pdf")
    with PdfPages(pdf_path) as pdf:
        fig_txt = plt.figure(figsize=(8.27, 11.69))
        fig_txt.text(0.01, 0.99, "Dashboard Report (Rule-based)\n\n", fontsize=14, weight='bold', va='top')
        fig_txt.text(0.01, 0.95, insights_text, fontsize=10, va='top')
        pdf.savefig(fig_txt)
        plt.close(fig_txt)
        pdf.savefig(fig_trend)
        pdf.savefig(fig_rating)
        pdf.savefig(fig_topic)

    img1 = _save_fig(fig_trend, "trend")
    img2 = _save_fig(fig_rating, "rating")
    img3 = _save_fig(fig_topic, "topic")
    html_path = _create_html_report([img1, img2, img3], insights_text, csv_path)

    return (fig_trend, fig_rating, fig_topic, fig_email, fig_spam_trend,
            df_spikes[['date','total_volume','note']].reset_index(drop=True),
            risky_products.reset_index(drop=True),
            df_spam_spikes[['date','spam_volume','note']].reset_index(drop=True),
            insights_text,
            txt_path,
            csv_path,
            pdf_path,
            html_path)

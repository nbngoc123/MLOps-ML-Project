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
    df_spam_spikes = pd.DataFrame(columns=['date', 'spam_volume', 'note']) # Default Empty
    daily_spam_vol = pd.DataFrame() # Default Empty
    email_insights = []

    if not df_email.empty:
        # 1. Chuẩn hóa Date
        date_col_name = next((c for c in df_email.columns if 'date' in c or 'timestamp' in c), None)
        if date_col_name:
            df_email['date'] = pd.to_datetime(df_email[date_col_name], errors='coerce')
            df_email = df_email.dropna(subset=['date'])

        # 2. Chuẩn hóa Spam Label
        if 'is_spam' in df_email.columns:
            df_email['is_spam_bool'] = df_email['is_spam'].astype(str).str.lower().isin(['true', '1'])
            
            # Spam Stats
            total_emails = len(df_email)
            spam_count = df_email['is_spam_bool'].sum()
            spam_rate = (spam_count / total_emails * 100) if total_emails > 0 else 0
            email_insights.append(f"📧 Tỷ lệ Spam: {spam_rate:.1f}%")

            # Spam Trend & Spikes
            if not df_email['date'].isnull().all():
                daily_spam_vol = df_email[df_email['is_spam_bool']].groupby('date', as_index=False).size().rename(columns={'size':'spam_volume'})
                
                if not daily_spam_vol.empty and daily_spam_vol['spam_volume'].sum() > 0:
                    s_mean = daily_spam_vol['spam_volume'].mean()
                    s_std = daily_spam_vol['spam_volume'].std()
                    s_thresh = s_mean + 1.5 * (s_std if not np.isnan(s_std) else 0)
                    
                    found_spikes = daily_spam_vol[daily_spam_vol['spam_volume'] > s_thresh].copy()
                    if not found_spikes.empty:
                        found_spikes['note'] = '🔥 High Spam Volume Spike'
                        df_spam_spikes = found_spikes # Update dataframe

        # 3. Email Label Summary
        if 'label' in df_email.columns:
            email_summary = df_email['label'].value_counts(normalize=True).mul(100).round(1).reset_index()
            email_summary.columns = ['Label', 'Percentage (%)']
            if not email_summary.empty:
                top_label = email_summary.iloc[0]['Label']
                top_pct = email_summary.iloc[0]['Percentage (%)']
                email_insights.append(f"📦 Nhãn phổ biến nhất: {top_label} ({top_pct:.1f}%)")



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
    # 1. Trend Line
    fig_trend, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(data=daily_vol, x='date', y='total_volume', marker='o', ax=ax)
    if not df_spikes.empty:
        ax.scatter(df_spikes['date'], df_spikes['total_volume'], color='red', s=80, label='Spike')
    ax.set_title('Daily Discussion Volume')
    ax.grid(alpha=0.3)
    fig_trend.tight_layout()
    
    # 2. Rating Dist
    fig_rating, ax2 = plt.subplots(figsize=(6,4))
    sns.histplot(df_rating['predicted_rating'].dropna(), bins=20, kde=True, ax=ax2)
    ax2.axvline(3.5, color='red', linestyle='--', label='Risk threshold')
    ax2.set_title('Rating Distribution')
    fig_rating.tight_layout()

    # 3. Topic Bar
    fig_topic, ax3 = plt.subplots(figsize=(8,4))
    plot_df = top_neg_topics.head(10).sort_values('neg_rate')
    sns.barplot(data=plot_df, x='neg_rate', y='topic', ax=ax3)
    ax3.set_title('Top Negative Topics')
    fig_topic.tight_layout()

    # 4. Email Pie (Optional)
    fig_email = None
    if not email_summary.empty:
        try:
            fig_email, ax4 = plt.subplots(figsize=(6,4))
            plot_df = email_summary.set_index('Label')['Percentage (%)']
            top_n = plot_df.nlargest(5)
            others = plot_df[~plot_df.index.isin(top_n.index)].sum()
            if others > 0: top_n['Others'] = others
            ax4.pie(top_n, labels=top_n.index, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Email Label Distribution')
            fig_email.tight_layout()
        except: fig_email = None

    # 5. Spam Trend (Optional)
    fig_spam_trend = None
    if not daily_spam_vol.empty:
        try:
            fig_spam_trend, ax5 = plt.subplots(figsize=(10,4))
            sns.lineplot(data=daily_spam_vol, x='date', y='spam_volume', marker='o', color='orange', ax=ax5)
            if not df_spam_spikes.empty:
                ax5.scatter(df_spam_spikes['date'], df_spam_spikes['spam_volume'], color='red', s=80, label='Spike')
            ax5.set_title('Daily SPAM Volume Trend')
            ax5.grid(alpha=0.3)
            fig_spam_trend.tight_layout()
        except: fig_spam_trend = None

    # ---------------------
    # F. Export files
    # ---------------------
    txt_path = os.path.join(GRADIO_TEMP_DIR, f"insights_{int(time.time())}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(insights_text)

    csv_path = os.path.join(GRADIO_TEMP_DIR, f"summary_{int(time.time())}.csv")
    risky_products.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # --- STEP 1: SAVE PDF (While figures are still OPEN) ---
    pdf_path = os.path.join(GRADIO_TEMP_DIR, f"report_{int(time.time())}.pdf")
    with PdfPages(pdf_path) as pdf:
        # Title Page
        fig_txt = plt.figure(figsize=(8.27, 11.69))
        fig_txt.text(0.1, 0.9, "Dashboard Report", fontsize=16, weight='bold')
        fig_txt.text(0.1, 0.8, insights_text, fontsize=10, va='top')
        pdf.savefig(fig_txt)
        plt.close(fig_txt)
        
        # Save Charts
        pdf.savefig(fig_trend)
        pdf.savefig(fig_rating)
        pdf.savefig(fig_topic)
        if fig_email: pdf.savefig(fig_email)
        if fig_spam_trend: pdf.savefig(fig_spam_trend)

    # --- STEP 2: SAVE PNGs (This CLOSES the figures) ---
    img_paths = []
    img_paths.append(_save_fig(fig_trend, "trend"))
    img_paths.append(_save_fig(fig_rating, "rating"))
    img_paths.append(_save_fig(fig_topic, "topic"))
    if fig_email:
        img_paths.append(_save_fig(fig_email, "email_dist"))
    if fig_spam_trend:
        img_paths.append(_save_fig(fig_spam_trend, "spam_trend"))

    # --- STEP 3: HTML REPORT (With ALL images) ---
    html_path = _create_html_report(img_paths, insights_text, csv_path)

    # Return empty placeholders if figures are None for Gradio compatibility
    if fig_email is None: 
        fig_email = plt.figure()
        plt.close(fig_email)
    if fig_spam_trend is None:
        fig_spam_trend = plt.figure()
        plt.close(fig_spam_trend)

    return (fig_trend, fig_rating, fig_topic, fig_email, fig_spam_trend,
            df_spikes[['date','total_volume','note']].reset_index(drop=True),
            risky_products.reset_index(drop=True),
            df_spam_spikes[['date','spam_volume','note']].reset_index(drop=True),
            insights_text,
            txt_path,
            csv_path,
            pdf_path,
            html_path)
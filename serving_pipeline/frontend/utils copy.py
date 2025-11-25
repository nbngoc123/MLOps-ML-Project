import os
import time
import base64
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# ensure temp dir (use the same GRADIO_TEMP_DIR used earlier)
GRADIO_TEMP_DIR = os.environ.get("GRADIO_TEMP_DIR", "/tmp/gradio")
os.makedirs(GRADIO_TEMP_DIR, exist_ok=True)
# ---------------------
# Helper: validation
# ---------------------
def _validate_inputs(df_rating, df_trend, df_email):
    """Return list of error messages (empty if ok)."""
    errs = []
    # Rating required cols
    if df_rating is None:
        errs.append("File Rating missing.")
    else:
        rcols = set(df_rating.columns.str.lower())
        need_r = {"product_id", "predicted_rating"}
        if not need_r.issubset(rcols):
            errs.append(f"Rating file thiếu cột: {sorted(list(need_r - rcols))}")

    # Trend required cols
    if df_trend is None:
        errs.append("File Trend missing.")
    else:
        tcols = set(df_trend.columns.str.lower())
        need_t = {"date", "topic", "total_volume", "negative", "positive"}
        if not need_t.issubset(tcols):
            errs.append(f"Trend file thiếu cột: {sorted(list(need_t - tcols))}")

    # Email optional but if present validate
    if df_email is not None and not df_email.empty:
        ecols = set(df_email.columns.str.lower())
        need_e = {"label", "is_spam"}  # adjust depending on your schema
        if not need_e.issubset(ecols):
            errs.append(f"Email file thiếu cột: {sorted(list(need_e - ecols))}")

    return errs

# ---------------------
# Utility: Save figure to PNG bytes & optionally return file path
# ---------------------
def _save_fig(fig, name_prefix="chart"):
    fname = os.path.join(GRADIO_TEMP_DIR, f"{name_prefix}_{int(time.time()*1000)}.png")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname

# ---------------------
# Utility: create simple HTML report with embedded images
# ---------------------
def _create_html_report(image_paths, insights_text, csv_summary_path):
    html_path = os.path.join(GRADIO_TEMP_DIR, f"report_{int(time.time())}.html")
    parts = ["<html><head><meta charset='utf-8'><title>Dashboard Report</title></head><body>"]
    parts.append(f"<h1>Dashboard Report</h1><pre>{insights_text}</pre><hr/>")
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
# Main analysis function (Gradio callable)
# ---------------------
def analyze_advanced_dashboard(file_rating, file_trend, file_email=None):
    """
    Inputs: gr.File objects (or objects with .name path)
    Returns:
      fig_trend (mpl fig), fig_rating (mpl fig), fig_topic (mpl fig),
      df_spikes (pandas df), df_risky (pandas df), insights_text (str),
      txt_path (str), csv_path (str), pdf_path (str), html_path (str)
    Note: Gradio outputs must match what the UI expects. Adjust accordingly.
    """
    # Validate existence
    # Accept either gr.File-like object (has .name) or raw path string
    def _get_path(f):
        if f is None: return None
        return getattr(f, "name", str(f))

    p_rating = _get_path(file_rating)
    p_trend = _get_path(file_trend)
    p_email = _get_path(file_email)

    # Load CSVs robustly
    try:
        df_rating = pd.read_csv(p_rating) if p_rating else pd.DataFrame()
        df_trend = pd.read_csv(p_trend) if p_trend else pd.DataFrame()
        df_email = pd.read_csv(p_email) if p_email else pd.DataFrame()
    except Exception as e:
        err_txt = f"Lỗi đọc file: {str(e)}"
        # return shapes consistent with former outputs: 3 figs, 2 dfs, text, txt, csv
        return (None, None, None, pd.DataFrame(), pd.DataFrame(), err_txt, None, None, None)

    # Normalize column names to lower-case
    df_rating.columns = [c.lower() for c in df_rating.columns]
    df_trend.columns = [c.lower() for c in df_trend.columns]
    df_email.columns = [c.lower() for c in df_email.columns] if not df_email.empty else df_email

    # Input validation
    errs = _validate_inputs(df_rating, df_trend, df_email)
    if errs:
        err_txt = " | ".join(errs)
        return (None, None, None, pd.DataFrame(), pd.DataFrame(), err_txt, None, None, None)

    # Preprocessing
    # Parse dates
    df_trend['date'] = pd.to_datetime(df_trend['date'], errors='coerce')
    # Fill missing numeric fields safely
    for col in ['total_volume', 'negative', 'positive', 'neutral']:
        if col in df_trend.columns:
            df_trend[col] = pd.to_numeric(df_trend[col], errors='coerce').fillna(0)
    df_trend['topic'] = df_trend['topic'].astype(str)

    df_rating['predicted_rating'] = pd.to_numeric(df_rating['predicted_rating'], errors='coerce')
    if 'user_id' in df_rating.columns:
        df_rating['user_id'] = df_rating['user_id'].astype(str)

    # ---------------------
    # A. TREND KPIs
    # ---------------------
    # Daily volume
    daily_vol = df_trend.groupby('date', as_index=False)['total_volume'].sum().sort_values('date')

    # Spike detection (mean + 1.5 * std)
    vol_mean = daily_vol['total_volume'].mean()
    vol_std = daily_vol['total_volume'].std() if daily_vol['total_volume'].std() == df_trend['total_volume'].std() else daily_vol['total_volume'].std()
    threshold = vol_mean + 1.5 * (vol_std if not np.isnan(vol_std) else 0)
    df_spikes = daily_vol[daily_vol['total_volume'] > threshold].copy()
    df_spikes['note'] = '🔥 High Volume Spike'

    # Topic negative rate and severity
    topic_stats = df_trend.groupby('topic', as_index=False).agg({
        'negative': 'sum',
        'total_volume': 'sum',
        'positive': 'sum'
    })
    # avoid division by zero
    topic_stats['neg_rate'] = np.where(topic_stats['total_volume'] > 0,
                                       topic_stats['negative'] / topic_stats['total_volume'] * 100, 0)
    topic_stats['severity'] = topic_stats['neg_rate'] * topic_stats['total_volume']  # simple score
    top_neg_topics = topic_stats.sort_values('neg_rate', ascending=False).head(10)

    # Topic growth rate (compare recent window)
    # compute last 7-day vs previous 7-day if possible
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
        # keep top growing topics
        topic_growth = growth_df.sort_values('growth_pct', ascending=False).head(5)
    except Exception:
        topic_growth = pd.DataFrame(columns=['topic','growth','growth_pct'])

    # ---------------------
    # B. RATING KPIs
    # ---------------------
    prod_stats = df_rating.groupby('product_id', as_index=False)['predicted_rating'].agg(['mean','count']).reset_index()
    prod_stats.columns = ['product_id', 'avg_rating', 'review_count']

    # rating distribution
    rating_dist = df_rating['predicted_rating'].describe().to_dict()

    # rating volatility (std) per product and overall
    prod_stats['rating_std'] = df_rating.groupby('product_id')['predicted_rating'].std().values

    # rating drift: compute weekly averages if date exists in df_rating (optional)
    rating_drift_summary = []
    if 'date' in df_rating.columns:
        try:
            df_rating['date'] = pd.to_datetime(df_rating['date'], errors='coerce')
            weekly = df_rating.set_index('date').groupby([pd.Grouper(freq='W'), 'product_id'])['predicted_rating'].mean().unstack(fill_value=np.nan)
            # compute drift last week vs prev week
            if weekly.shape[0] >= 2:
                last = weekly.iloc[-1]
                prev = weekly.iloc[-2]
                drift = ((last - prev) / prev.replace(0, np.nan)) * 100
                drift = drift.replace([np.inf, -np.inf], np.nan).fillna(0)
                drift_summary = drift.reset_index().rename(columns={drift.name: 'drift_pct'}) if hasattr(drift, 'name') else pd.DataFrame()
                rating_drift_summary = drift_summary
        except Exception:
            rating_drift_summary = pd.DataFrame()

    # risky products (avg_rating < 3.5)
    risky_products = prod_stats[prod_stats['avg_rating'] < 3.5].sort_values('avg_rating')

    # percent >=4.0 etc.
    prod_stats['pct_ge_4'] = df_rating.groupby('product_id')['predicted_rating'].apply(lambda x: (x >= 4.0).mean()).values * 100

    # user variance (if user_id exists)
    user_var = None
    if 'user_id' in df_rating.columns:
        try:
            user_std = df_rating.groupby('user_id')['predicted_rating'].std()
            user_var = user_std.mean()
        except Exception:
            user_var = None

    # ---------------------
    # C. EMAIL KPIs (if provided)
    # ---------------------
    email_summary = pd.DataFrame()
    complaint_summary_text = ""
    if not df_email.empty:
        # assume label column indicates categories
        if 'label' in df_email.columns:
            cat_counts = df_email['label'].value_counts().reset_index()
            cat_counts.columns = ['label', 'count']
            email_summary = cat_counts
            complaint_summary_text = f"Email categories: {cat_counts.to_dict(orient='records')}"
        # daily volumes
        if 'date' in df_email.columns:
            df_email['date'] = pd.to_datetime(df_email['date'], errors='coerce')
            email_daily = df_email.groupby('date').size().reset_index(name='count')
        else:
            email_daily = pd.DataFrame()

    # ---------------------
    # D. Cross-analytics / Business rules -> generate insights list
    # ---------------------
    insights = []
    insights.append("=== BÁO CÁO PHÂN TÍCH TỔNG HỢP (Rule-based) ===\n")
    avg_r = df_rating['predicted_rating'].mean()
    total_neg = df_trend['negative'].sum()
    total_vol = df_trend['total_volume'].sum() if df_trend['total_volume'].sum() > 0 else 1
    avg_n_rate = (total_neg / total_vol) * 100

    insights.append(f"- Rating trung bình toàn hệ thống: {avg_r:.2f}/5")
    insights.append(f"- Tỷ lệ tiêu cực trung bình (trend): {avg_n_rate:.1f}%")
    insights.append(f"- Tổng volume (trend): {int(total_vol)}")

    # rule examples
    # 1. Polarization (high avg rating but high negative)
    if avg_r > 4.0 and avg_n_rate > 30:
        insights.append("⚠️ Cảnh báo POLARIZATION: Rating cao nhưng dư luận tiêu cực nhiều — kiểm tra seeding / tranh cãi sản phẩm.")
    # 2. Crisis
    if avg_r < 3.0 and avg_n_rate > 50:
        insights.append("🚨 KHỦNG HOẢNG: Rating thấp & dư luận tiêu cực -> Xem xét dừng bán / thu hồi.")
    # 3. Spike insight
    if not df_spikes.empty:
        spike_dates = df_spikes['date'].dt.strftime('%Y-%m-%d').tolist()
        insights.append(f"🔥 Phát hiện spike volume tại các ngày: {', '.join(spike_dates)}")
    # 4. Worst topic
    if not top_neg_topics.empty:
        bad_topic_row = top_neg_topics.iloc[0]
        insights.append(f"❌ Chủ đề nhiều than phiền nhất: {bad_topic_row['topic']} (neg_rate={bad_topic_row['neg_rate']:.1f}%, volume={int(bad_topic_row['total_volume'])})")
    # 5. Risky products
    if not risky_products.empty:
        sample = risky_products.head(5)
        insights.append(f"🚨 Các sản phẩm cần kiểm tra (rating < 3.5): {', '.join(sample['product_id'].astype(str).tolist())}")
    # 6. Email-linked rules
    if not df_email.empty and 'label' in df_email.columns:
        if 'complaint' in df_email['label'].str.lower().unique().tolist():
            comp_count = df_email['label'].str.lower().value_counts().get('complaint', 0)
            insights.append(f"- Email complaint count: {int(comp_count)}")
    # 7. Topic growth highlights
    if isinstance(topic_growth, pd.DataFrame) and not topic_growth.empty:
        top_growth = topic_growth.iloc[0]
        insights.append(f"📈 Topic tăng mạnh: {top_growth['topic']} (growth% ≈ {top_growth['growth_pct']:.1f}%)")

    insights_text = "\n".join(insights)

    # ---------------------
    # E. VISUALIZATIONS (matplotlib figures)
    # ---------------------
    # Fig 1: Daily volume with spikes
    fig_trend, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=daily_vol, x='date', y='total_volume', marker='o', ax=ax)
    ax.set_title('Daily Discussion Volume')
    ax.set_ylabel('Total Volume')
    if not df_spikes.empty:
        ax.scatter(df_spikes['date'], df_spikes['total_volume'], color='red', s=80, zorder=5)
        for idx, row in df_spikes.iterrows():
            ax.annotate('Spike', (row['date'], row['total_volume']), textcoords="offset points", xytext=(0,8), ha='center', color='red')
    ax.grid(alpha=0.3)
    fig_trend.tight_layout()

    # Fig 2: Rating distribution
    fig_rating, ax2 = plt.subplots(figsize=(6, 4))
    sns.histplot(df_rating['predicted_rating'].dropna(), bins=20, kde=True, ax=ax2)
    ax2.axvline(3.5, color='red', linestyle='--', label='Risk threshold 3.5')
    ax2.set_title('Rating Distribution')
    ax2.set_xlabel('Predicted Rating')
    ax2.legend()
    fig_rating.tight_layout()

    # Fig 3: Top negative topics (bar)
    fig_topic, ax3 = plt.subplots(figsize=(8, 4))
    plot_df = top_neg_topics.head(10).sort_values('neg_rate')
    sns.barplot(data=plot_df, x='neg_rate', y='topic', ax=ax3)
    ax3.set_xlabel('Negative Rate (%)')
    ax3.set_title('Top Negative Topics')
    fig_topic.tight_layout()

    # ---------------------
    # F. EXPORT FILES (TXT, CSV, PDF, HTML)
    # ---------------------
    # txt insights
    txt_path = os.path.join(GRADIO_TEMP_DIR, f"insights_{int(time.time())}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(insights_text + "\n\n")
        f.write("--- Top risk products ---\n")
        if not risky_products.empty:
            f.write(risky_products.to_string(index=False))
        else:
            f.write("None\n")
        f.write("\n\n--- Top negative topics ---\n")
        f.write(top_neg_topics.to_string(index=False))

    # csv summary (risky products + top topics merged for convenience)
    csv_path = os.path.join(GRADIO_TEMP_DIR, f"summary_{int(time.time())}.csv")
    # create a merged summary table (left join)
    summary_prod = prod_stats.copy()
    # add top negative topic aggregated info for the whole dataset as columns
    top_topics_small = top_neg_topics[['topic','neg_rate','total_volume']].head(5)
    # save risky products as main summary
    risky_products.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # PDF report (compose figs + a text page)
    pdf_path = os.path.join(GRADIO_TEMP_DIR, f"report_{int(time.time())}.pdf")
    with PdfPages(pdf_path) as pdf:
        # page 1: text summary (rendered as figure)
        fig_txt = plt.figure(figsize=(8.27, 11.69))  # A4
        fig_txt.text(0.01, 0.99, "Dashboard Report (Rule-based)\n\n", fontsize=14, weight='bold', va='top')
        wrapped = insights_text
        fig_txt.text(0.01, 0.95, wrapped, fontsize=10, va='top')
        pdf.savefig(fig_txt)
        plt.close(fig_txt)
        # append charts
        pdf.savefig(fig_trend)
        pdf.savefig(fig_rating)
        pdf.savefig(fig_topic)

    # HTML report with embedded images
    img1 = _save_fig(fig_trend, "trend")
    img2 = _save_fig(fig_rating, "rating")
    img3 = _save_fig(fig_topic, "topic")
    html_path = _create_html_report([img1, img2, img3], insights_text, csv_path)

    # Return objects expected by Gradio wiring
    # NOTE: Gradio Plot can accept matplotlib figures
    # We'll return figures, DataFrames, text, and file paths for download
    return (
        fig_trend,                             # plot_trend_view
        fig_rating,                            # plot_rating_view
        fig_topic,                             # plot_topic_view
        df_spikes[['date', 'total_volume', 'note']].reset_index(drop=True),  # df_spike_view
        risky_products.reset_index(drop=True), # df_risky_view
        insights_text,                         # txt_insights_view
        txt_path,                              # dl_txt
        csv_path,                              # dl_csv
        pdf_path,                              # additional (if you want show)
        html_path                              # additional
    )
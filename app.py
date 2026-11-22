# キャベツ価格予測 — 表示画面 (Phase 4)
# results/latest.json と data/weekly.csv を読み込んで表示するだけの画面。
# 予測の計算は predict.py が担当し、この画面は計算しない。
import json
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent

st.set_page_config(page_title="キャベツ価格予測", page_icon="🥬")
st.title("🥬 キャベツ価格予測")
st.caption("大田市場・相対取引・中値 / 10kgあたり / 東京都オープンデータ + Open-Meteo")

latest_path = BASE / "results" / "latest.json"
weekly_path = BASE / "data" / "weekly.csv"
history_path = BASE / "results" / "prediction_history.csv"

if not latest_path.exists():
    st.warning("まだ予測結果がありません。GitHub Actions の自動実行が1回成功すると、ここに結果が出ます。")
    st.stop()

latest = json.loads(latest_path.read_text(encoding="utf-8"))

pred = latest["predicted_next_week_yen"]
now_price = latest["latest_price_yen"]
diff = pred - now_price

col1, col2, col3 = st.columns(3)
col1.metric("最新の週の価格", f"{now_price:,} 円")
col2.metric("来週の予測", f"{pred:,} 円", delta=f"{diff:+,} 円")
col3.metric("モデルの誤差 (MAE)", f"{latest['model_mae_yen']:,} 円",
            help=f"単純な予想(来週=今週)の誤差は {latest['baseline_mae_yen']:,} 円")

if diff > 30:
    st.info("傾向：来週は **上がりそう** です。")
elif diff < -30:
    st.info("傾向：来週は **下がりそう** です。")
else:
    st.info("傾向：来週は **横ばい** の見込みです。")

st.caption(f"基準の週：{latest['base_week']} / 一番効いた材料：{latest['top_feature']} / 更新：{latest['generated_at']}")

if weekly_path.exists():
    st.subheader("価格の推移（週ごと）")
    weekly = pd.read_csv(weekly_path, parse_dates=["week"])
    chart = weekly.set_index("week")["price"].rename("価格 (円/10kg)")
    st.line_chart(chart)

if history_path.exists():
    st.subheader("予測の履歴")
    hist = pd.read_csv(history_path)
    st.dataframe(hist, use_container_width=True)

st.divider()
st.caption("この予測は「傾向を見る道具」です。誤差は10kgあたり100円前後あります。急騰・急落は外しやすいので、現場の感覚と合わせて使ってください。")

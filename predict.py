# キャベツ価格予測 — 自動実行用スクリプト (Phase 3)
# 価格データ(東京都オープンデータ)と天気(Open-Meteo)を毎回取り直し、
# 週ごとの表を作って学習し、来週の価格を予測して results/ に書き出す。
import json
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

JST = timezone(timedelta(hours=9))
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ---------- 1. 価格データを取る ----------
urls = [
    "https://www.opendata.metro.tokyo.lg.jp/shijyou/2024/result_price_fresh.csv",
    "https://www.opendata.metro.tokyo.lg.jp/shijyou/2025/result_price_fresh.csv",
    "https://www.opendata.metro.tokyo.lg.jp/shijyou/2026/result_price_fresh.csv",
]
frames = [pd.read_csv(u, encoding="utf-8-sig") for u in urls]
p = pd.concat(frames, ignore_index=True)
print("価格データ行数:", len(p))

# ---------- 2. キャベツだけ抜き出す(大田市場・相対・中値) ----------
c = p[(p["品名"] == "キャベツ") & (p["市場"] == "大田") & (p["販売方法"] == "相対")].copy()
c["日付"] = pd.to_datetime(c["日付"])
c["中値(円)"] = pd.to_numeric(c["中値(円)"], errors="coerce")
daily_price = c.groupby("日付")["中値(円)"].mean().rename("price").reset_index()
print("価格の日数:", len(daily_price), daily_price["日付"].min(), "→", daily_price["日付"].max())

# ---------- 3. 天気データを取る(東京) ----------
today = date.today()
url = ("https://archive-api.open-meteo.com/v1/archive"
       "?latitude=35.68&longitude=139.69"
       f"&start_date=2024-04-01&end_date={today.isoformat()}"
       "&daily=temperature_2m_mean,precipitation_sum,sunshine_duration"
       "&timezone=Asia%2FTokyo")
with urllib.request.urlopen(url) as r:
    w = json.load(r)
wd = pd.DataFrame({
    "date": pd.to_datetime(w["daily"]["time"]),
    "temp": w["daily"]["temperature_2m_mean"],
    "rain": w["daily"]["precipitation_sum"],
    "sun": pd.to_numeric(w["daily"]["sunshine_duration"], errors="coerce") / 3600.0,
})

# ---------- 4. 週ごとにまとめて材料を作る ----------
df = daily_price.merge(wd, left_on="日付", right_on="date", how="inner").drop(columns=["date"])
df = df.set_index("日付").sort_index()
weekly = df.resample("W-MON").agg(price=("price", "mean"), temp=("temp", "mean"),
                                  rain=("rain", "sum"), sun=("sun", "sum"))
weekly["price_lag1"] = weekly["price"].shift(1)
weekly["price_lag4"] = weekly["price"].shift(4)
weekly["temp_lag1"] = weekly["temp"].shift(1)
weekly["rain_lag1"] = weekly["rain"].shift(1)
weekly["month"] = weekly.index.month
weekly["weekofyear"] = weekly.index.isocalendar().week.astype(int)
weekly["target_next"] = weekly["price"].shift(-1)  # 来週の価格が「答え」
weekly_out = weekly.reset_index().rename(columns={"日付": "week"})
weekly_out.to_csv(DATA_DIR / "weekly.csv", index=False)
print("週数:", len(weekly))

feats = ["price_lag1", "price_lag4", "temp", "temp_lag1", "rain", "rain_lag1",
         "sun", "month", "weekofyear"]
with_feats = weekly.dropna(subset=feats)
labeled = with_feats.dropna(subset=["target_next"])
print("学習に使える週数:", len(labeled))

# ---------- 5. 学習と腕試し(直近12週をテストにする) ----------
test = labeled.iloc[-12:]
train = labeled.iloc[:-12]
model = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
model.fit(train[feats], train["target_next"])
pred = model.predict(test[feats])
mae = mean_absolute_error(test["target_next"], pred)
baseline = mean_absolute_error(test["target_next"], test["price_lag1"])
importances = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
print(f"モデルの誤差(MAE): {mae:.0f} 円/10kg")
print(f"単純な予想の誤差: {baseline:.0f} 円/10kg")
print("一番効いた材料:", importances.index[0], round(float(importances.iloc[0]), 3))

# ---------- 6. 来週の予測を出す ----------
last = with_feats.iloc[[-1]][feats]
next_pred = float(model.predict(last)[0])
base_week = with_feats.index[-1].date()
latest_price = float(with_feats["price"].iloc[-1])
print(f"最新の週({base_week})の価格: {latest_price:.0f} 円/10kg")
print(f"来週の予測: {next_pred:.0f} 円/10kg")

# ---------- 結果を保存 ----------
now_jst = datetime.now(JST)
latest = {
    "item": "キャベツ(大田市場・相対・中値, 10kgあたり)",
    "base_week": str(base_week),
    "latest_price_yen": round(latest_price),
    "predicted_next_week_yen": round(next_pred),
    "model_mae_yen": round(float(mae)),
    "baseline_mae_yen": round(float(baseline)),
    "top_feature": str(importances.index[0]),
    "generated_at": now_jst.strftime("%Y-%m-%d %H:%M JST"),
}
(RESULTS_DIR / "latest.json").write_text(
    json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")

hist_path = RESULTS_DIR / "prediction_history.csv"
row = pd.DataFrame([{
    "run_date": now_jst.strftime("%Y-%m-%d"),
    "base_week": str(base_week),
    "latest_price_yen": round(latest_price),
    "predicted_next_week_yen": round(next_pred),
    "model_mae_yen": round(float(mae)),
    "baseline_mae_yen": round(float(baseline)),
}])
if hist_path.exists():
    hist = pd.read_csv(hist_path)
    hist = hist[hist["base_week"] != str(base_week)]
    hist = pd.concat([hist, row], ignore_index=True).sort_values("base_week")
else:
    hist = row
hist.to_csv(hist_path, index=False)
print("保存しました: data/weekly.csv, results/latest.json, results/prediction_history.csv")

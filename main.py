import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="영화 흥행 예측기", layout="wide")
st.title("🎬 영화 흥행 예측기")
st.caption("박스오피스 데이터와 영화 정보를 결합해 총 관객 수를 예측합니다.")

# ----------------------------
# 데이터 불러오기
# ----------------------------
DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

@st.cache_data
def load_data():
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")
    return daily, movies

daily_df, movies_df = load_data()

# ----------------------------
# 기준 기간 (일별 데이터의 최소~최대 날짜)
# ----------------------------
date_col = daily_df["날짜"].astype(str)
min_date = date_col.min()
max_date = date_col.max()

def fmt_date(d):
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

st.info(f"📅 기준 기간: {fmt_date(min_date)} ~ {fmt_date(max_date)}")

# ----------------------------
# 두 표 병합 (movieCd 기준)
# 영화별 표(movies_df)에 있는 영화를 모두 사용
# ----------------------------
merged = movies_df.copy()

# 일별 데이터에서 영화코드별 요약 정보(스크린수/상영횟수 평균 등)를 추가로 붙일 수도 있으나
# 영화별 표에 이미 first_scrn, first_show 등이 있으므로 movies_df를 기준(all)으로 사용
# daily_df와의 병합 확인용으로 movieCd, movieNm이 일치하는지만 체크
merged = merged.merge(
    daily_df[["영화코드", "영화명"]].drop_duplicates(subset="영화코드"),
    left_on="movieCd", right_on="영화코드", how="left"
)

# ----------------------------
# 영화코드 순 정렬 후 학습/시험 분리
# 열 편마다 앞의 세 편을 시험용으로 사용
# ----------------------------
merged = merged.sort_values("movieCd").reset_index(drop=True)
merged["is_test"] = (merged.index % 10) < 3

train_df = merged[~merged["is_test"]].copy()
test_df = merged[merged["is_test"]].copy()

# ----------------------------
# peak 컬럼 처리 (이미 0/1로 존재)
# openDt에서 개봉월 추출 (참고용, peak가 이미 있으므로 필수는 아님)
# ----------------------------
merged["openDt"] = merged["openDt"].astype(str)

# ----------------------------
# 사용 가능한 수치형 특성 후보
# ----------------------------
candidate_features = {
    "first_scrn": "첫 관측일 스크린수",
    "first_show": "첫 관측일 상영횟수",
    "peak": "성수기 개봉 여부(1/0)",
    "first_week_audi": "첫 주 관객",
    "days_in_top10": "톱10 유지일수",
}

st.subheader("📌 예측에 사용할 변수 선택")
st.write("사용할 특성을 체크하세요. (최소 1개 이상 선택)")

cols = st.columns(len(candidate_features))
selected_features = []
for i, (col_name, label) in enumerate(candidate_features.items()):
    with cols[i]:
        checked = st.checkbox(label, value=True, key=f"chk_{col_name}")
        if checked:
            selected_features.append(col_name)

if len(selected_features) == 0:
    st.warning("⚠️ 최소 한 개 이상의 변수를 선택해야 합니다.")
    st.stop()

# ----------------------------
# 결측치 제거 (선택된 특성 + 타깃 기준)
# ----------------------------
needed_cols = selected_features + ["total_audi"]
train_clean = train_df.dropna(subset=needed_cols).copy()
test_clean = test_df.dropna(subset=needed_cols).copy()

if len(train_clean) < 2 or len(test_clean) < 1:
    st.error("❌ 선택한 변수 조합으로는 학습/시험에 사용할 데이터가 부족합니다. 다른 변수를 선택해보세요.")
    st.stop()

X_train = train_clean[selected_features].values
y_train = train_clean["total_audi"].values
X_test = test_clean[selected_features].values
y_test = test_clean["total_audi"].values

# ----------------------------
# 모델 학습
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# ----------------------------
# 결과 요약 출력
# ----------------------------
st.subheader("📊 모델 학습 결과")

c1, c2, c3 = st.columns(3)
c1.metric("학습에 쓴 영화 편수", f"{len(train_clean)} 편")
c2.metric("점수를 잰 영화 편수 (시험용)", f"{len(test_clean)} 편")
c3.metric("결정계수 (R²)", f"{r2:.3f}")

st.write(f"**평균 절대 오차 (MAE)**: 예측이 실제와 평균적으로 **{mae:,.0f} 명** 정도 차이 납니다.")

# ----------------------------
# 1,000명 미만 예측값 바닥 처리
# ----------------------------
FLOOR = 1000
floor_mask = y_pred < FLOOR
n_floored = int(floor_mask.sum())

y_pred_plot = np.where(y_pred < FLOOR, FLOOR, y_pred)
y_test_plot = np.where(y_test < FLOOR, FLOOR, y_test)  # 실제값도 로그 축을 위해 방어적으로 처리

st.write(f"🔻 예측 관객 수가 **1,000명 미만**으로 나온 영화: **{n_floored} 편** (그래프 하단에 붙여 표시됨)")

# ----------------------------
# Plotly 산점도 (로그-로그, 기준선 포함)
# ----------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=y_test_plot,
    y=y_pred_plot,
    mode="markers",
    name="시험용 영화",
    text=test_clean.get("movieNm_x", test_clean.get("movieNm", "")),
    hovertemplate="영화: %{text}<br>실제: %{x:,.0f}<br>예측: %{y:,.0f}<extra></extra>",
    marker=dict(
        color=np.where(floor_mask, "red", "royalblue"),
        size=9,
        opacity=0.7,
        line=dict(width=1, color="black")
    )
))

# 기준선 (예측 = 실제)
min_val = min(y_test_plot.min(), y_pred_plot.min())
max_val = max(y_test_plot.max(), y_pred_plot.max())
fig.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode="lines",
    name="예측 = 실제 (기준선)",
    line=dict(color="gray", dash="dash")
))

fig.update_xaxes(type="log", title="실제 총 관객 수 (로그 스케일)")
fig.update_yaxes(type="log", title="예측 총 관객 수 (로그 스케일)")
fig.update_layout(
    title="시험용 영화: 실제 vs 예측 총 관객 수",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 참고: 사용된 특성과 회귀 계수
# ----------------------------
with st.expander("🔍 회귀 계수 살펴보기"):
    coef_df = pd.DataFrame({
        "변수": selected_features,
        "계수": model.coef_
    })
    st.dataframe(coef_df, use_container_width=True)
    st.write(f"절편(intercept): {model.intercept_:,.2f}")

with st.expander("📄 시험용 영화 상세 데이터 보기"):
    display_cols = ["movieCd"] + selected_features + ["total_audi"]
    show_df = test_clean[display_cols].copy()
    show_df["예측값"] = y_pred
    st.dataframe(show_df, use_container_width=True)

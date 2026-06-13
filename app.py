import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Kaliforniya Uy-joy Tahlili", layout="wide")


@st.cache_data
def load_data():return pd.read_csv("california_housing_test.csv")


df = load_data()

LABEL_MAP = {
    "longitude": "Uzunlik (longitude)",
    "latitude": "Kenglik (latitude)",
    "housing_median_age": "Uyning o'rtacha yoshi (yil)",
    "total_rooms": "Xonalar umumiy soni",
    "total_bedrooms": "Yotoqxonalar umumiy soni",
    "population": "Aholi soni",
    "households": "Xonadonlar soni",
    "median_income": "O'rtacha daromad ($)",
    "median_house_value": "Uy narxi ($)",
}


def to_label(col):return LABEL_MAP.get(col, col)


def from_label(label):
    for k, v in LABEL_MAP.items():
        if v == label:return k
    return label


st.sidebar.title("Navigatsiya")
st.sidebar.markdown("Kerakli bo'limni tanlang:")
st.sidebar.markdown("**Bo'limlar:**")
page = st.sidebar.radio(
    label="",
    options=["EDA Tahlil", "Bashorat (Model)", "Ma'lumot Haqida"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.info(f"Datasetda jami: **{len(df):,} ta** faol qator mavjud.")

if page == "EDA Tahlil":
    st.title("Kaliforniya Uy-joy Narxlari Tahlili")
    st.header("Ma'lumotlarni Tahlil qilish va Vizuallashtirish (EDA)")

    st.subheader("Geografik xarita")
    st.write(
        "Uylarning joylashuvi va narxlari."
        " Rang qancha 'issiq' bo'lsa, narx shuncha yuqori. Okean bo'yidagi"
        " hududlarda narxlar odatda balandroq."
    )

    color_options = ["median_house_value", "median_income", "housing_median_age", "population"]
    selected_label = st.selectbox(
        "Rang bo'yicha ko'rsatish uchun parametrni tanlang:",
        [to_label(c) for c in color_options], index=0
    )
    color_metric = from_label(selected_label)

    fig_map = px.scatter_mapbox(
        df, lat="latitude", lon="longitude",
        color=color_metric, size="population",
        color_continuous_scale=px.colors.sequential.Viridis,
        size_max=15, zoom=4.5, mapbox_style="carto-positron",
        hover_data=["median_income", "housing_median_age", "median_house_value"],
        labels=LABEL_MAP,
    )
    fig_map.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown("---")
    st.subheader("Korrelyatsiya matritsasi")
    st.write(
        "Qaysi faktorlar uy narxiga eng ko'p ta'sir qiladi? "
        "Qiymat +1 ga yaqin bo'lsa — kuchli musbat bog'liqlik, "
        "-1 ga yaqin bo'lsa — kuchli teskari bog'liqlik."
    )

    corr = df.corr(numeric_only=True)
    corr_display = corr.rename(index=LABEL_MAP, columns=LABEL_MAP)
    fig_corr = px.imshow(
        corr_display, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto", zmin=-1, zmax=1,
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

    income_corr = corr.loc["median_income", "median_house_value"]
    st.success(
        f"**O'rtacha daromad** va **Uy narxi** orasidagi korrelyatsiya: "
        f"**{income_corr:.2f}** — bu eng kuchli bog'liqlik bo'lib, daromad qancha "
        f"yuqori bo'lsa, uy narxi ham shuncha yuqori bo'lishini ko'rsatadi."
    )

    st.markdown("---")
    st.subheader("Taqsimotlar (Distributions)")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{to_label('housing_median_age')}**")
        fig_age = px.histogram(
            df, x="housing_median_age", nbins=30,
            labels=LABEL_MAP, color_discrete_sequence=["#4C78A8"]
        )
        fig_age.update_layout(yaxis_title="Soni")
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        st.markdown(f"**{to_label('population')}**")
        fig_pop = px.histogram(
            df, x="population", nbins=30,
            labels=LABEL_MAP, color_discrete_sequence=["#F58518"]
        )
        fig_pop.update_layout(yaxis_title="Soni")
        st.plotly_chart(fig_pop, use_container_width=True)

    st.markdown("##### Qo'shimcha: barcha parametrlar taqsimoti")
    selected_extra_label = st.selectbox(
        "Parametrni tanlang:",
        [to_label(c) for c in df.columns],
        index=list(df.columns).index("median_house_value")
    )
    selected_col = from_label(selected_extra_label)
    fig_extra = px.histogram(
        df, x=selected_col, nbins=40,
        labels=LABEL_MAP,
        color_discrete_sequence=["#54A24B"]
    )
    fig_extra.update_layout(yaxis_title="Soni", height=400)
    st.plotly_chart(fig_extra, use_container_width=True)


elif page == "Bashorat (Model)":
    st.title("Uy Narxini Bashorat Qilish")

    st.subheader("Yangi xususiyatlar (Feature Engineering)")
    st.write(
        "Mavjud ustunlardan modelni yaxshilash uchun 3 ta yangi, "
        "ma'noliroq parametr yaratamiz:"
    )

    df_model = df.copy()
    df_model["rooms_per_household"] = df_model["total_rooms"] / df_model["households"]
    df_model["bedrooms_per_room"] = df_model["total_bedrooms"] / df_model["total_rooms"]
    df_model["population_per_household"] = df_model["population"] / df_model["households"]

    new_features_labels = {
        "rooms_per_household": "Xonadonga to'g'ri keladigan xonalar soni",
        "bedrooms_per_room": "Yotoqxonalar ulushi (jami xonalardan)",
        "population_per_household": "Xonadondagi o'rtacha odam soni",
    }

    st.dataframe(
        df_model[list(new_features_labels.keys())]
        .rename(columns=new_features_labels)
        .head(10),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Modelni o'qitish (Random Forest)")

    feature_cols = [
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income",
        "rooms_per_household", "bedrooms_per_room", "population_per_household"
    ]
    target_col = "median_house_value"

    X = df_model[feature_cols]
    y = df_model[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    @st.cache_resource
    def train_model(X_train, y_train):
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        return model

    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    col1.metric("O'rtacha xato (MAE)", f"${mae:,.0f}")
    col2.metric("R² ko'rsatkichi (aniqlik)", f"{r2:.3f}")

    st.caption(
        "MAE — modelning o'rtacha qancha dollarga adashishi. "
        "R² — 1 ga yaqin bo'lsa, model shuncha yaxshi ishlaydi."
    )

    st.markdown("##### Qaysi parametrlar narxga ko'proq ta'sir qiladi?")

    all_labels = {**LABEL_MAP, **new_features_labels}
    importance_df = pd.DataFrame({
        "Parametr": [all_labels.get(c, c) for c in feature_cols],
        "Ahamiyati": model.feature_importances_
    }).sort_values("Ahamiyati", ascending=False)

    fig_imp = px.bar(importance_df, x="Ahamiyati", y="Parametr", orientation="h")
    fig_imp.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.subheader("O'zingiz uchun bashorat qiling")
    st.write("Quyidagi qiymatlarni kiriting va tugmani bosing — model uy narxini taxmin qiladi.")

    c1, c2, c3 = st.columns(3)

    with c1:
        longitude = st.slider(
            "Uzunlik (longitude)",
            float(df["longitude"].min()), float(df["longitude"].max()),
            float(df["longitude"].mean())
        )
        latitude = st.slider(
            "Kenglik (latitude)",
            float(df["latitude"].min()), float(df["latitude"].max()),
            float(df["latitude"].mean())
        )
        housing_median_age = st.slider("Uyning o'rtacha yoshi (yil)", 1, 52, 25)

    with c2:
        total_rooms = st.number_input(
            "Xonalar umumiy soni", min_value=1, value=int(df["total_rooms"].median())
        )
        total_bedrooms = st.number_input(
            "Yotoqxonalar umumiy soni", min_value=1, value=int(df["total_bedrooms"].median())
        )
        population = st.number_input(
            "Aholi soni", min_value=1, value=int(df["population"].median())
        )

    with c3:
        households = st.number_input(
            "Xonadonlar soni", min_value=1, value=int(df["households"].median())
        )
        median_income = st.slider("O'rtacha daromad (10,000$ birligida)", 0.5, 15.0, 3.8)

    if st.button("Narxni bashorat qilish"):
        rooms_per_household = total_rooms / households
        bedrooms_per_room = total_bedrooms / total_rooms
        population_per_household = population / households

        input_data = pd.DataFrame([{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "rooms_per_household": rooms_per_household,
            "bedrooms_per_room": bedrooms_per_room,
            "population_per_household": population_per_household,
        }])

        prediction = model.predict(input_data)[0]
        st.success(f"Taxminiy uy narxi: **${prediction:,.0f}**")


# ============================================================
# 3-SAHIFA: MA'LUMOT HAQIDA
# ============================================================
else:
    st.title("Dataset Haqida")
    st.markdown("""
    **California Housing Test** dataseti — Kaliforniya shtatidagi turli hududlar
    (blok guruhlari) bo'yicha uy-joy statistikasi.

    | Ustun | Tavsif |
    |---|---|
    | `longitude` | Uzunlik (geografik koordinata) |
    | `latitude` | Kenglik (geografik koordinata) |
    | `housing_median_age` | Uylarning o'rtacha yoshi (yil) |
    | `total_rooms` | Hududdagi xonalar umumiy soni |
    | `total_bedrooms` | Yotoqxonalar umumiy soni |
    | `population` | Aholi soni |
    | `households` | Xonadonlar soni |
    | `median_income` | O'rtacha daromad (10,000$ birligida) |
    | `median_house_value` | O'rtacha uy narxi ($) — **maqsadli (target) ustun** |
    """)
    st.subheader("Umumiy statistika")
    st.dataframe(df.describe(), use_container_width=True)
    st.subheader("Xom ma'lumotlar (birinchi 100 qator)")
    st.dataframe(df.head(100), use_container_width=True)
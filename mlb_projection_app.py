import streamlit as st
import pandas as pd
from xgboost import XGBRegressor

st.set_page_config(
    page_title="MLB Projection Engine",
    layout="centered"
)

FEATURES = [
    "age",
    "age2",
    "G",
    "PA",
    "AVG",
    "HR_rate",
    "BB_rate",
    "SO_rate",
    "SB_rate",
    "weighted_HR_rate",
    "weighted_BB_rate",
    "weighted_SO_rate",
    "weighted_SB_rate",
    "weighted_AVG",
    "weighted_PA",
    "weighted_G"
]

TARGETS = {
    "HR": "next_HR",
    "RBI": "next_RBI",
    "BB": "next_BB",
    "SO": "next_SO",
    "SB": "next_SB",
    "H": "next_H",
    "AB": "next_AB"
}


@st.cache_data
def load_data():
    batting = pd.read_csv("Batting.csv")
    people = pd.read_csv("People.csv")

    people["birthYear"] = pd.to_numeric(
        people["birthYear"],
        errors="coerce"
    )

    batting = batting.merge(
        people[["playerID", "nameFirst", "nameLast", "birthYear"]],
        on="playerID",
        how="left"
    )

    batting = batting[batting["AB"] >= 100].copy()
    batting = batting.sort_values(["playerID", "yearID"])

    # Age features
    batting["age"] = pd.to_numeric(
        batting["yearID"] - batting["birthYear"],
        errors="coerce"
    )

    batting["age2"] = batting["age"] ** 2

    # Estimated plate appearances
    batting["PA"] = pd.to_numeric(
        batting["AB"]
        + batting["BB"]
        + batting["HBP"]
        + batting["SF"]
        + batting["SH"],
        errors="coerce"
    )

    # Rate stats
    batting["HR_rate"] = batting["HR"] / batting["PA"]
    batting["BB_rate"] = batting["BB"] / batting["PA"]
    batting["SO_rate"] = batting["SO"] / batting["PA"]
    batting["SB_rate"] = batting["SB"] / batting["PA"]
    batting["AVG"] = batting["H"] / batting["AB"]

    stats = [
        "HR_rate",
        "BB_rate",
        "SO_rate",
        "SB_rate",
        "AVG",
        "PA",
        "G"
    ]

    # Weighted recency features
    for stat in stats:
        batting[f"lag1_{stat}"] = batting.groupby("playerID")[stat].shift(1)
        batting[f"lag2_{stat}"] = batting.groupby("playerID")[stat].shift(2)
        batting[f"lag3_{stat}"] = batting.groupby("playerID")[stat].shift(3)

        batting[f"weighted_{stat}"] = (
            batting[f"lag1_{stat}"] * 0.5
            + batting[f"lag2_{stat}"] * 0.3
            + batting[f"lag3_{stat}"] * 0.2
        )

    # Projection targets
    for stat in ["HR", "RBI", "BB", "SO", "SB", "H", "AB"]:
        batting[f"next_{stat}"] = (
            batting.groupby("playerID")[stat].shift(-1)
        )

    numeric_cols = FEATURES + list(TARGETS.values())

    batting[numeric_cols] = batting[numeric_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    batting = batting.dropna()

    batting["fullName"] = (
        batting["nameFirst"] + " " + batting["nameLast"]
    )

    return batting


@st.cache_resource
def train_models(df):
    models = {}

    X = df[FEATURES].astype(float)

    for stat, target in TARGETS.items():
        y = df[target].astype(float)

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X, y)
        models[stat] = model

    return models


# Load data + train
batting = load_data()
models = train_models(batting)

# Active player filter
latest_year = batting["yearID"].max()

active_players = (
    batting.groupby("fullName")["yearID"]
    .max()
    .reset_index()
)

active_players = active_players[
    active_players["yearID"] >= latest_year - 2
]

player_names = sorted(active_players["fullName"].unique())

# UI
st.title("⚾ MLB Hitter Projection Engine")
st.caption("Machine learning next-season hitter projections")

selected_player = st.selectbox(
    "Choose a player",
    player_names
)

if st.button("Generate Projection"):
    player = batting[
        batting["fullName"] == selected_player
    ]

    latest = player.sort_values("yearID").iloc[-1]

    sample = latest[FEATURES].to_frame().T.astype(float)

    projections = {}

    for stat, model in models.items():
        pred = model.predict(sample)[0]
        projections[stat] = round(pred)

    avg = (
        projections["H"] / projections["AB"]
        if projections["AB"] > 0 else 0
    )

    obp = (
        (projections["H"] + projections["BB"])
        / (projections["AB"] + projections["BB"])
        if (projections["AB"] + projections["BB"]) > 0
        else 0
    )

    st.subheader(f"{selected_player} Projection")

    col1, col2, col3 = st.columns(3)
    col1.metric("AVG", f"{avg:.3f}")
    col2.metric("HR", projections["HR"])
    col3.metric("RBI", projections["RBI"])

    col4, col5, col6 = st.columns(3)
    col4.metric("BB", projections["BB"])
    col5.metric("SO", projections["SO"])
    col6.metric("SB", projections["SB"])

    st.metric("Estimated OBP", f"{obp:.3f}")



"""Used Car Price Estimator – India (Streamlit app)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.explain import Explainer  # noqa: E402
from src.features import FEATURES  # noqa: E402
from src.predict import (build_input, confidence_warnings, find_similar,  # noqa: E402
                         load_artifacts, predict_price, price_range, verdict)

st.set_page_config(page_title="Used Car Price Estimator – India", page_icon="🚗", layout="wide")

OWNERS = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]


# ---------- loading ----------
@st.cache_resource(show_spinner="Loading model...")
def get_art():
    return load_artifacts()


@st.cache_resource(show_spinner=False)
def get_explainer(_model):
    return Explainer(_model)


@st.cache_data
def read_csv(rel_path):
    p = ROOT / rel_path
    return pd.read_csv(p) if p.exists() else None


def read_json(rel_path):
    p = ROOT / rel_path
    return json.load(open(p)) if p.exists() else None


def lakh(x):
    return f"₹{x / 1e5:,.2f} L"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def fmt_value(v):
    return f"{v:,.1f}" if isinstance(v, (int, float, np.number)) else str(v)


art = get_art()
REF = int(art["ref_year"])
listings = art["listings"].copy()
listings["car_age"] = REF - listings["year"]
listings["price_lakh"] = listings["selling_price"] / 1e5


def defaults_for(brand, model):
    """Typical specs of the chosen model, used to pre-fill the form."""
    pool = listings[(listings["brand"] == brand) & (listings["model"] == model)]
    if len(pool) < 3:
        pool = listings[listings["brand"] == brand]

    def med(col, fallback):
        s = pool[col].dropna()
        return float(s.median()) if len(s) else fallback

    def mode(col, fallback):
        s = pool[col].dropna()
        return s.mode().iloc[0] if len(s) else fallback

    return {"year": int(med("year", REF - 5)), "km": int(med("km_driven", 60000)),
            "engine": int(med("engine_cc", 1200)), "power": round(med("max_power_bhp", 80.0), 1),
            "mileage": round(med("mileage_kmpl", 18.0), 1), "torque": round(med("torque_nm", 110.0), 1),
            "seats": int(med("seats", 5)), "fuel": mode("fuel", "Petrol"),
            "transmission": mode("transmission", "Manual")}


# ---------- page 1: price estimator ----------
def page_estimator():
    st.title("🚗 Used Car Price Estimator – India")
    st.write("Enter a car's details to get an estimated **listing-market value**, a likely range, "
             "and the reasons behind the estimate.")

    brands = listings["brand"].value_counts().index.tolist()
    c1, c2 = st.columns(2)
    brand = c1.selectbox("Brand", brands)
    models = listings.loc[listings["brand"] == brand, "model"].value_counts().index.tolist()
    model_name = c2.selectbox("Model", models)
    d = defaults_for(brand, model_name)
    k = f"{brand}|{model_name}"          # new widget keys per car, so defaults refresh when the model changes

    fuels = sorted(listings["fuel"].dropna().unique())
    trans = sorted(listings["transmission"].dropna().unique())
    sellers = sorted(listings["seller_type"].dropna().unique())

    c1, c2, c3 = st.columns(3)
    year = c1.number_input("Manufacturing year", 1995, REF, clamp(d["year"], 1995, REF), key=f"year|{k}")
    km = c2.number_input("Kilometres driven", 500, 500_000, clamp(d["km"], 500, 500_000),
                         step=5000, key=f"km|{k}")
    owner = c3.selectbox("Owner", OWNERS)

    c1, c2, c3 = st.columns(3)
    fuel = c1.selectbox("Fuel", fuels, index=fuels.index(d["fuel"]) if d["fuel"] in fuels else 0, key=f"fuel|{k}")
    transmission = c2.selectbox("Transmission", trans,
                                index=trans.index(d["transmission"]) if d["transmission"] in trans else 0,
                                key=f"tr|{k}")
    seller = c3.selectbox("Seller type", sellers, index=sellers.index("Individual") if "Individual" in sellers else 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    engine = c1.number_input("Engine (CC)", 500, 6000, clamp(d["engine"], 500, 6000), step=50, key=f"cc|{k}")
    power = c2.number_input("Max power (bhp)", 20.0, 700.0, float(clamp(d["power"], 20.0, 700.0)),
                            step=1.0, key=f"bhp|{k}")
    mileage = c3.number_input("Mileage (kmpl)", 3.0, 40.0, float(clamp(d["mileage"], 3.0, 40.0)),
                              step=0.5, key=f"kmpl|{k}")
    torque = c4.number_input("Torque (Nm)", 30.0, 1500.0, float(clamp(d["torque"], 30.0, 1500.0)),
                             step=5.0, key=f"nm|{k}")
    seats = c5.number_input("Seats", 2, 14, clamp(d["seats"], 2, 14), key=f"seats|{k}")

    asking = st.number_input("Asking price in ₹ (optional, leave 0 to skip)", 0, 100_000_000, 0, step=10_000)

    if st.button("Estimate price", type="primary"):
        row = build_input(brand, model_name, year, km, fuel, transmission, seller, owner,
                          engine, power, mileage, torque, seats, ref_year=REF)
        pred = predict_price(art["model"], row)
        low, high, band = price_range(pred, art["calibration"])
        st.session_state["res"] = dict(row=row, pred=pred, low=low, high=high, band=band, asking=asking,
                                       brand=brand, model=model_name, year=year, km=km, fuel=fuel,
                                       transmission=transmission)

    res = st.session_state.get("res")
    if not res:
        return

    cov = int(art["calibration"]["coverage"] * 100)
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated listing value", lakh(res["pred"]))
    m2.metric(f"Likely range (~{cov}%)", f"{lakh(res['low'])} – {lakh(res['high'])}")
    m3.metric("Price level", res["band"])
    st.caption(f"The range is built from the model's errors on cars it had not seen, at this price level. "
               f"It is designed to contain about {cov}% of actual listing prices; it is not a guarantee.")

    if res["asking"] > 0:
        v = verdict(res["asking"], res["pred"], res["low"], res["high"])
        box = {"Above the expected range": st.warning, "Below the expected range": st.info,
               "Within the expected range": st.success}[v["status"]]
        box(f"**{v['status']}**: the asking price is {v['diff_pct']:+.1f}% compared with the estimate. {v['message']}")
        if v["negotiation_range"]:
            lo, hi = v["negotiation_range"]
            st.write(f"Suggested negotiation range: **{lakh(lo)} – {lakh(hi)}**")

    st.subheader("How reliable is this estimate?")
    for level, text in confidence_warnings(res["row"], res["pred"], art["support"], art["calibration"], res["band"]):
        (st.warning if level == "warning" else st.info)(text)

    st.subheader("Why this price?")
    eff = get_explainer(art["model"]).explain_one(res["row"][FEATURES]).head(8).copy()
    eff["label"] = eff["feature"] + " = " + eff["value"].map(fmt_value)
    eff["direction"] = np.where(eff["approx_pct_effect"] > 0, "Raises price", "Lowers price")
    fig = px.bar(eff.iloc[::-1], x="approx_pct_effect", y="label", orientation="h", color="direction",
                 color_discrete_map={"Raises price": "#2e7d32", "Lowers price": "#c62828"},
                 labels={"approx_pct_effect": "Approx. effect on price vs a typical car (%)", "label": ""})
    st.plotly_chart(fig)
    st.caption("Effects come from SHAP values. They are approximate percentages and multiply together rather than adding up.")

    st.subheader("Similar listings in the dataset")
    sim, info = find_similar(art["listings"], res["brand"], res["model"], res["year"], res["km"],
                             res["fuel"], res["transmission"])
    msg = f"Match quality: **{info['quality']}** ({info['n_close']} close matches; searched {info['scope']})."
    if "median_price" in info:
        msg += f" Median price of the listings shown: {lakh(info['median_price'])}."
    st.caption(msg)
    if not sim.empty:
        show = sim.drop(columns="distance").copy()
        show["selling_price"] = (show["selling_price"] / 1e5).round(2)
        show = show.rename(columns={"name": "Car", "year": "Year", "km_driven": "Km driven", "fuel": "Fuel",
                                    "transmission": "Transmission", "owner": "Owner",
                                    "selling_price": "Price (lakh)"})
        st.dataframe(show, hide_index=True)


# ---------- page 2: market insights ----------
def page_insights():
    st.title("📊 Market Insights")
    c1, c2, c3 = st.columns(3)
    brands = c1.multiselect("Brand (empty = all)", listings["brand"].value_counts().index.tolist())
    fuels = c2.multiselect("Fuel (empty = all)", sorted(listings["fuel"].dropna().unique()))
    y_min = int(listings["year"].min())
    y0, y1 = c3.slider("Manufacturing year", y_min, REF, (y_min, REF))

    d = listings[listings["year"].between(y0, y1)]
    if brands:
        d = d[d["brand"].isin(brands)]
    if fuels:
        d = d[d["fuel"].isin(fuels)]
    if len(d) < 30:
        st.warning("Fewer than 30 listings match these filters. Please widen them.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Listings", f"{len(d):,}")
    k2.metric("Median price", lakh(d["selling_price"].median()))
    k3.metric("Median car age", f"{d['car_age'].median():.0f} yrs")
    k4.metric("Median km driven", f"{d['km_driven'].median():,.0f}")

    left, right = st.columns(2)

    dep = d.groupby("car_age")["price_lakh"].agg(["median", "size"]).reset_index()
    dep = dep[dep["size"] >= 10]
    left.plotly_chart(px.line(dep, x="car_age", y="median", markers=True,
                              labels={"car_age": "Car age (years)", "median": "Median price (lakh)"},
                              title="Depreciation: median price by car age"))
    young, old = d.loc[d["car_age"] <= 2, "price_lakh"], d.loc[d["car_age"] >= 9, "price_lakh"]
    if len(young) >= 10 and len(old) >= 10:
        left.caption(f"Median price is {young.median():.2f} L for cars up to 2 years old, "
                     f"against {old.median():.2f} L for cars 9+ years old.")

    bt = (d.groupby("brand")["price_lakh"].agg(["median", "size"]).query("size >= 20")
           .sort_values("median").tail(15).reset_index())
    right.plotly_chart(px.bar(bt, x="median", y="brand", orientation="h",
                              labels={"median": "Median price (lakh)", "brand": ""},
                              title="Median price by brand (brands with 20+ listings)"))

    fig = px.box(d, x="fuel", y="price_lakh", labels={"price_lakh": "Price (lakh)", "fuel": "Fuel"},
                 title="Price by fuel type")
    fig.update_yaxes(range=[0, float(d["price_lakh"].quantile(0.99))])
    left.plotly_chart(fig)
    left.caption("The chart's axis stops at the 99th percentile for readability. No listings are removed.")

    own = d.groupby("owner")["price_lakh"].median().reindex(OWNERS).dropna().reset_index()
    right.plotly_chart(px.bar(own, x="owner", y="price_lakh", title="Median price by owner type",
                              labels={"price_lakh": "Median price (lakh)", "owner": ""}))

    tr = d.groupby("transmission")["price_lakh"].median().reset_index()
    left.plotly_chart(px.bar(tr, x="transmission", y="price_lakh", title="Median price by transmission",
                             labels={"price_lakh": "Median price (lakh)", "transmission": ""}))


# ---------- page 3: model report ----------
def page_report():
    st.title("🧪 Model Report")
    summary = read_json("models/model_summary.json")
    base = read_csv("reports/model_results/baseline_comparison.csv")
    tune = read_csv("reports/model_results/tuning_comparison.csv")
    oof = read_csv("reports/model_results/oof_predictions.csv")
    cal = art["calibration"]

    if summary:
        st.subheader(f"Final model: {summary['model']}")
        ts = summary["test_scores"]
        for col, (name, val) in zip(st.columns(len(ts)), ts.items()):
            col.metric(name, f"{val:.3f}")
        st.caption("These scores are from a held-out test set that was used once, after tuning was finished.")
        with st.expander("Best parameters"):
            st.json(summary["best_params"])

    if base is not None:
        st.subheader("1. Model comparison (5-fold grouped cross-validation on training data)")
        st.dataframe(base, hide_index=True)
        st.plotly_chart(px.bar(base, x="model", y="MAE (lakh)", title="Cross-validated MAE by model (lower is better)"))

    if tune is not None:
        st.subheader("2. Hyperparameter tuning: before and after")
        st.dataframe(tune, hide_index=True)

    if oof is not None:
        d = oof.copy()
        d["band"] = pd.cut(d["predicted"], cal["edges"], labels=cal["labels"], right=False)
        d["abs_pct"] = (d["predicted"] - d["actual"]).abs() / d["actual"] * 100
        d["abs_err_lakh"] = (d["predicted"] - d["actual"]).abs() / 1e5
        tbl = (d.groupby("band", observed=True)
                .agg(listings=("actual", "size"), MAE_lakh=("abs_err_lakh", "mean"),
                     median_pct_error=("abs_pct", "median"))
                .round(2).reset_index())
        st.subheader("3. Where is the model less accurate? (error by price level)")
        st.dataframe(tbl, hide_index=True)
        st.plotly_chart(px.bar(tbl, x="band", y="median_pct_error",
                               title="Median % error by predicted price level",
                               labels={"band": "Predicted price level", "median_pct_error": "Median % error"}))

    st.subheader("4. How the price range is built")
    st.write("For each price level, the range uses the 10th and 90th percentile of the model's out-of-fold errors "
             "(errors on cars the model had not seen during training).")
    rng = pd.DataFrame(cal["bands"]).T
    rng["below estimate (%)"] = ((np.exp(rng["lo"].astype(float)) - 1) * 100).round(1)
    rng["above estimate (%)"] = ((np.exp(rng["hi"].astype(float)) - 1) * 100).round(1)
    st.dataframe(rng[["n", "below estimate (%)", "above estimate (%)"]].rename(columns={"n": "listings used"}))

    st.subheader("5. Explainability and error plots")
    for title, fname in [("SHAP feature importance", "shap_importance.png"),
                         ("SHAP effect directions", "shap_beeswarm.png"),
                         ("Actual vs predicted and residuals", "error_analysis.png"),
                         ("Partial dependence", "partial_dependence.png")]:
        p = ROOT / "reports" / "figures" / fname
        if p.exists():
            st.markdown(f"**{title}**")
            st.image(str(p), width=800)


# ---------- page 4: about ----------
def page_about():
    st.title("ℹ️ About & Limitations")
    st.markdown(f"""
**What this is:** an end-to-end machine learning project that estimates the *listing-market value* of used cars
in India, gives a likely price range, explains the estimate, and checks an asking price against it.

**Data:** about {len(listings):,} used-car listings from CarDekho (Kaggle "Vehicle dataset", file *Car details v3*),
after cleaning, with manufacturing years {int(listings['year'].min())} to {REF}.

**Method:** data cleaning, EDA, feature engineering, model comparison with grouped 5-fold cross-validation,
hyperparameter tuning (RandomizedSearchCV then GridSearchCV), a single final test-set evaluation, SHAP
explanations, and a price range calibrated on out-of-fold errors.

**Limitations**
- Prices are **listing prices** from a data snapshot ending around {REF}. They are not verified sale prices
  and not today's market prices.
- The data has no city, condition, accident or service history, colour, or listing date.
- Rare brands and models, and expensive cars, have little data, so errors are larger there.
- A model is the first word after the brand, so some variants are only separated through fuel and specifications.
- The price range is statistical (about 80% coverage by design). Use the estimate as one input, and always
  inspect the car and compare real listings.

**Code:** https://github.com/Patil240907/used-car-price-predictor
""")


# ---------- navigation ----------
page = st.sidebar.radio("Go to", ["Price Estimator", "Market Insights", "Model Report", "About & Limitations"])
st.sidebar.caption("Estimates are listing-market values from CarDekho listings, not verified sale prices.")

{"Price Estimator": page_estimator, "Market Insights": page_insights,
 "Model Report": page_report, "About & Limitations": page_about}[page]()
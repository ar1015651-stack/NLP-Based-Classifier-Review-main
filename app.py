
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# 1. APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Review Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.joblib"
DATA_PATH = BASE_DIR / "data" / "reviews.csv"
REPORT_PATH = BASE_DIR / "models" / "evaluation_report.json"

SENTIMENTS = ["positive", "neutral", "negative"]

COLORS = {
    "positive": "#22c55e",
    "neutral": "#f59e0b",
    "negative": "#ef4444",
}

# =========================================================
# 2. CUSTOM DASHBOARD STYLING
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263244;
    }

    [data-testid="stMetric"] {
        background: #151f30;
        border: 1px solid #29364b;
        padding: 18px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #9ca3af;
    }

    [data-testid="stMetricValue"] {
        color: #f9fafb;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 750;
        color: #f8fafc;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 24px;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        color: #f3f4f6;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .result-card {
        background: #151f30;
        border: 1px solid #29364b;
        border-radius: 14px;
        padding: 20px;
        margin: 8px 0 16px 0;
    }

    .small-note {
        color: #9ca3af;
        font-size: 0.85rem;
    }

    div.stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 3. MODEL LOADING
# =========================================================

@st.cache_resource
def load_tfidf_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_pretrained_model():
    """
    Load the pretrained three-class multilingual model.
    Its weights download on the first use and are cached locally.
    CPU inference is used for compatibility.
    """
    from transformers import pipeline

    return pipeline(
        task="text-classification",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        device=-1,
    )


def get_tfidf_model():
    try:
        return load_tfidf_model(), None
    except Exception as exc:
        return None, str(exc)


# =========================================================
# 4. DATA LOADING AND CLEANING
# =========================================================

@st.cache_data
def load_reviews_dataset():
    if not DATA_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)

    required = {"review", "sentiment"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df = df[["review", "sentiment"]].copy()
    df["review"] = df["review"].fillna("").astype(str).str.strip()
    df["sentiment"] = (
        df["sentiment"].fillna("").astype(str).str.strip().str.lower()
    )

    df = df[
        (df["review"] != "")
        & (df["sentiment"].isin(SENTIMENTS))
    ]

    return df.reset_index(drop=True)


def load_evaluation_report():
    if not REPORT_PATH.exists():
        return {}

    try:
        import json

        with open(REPORT_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


# =========================================================
# 5. SENTIMENT PREDICTION
# =========================================================

def predict_tfidf(text):
    model, error = get_tfidf_model()

    if model is None:
        raise RuntimeError(
            error or "Trained model not found. Run python train_model.py."
        )

    prediction = str(model.predict([text])[0]).lower()

    result = {
        "model": "TF-IDF + Logistic Regression",
        "sentiment": prediction,
        "scores": {},
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        classes = model.classes_

        result["scores"] = {
            str(label).lower(): float(score)
            for label, score in zip(classes, probabilities)
        }

    return result


def predict_pretrained(text):
    classifier = load_pretrained_model()

    output = classifier(
        text[:4000],
        top_k=None,
        truncation=True,
        max_length=512,
    )

    # Transformers can return either a list of dictionaries
    # or a nested list, depending on the version.
    if output and isinstance(output[0], list):
        output = output[0]

    scores = {}

    for item in output:
        label = str(item["label"]).strip().lower()
        score = float(item["score"])

        # Handle common label naming conventions.
        if "positive" in label or label in {"label_2", "2"}:
            normalized = "positive"
        elif "negative" in label or label in {"label_0", "0"}:
            normalized = "negative"
        elif "neutral" in label or label in {"label_1", "1"}:
            normalized = "neutral"
        else:
            continue

        scores[normalized] = score

    if not scores:
        raise ValueError(
            "Could not map the pretrained model's output labels."
        )

    # If the model exposes all three classes, use its highest score.
    prediction = max(scores, key=scores.get)

    return {
        "model": "Pretrained multilingual DistilBERT",
        "sentiment": prediction,
        "scores": scores,
    }


# =========================================================
# 6. SHARED UI COMPONENTS
# =========================================================

def show_header(title, subtitle):
    st.markdown(
        f'<div class="main-title">{title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def sentiment_badge(sentiment):
    colors = {
        "positive": "#166534",
        "neutral": "#92400e",
        "negative": "#991b1b",
    }
    color = colors.get(sentiment, "#374151")

    st.markdown(
        f"""
        <div style="
            display:inline-block;
            background:{color};
            color:white;
            padding:7px 13px;
            border-radius:20px;
            font-weight:700;
            text-transform:uppercase;
            letter-spacing:0.5px;
        ">
            {sentiment}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_prediction(result):
    sentiment = result["sentiment"]

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.caption("PREDICTED SENTIMENT")
    sentiment_badge(sentiment)
    st.markdown(f"**Model:** {result['model']}")
    st.markdown("</div>", unsafe_allow_html=True)

    scores = result.get("scores", {})

    if scores:
        score_df = pd.DataFrame(
            [
                {"Sentiment": label.title(), "Score": score}
                for label, score in scores.items()
            ]
        )

        fig = px.bar(
            score_df,
            x="Sentiment",
            y="Score",
            color="Sentiment",
            color_discrete_map={
                "Positive": COLORS["positive"],
                "Neutral": COLORS["neutral"],
                "Negative": COLORS["negative"],
            },
            range_y=[0, 1],
            text=score_df["Score"].map(lambda x: f"{x:.1%}"),
        )

        fig.update_layout(
            title="Model output scores",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            yaxis_title="Score",
            xaxis_title="",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Scores are model outputs, not guaranteed probabilities "
            "that the prediction is correct."
        )


def save_to_history(text, result):
    if "review_history" not in st.session_state:
        st.session_state.review_history = []

    st.session_state.review_history.insert(
        0,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "review": text,
            "sentiment": result["sentiment"],
            "model": result["model"],
            "scores": result.get("scores", {}),
        },
    )

    # Keep the latest 500 records in this browser session.
    st.session_state.review_history = (
        st.session_state.review_history[:500]
    )


# =========================================================
# 7. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown("## 📊 Review Intelligence")
    st.caption("Customer feedback analytics")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Analyze Review",
            "Model Comparison",
            "Review History",
            "About",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("**Current project**")
    st.caption("TF-IDF + Logistic Regression")

    if MODEL_PATH.exists():
        st.success("Local model file found")
    else:
        st.error("Local model missing")
        st.caption("Run: python train_model.py")

    st.markdown(
        '<div class="small-note">Sentiment: positive · neutral · negative</div>',
        unsafe_allow_html=True,
    )


# Load the dataset and report once.
dataset = load_reviews_dataset()
evaluation = load_evaluation_report()

history = st.session_state.get("review_history", [])
history_df = pd.DataFrame(history)


# =========================================================
# 8. OVERVIEW PAGE
# =========================================================

if page == "Overview":
    show_header(
        "Review Intelligence Dashboard",
        "Understand customer sentiment and explore review patterns.",
    )

    total_dataset = len(dataset)
    total_analyzed = len(history_df)

    if not dataset.empty:
        distribution = (
            dataset["sentiment"]
            .value_counts()
            .reindex(SENTIMENTS, fill_value=0)
        )
        most_common = distribution.idxmax().title()
    else:
        distribution = pd.Series(
            [0, 0, 0],
            index=SENTIMENTS,
        )
        most_common = "N/A"

    positive_pct = (
        float(distribution["positive"] / total_dataset * 100)
        if total_dataset else 0
    )
    negative_pct = (
        float(distribution["negative"] / total_dataset * 100)
        if total_dataset else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Dataset reviews", f"{total_dataset:,}")
    c2.metric("Analyzed this session", f"{total_analyzed:,}")
    c3.metric("Positive in dataset", f"{positive_pct:.1f}%")
    c4.metric("Negative in dataset", f"{negative_pct:.1f}%")

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.markdown(
            '<div class="section-title">Dataset sentiment distribution</div>',
            unsafe_allow_html=True,
        )

        if total_dataset:
            chart_df = pd.DataFrame(
                {
                    "Sentiment": distribution.index.str.title(),
                    "Reviews": distribution.values,
                }
            )

            fig = px.pie(
                chart_df,
                names="Sentiment",
                values="Reviews",
                hole=0.62,
                color="Sentiment",
                color_discrete_map={
                    "Positive": COLORS["positive"],
                    "Neutral": COLORS["neutral"],
                    "Negative": COLORS["negative"],
                },
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=5, r=5, t=15, b=5),
                legend_title="Sentiment",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add a valid data/reviews.csv to view dataset analytics.")

    with right:
        st.markdown(
            '<div class="section-title">Dataset breakdown</div>',
            unsafe_allow_html=True,
        )

        if total_dataset:
            breakdown = pd.DataFrame(
                {
                    "Sentiment": distribution.index.str.title(),
                    "Reviews": distribution.values,
                }
            )

            fig = px.bar(
                breakdown,
                x="Sentiment",
                y="Reviews",
                color="Sentiment",
                color_discrete_map={
                    "Positive": COLORS["positive"],
                    "Neutral": COLORS["neutral"],
                    "Negative": COLORS["negative"],
                },
                text="Reviews",
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                margin=dict(l=5, r=5, t=15, b=5),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No dataset records available.")

    st.divider()
    st.markdown("### Model evaluation")

    if evaluation:
        a, b, c = st.columns(3)

        accuracy = evaluation.get("accuracy")
        macro_f1 = evaluation.get("macro_f1")

        a.metric(
            "Test accuracy",
            f"{accuracy:.1%}" if accuracy is not None else "N/A",
        )
        b.metric(
            "Macro F1",
            f"{macro_f1:.3f}" if macro_f1 is not None else "N/A",
        )
        c.metric(
            "Test examples",
            str(evaluation.get("test_examples", "N/A")),
        )

        st.warning(
            "These scores come from your saved evaluation report. "
            "Because your current dataset is synthetic, these scores "
            "should not be interpreted as real-world performance."
        )
    else:
        st.info("No evaluation report found. Run python train_model.py.")

    st.markdown("### Quick start")

    if st.button("Analyze a customer review", type="primary"):
        st.session_state["requested_page"] = "Analyze Review"
        st.rerun()


# =========================================================
# 9. ANALYZE REVIEW PAGE
# =========================================================

elif page == "Analyze Review":
    show_header(
        "Analyze a Review",
        "Enter customer feedback to classify its sentiment.",
    )

    model_choice = st.selectbox(
        "Choose a model",
        [
            "TF-IDF + Logistic Regression",
            "Pretrained multilingual DistilBERT",
        ],
    )

    review_text = st.text_area(
        "Customer review",
        placeholder=(
            "Example: The product works well, but delivery was late."
        ),
        height=160,
        max_chars=4000,
    )

    st.caption(f"{len(review_text)} / 4000 characters")

    analyze_clicked = st.button(
        "Analyze sentiment",
        type="primary",
        use_container_width=True,
    )

    if analyze_clicked:
        if not review_text.strip():
            st.warning("Please enter a review first.")
        else:
            try:
                with st.spinner("Analyzing review..."):
                    if model_choice == "TF-IDF + Logistic Regression":
                        result = predict_tfidf(review_text.strip())
                    else:
                        result = predict_pretrained(review_text.strip())

                st.session_state["last_prediction"] = {
                    "text": review_text.strip(),
                    "result": result,
                }

                save_to_history(review_text.strip(), result)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

                if model_choice == "TF-IDF + Logistic Regression":
                    st.info(
                        "Check that models/sentiment_model.joblib exists "
                        "and that training completed successfully."
                    )
                else:
                    st.info(
                        "The pretrained model needs an internet connection "
                        "the first time it downloads. It is a large model "
                        "and may take time or memory on your computer."
                    )

    last = st.session_state.get("last_prediction")

    if last:
        st.divider()
        st.markdown("### Prediction result")
        st.markdown("**Review:**")
        st.write(last["text"])
        show_prediction(last["result"])

        if last["result"]["sentiment"] == "negative":
            st.info(
                "Suggested next step: investigate the complaint and "
                "identify the customer's main issue."
            )
        elif last["result"]["sentiment"] == "neutral":
            st.info(
                "Suggested next step: review the details to identify "
                "what worked and what could be improved."
            )
        else:
            st.info(
                "Suggested next step: identify which product or service "
                "features the customer appreciated."
            )


# =========================================================
# 10. MODEL COMPARISON PAGE
# =========================================================

elif page == "Model Comparison":
    show_header(
        "Model Comparison",
        "Run both classifiers on the same text and inspect their outputs.",
    )

    st.info(
        "This page compares predictions, not independently measured "
        "accuracy. To establish which model performs better, evaluate "
        "both on the same real, labeled test dataset."
    )

    comparison_text = st.text_area(
        "Enter a review to compare",
        placeholder="The product is terrible and completely useless.",
        height=130,
        max_chars=4000,
        key="comparison_review",
    )

    if st.button(
        "Compare both models",
        type="primary",
        use_container_width=True,
    ):
        if not comparison_text.strip():
            st.warning("Enter a review to continue.")
        else:
            results = {}

            with st.spinner("Running the available models..."):
                try:
                    results["TF-IDF + Logistic Regression"] = (
                        predict_tfidf(comparison_text.strip())
                    )
                except Exception as exc:
                    st.error(f"TF-IDF model failed: {exc}")

                try:
                    results["Pretrained multilingual DistilBERT"] = (
                        predict_pretrained(comparison_text.strip())
                    )
                except Exception as exc:
                    st.error(f"Pretrained model failed: {exc}")

            st.session_state["comparison_results"] = results
            st.session_state["comparison_text"] = comparison_text.strip()

    results = st.session_state.get("comparison_results", {})

    if results:
        st.markdown("**Review:**")
        st.write(st.session_state.get("comparison_text", ""))

        cols = st.columns(len(results))

        for col, (model_name, result) in zip(cols, results.items()):
            with col:
                st.markdown(f"### {model_name}")
                sentiment_badge(result["sentiment"])
                st.markdown("")

                for label, score in sorted(
                    result["scores"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                ):
                    st.write(f"{label.title()}: {score:.1%}")

        predictions = [
            {
                "Model": model_name,
                "Predicted sentiment": result["sentiment"].title(),
            }
            for model_name, result in results.items()
        ]

        st.divider()
        st.markdown("### Prediction summary")
        st.dataframe(
            pd.DataFrame(predictions),
            use_container_width=True,
            hide_index=True,
        )

        if len(results) == 2:
            sentiments = {
                result["sentiment"] for result in results.values()
            }

            if len(sentiments) == 1:
                st.success("Both models predicted the same sentiment.")
            else:
                st.warning(
                    "The models disagree. Review the text and, if possible, "
                    "compare their predictions with a human-assigned label."
                )


# =========================================================
# 11. REVIEW HISTORY PAGE
# =========================================================

elif page == "Review History":
    show_header(
        "Review History",
        "Browse the reviews analyzed during this Streamlit session.",
    )

    history = st.session_state.get("review_history", [])

    if not history:
        st.info(
            "No reviews analyzed yet. Go to Analyze Review to get started."
        )
    else:
        history_df = pd.DataFrame(history)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total analyzed", len(history_df))
        c2.metric(
            "Positive",
            int((history_df["sentiment"] == "positive").sum()),
        )
        c3.metric(
            "Negative",
            int((history_df["sentiment"] == "negative").sum()),
        )

        filter_sentiment = st.selectbox(
            "Filter by sentiment",
            ["All", "positive", "neutral", "negative"],
        )

        filtered = history_df.copy()

        if filter_sentiment != "All":
            filtered = filtered[
                filtered["sentiment"] == filter_sentiment
            ]

        st.dataframe(
            filtered[
                ["time", "review", "sentiment", "model"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        csv_data = filtered[
            ["time", "review", "sentiment", "model"]
        ].to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download history as CSV",
            data=csv_data,
            file_name="review_history.csv",
            mime="text/csv",
        )

        if st.button("Clear session history"):
            st.session_state["review_history"] = []
            st.session_state.pop("last_prediction", None)
            st.rerun()

        st.caption(
            "History is stored in Streamlit session state, not a permanent "
            "database. It may disappear when the session resets."
        )


# =========================================================
# 12. ABOUT PAGE
# =========================================================

elif page == "About":
    show_header(
        "About Review Intelligence",
        "A machine-learning dashboard for customer feedback analysis.",
    )

    st.markdown(
        """
        ### Project objectives

        - Classify customer reviews into positive, neutral, or negative.
        - Explore the sentiment distribution of labeled reviews.
        - Compare a locally trained model with a pretrained transformer.
        - Export the current session's analyzed review history.

        ### Models

        **TF-IDF + Logistic Regression**

        Uses word and two-word phrase features to classify text. It is
        relatively lightweight and runs locally using the saved model.

        **Pretrained multilingual DistilBERT**

        Uses a pretrained transformer for three-class sentiment
        classification. The first run downloads model files and requires
        additional memory and processing time.

        ### Limitations

        - Predictions may be wrong, especially for sarcasm, mixed opinions,
          domain-specific terms, and ambiguous reviews.
        - Model scores are not guarantees of correctness.
        - Performance must be measured on independent, real labeled data.
        - Dataset charts describe the dataset, not necessarily all customers.
        - This demo does not store history permanently in a database.

        ### Project files

        `app.py` — dashboard and prediction interface

        `train_model.py` — training and evaluation

        `data/reviews.csv` — labeled review dataset

        `models/sentiment_model.joblib` — saved TF-IDF pipeline

        `models/evaluation_report.json` — saved evaluation metrics
        """
    )

    st.markdown("### Current environment")

    st.write(f"Dataset available: {'Yes' if DATA_PATH.exists() else 'No'}")
    st.write(f"TF-IDF model available: {'Yes' if MODEL_PATH.exists() else 'No'}")
    st.write(f"Evaluation report available: {'Yes' if REPORT_PATH.exists() else 'No'}")

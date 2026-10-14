"""
🎬 Review Cinema - Streamlit sentiment app (movie-theatre theme)
Run with:  streamlit run app.py
Keep `review.joblib` in the same folder as this file.
(Optional) keep the `.streamlit/config.toml` folder next to it for the dark theme.
"""

import os
import re
import string
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# Text pre-processing - the saved pipeline references `wordopt`, so it must
# exist BEFORE the model is loaded.
# --------------------------------------------------------------------------
def wordopt(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text


import __main__  # noqa: E402

__main__.wordopt = wordopt

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review.joblib")
NEUTRAL_BAND = 0.10  # 0.5 +/- 0.10 -> "Mixed"

# --------------------------------------------------------------------------
# Page setup + cinema styling
# --------------------------------------------------------------------------
st.set_page_config(page_title="Review Cinema", page_icon="🎬", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Playfair+Display:ital,wght@0,600;1,500&family=Inter:wght@400;600&display=swap');

    :root {
        --gold: #f5c518;
        --gold-soft: #ffe08a;
        --curtain: #7a0c1a;
        --ink: #0b0b0f;
        --card: rgba(20, 20, 28, .82);
        --text: #f3ead8;
    }

    /* ---------- cinema background ---------- */
    .stApp {
        background:
            radial-gradient(ellipse 60% 45% at 50% -5%, rgba(255,224,138,.28), transparent 70%),
            radial-gradient(ellipse 120% 80% at 50% 120%, rgba(122,12,26,.45), transparent 60%),
            linear-gradient(180deg, #14090d 0%, #0b0b0f 45%, #0b0b0f 100%);
        background-attachment: fixed;
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }
    /* red velvet curtains on both sides */
    .stApp::before {
        content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
        background:
            repeating-linear-gradient(90deg,
                rgba(122,12,26,.95) 0px, rgba(60,4,12,.95) 14px,
                rgba(150,20,36,.95) 28px, rgba(60,4,12,.95) 42px) left / 11% 100% no-repeat,
            repeating-linear-gradient(90deg,
                rgba(122,12,26,.95) 0px, rgba(60,4,12,.95) 14px,
                rgba(150,20,36,.95) 28px, rgba(60,4,12,.95) 42px) right / 11% 100% no-repeat;
        -webkit-mask-image: linear-gradient(90deg, #000 0, #000 9%, transparent 11%, transparent 89%, #000 91%, #000 100%);
                mask-image: linear-gradient(90deg, #000 0, #000 9%, transparent 11%, transparent 89%, #000 91%, #000 100%);
        opacity: .75;
    }
    /* film grain + vignette */
    .stApp::after {
        content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
        background:
            radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,.65) 100%),
            url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%' height='100%' filter='url(%23n)' opacity='.55'/></svg>");
        mix-blend-mode: overlay; opacity: .35;
    }
    [data-testid="stHeader"] {background: transparent;}
    .block-container {position: relative; z-index: 1; padding-top: 2rem; max-width: 800px;}
    footer, #MainMenu {visibility: hidden;}

    /* ---------- marquee hero ---------- */
    .marquee {
        position: relative; text-align: center; padding: 2.2rem 1rem 1.8rem;
        margin-bottom: 1.2rem; border-radius: 18px;
        background: linear-gradient(180deg, #1c1018, #0e0a10);
        border: 3px solid var(--gold);
        box-shadow: 0 0 40px rgba(245,197,24,.35), inset 0 0 40px rgba(0,0,0,.8);
    }
    .marquee::before, .marquee::after {   /* blinking bulbs top & bottom */
        content: ""; position: absolute; left: 12px; right: 12px; height: 8px;
        background: radial-gradient(circle, #fff3b0 0 3px, var(--gold) 3.5px 4px, transparent 5px) 0 0 / 22px 8px repeat-x;
        animation: bulbs 1.2s steps(2) infinite;
        filter: drop-shadow(0 0 4px var(--gold));
    }
    .marquee::before {top: 8px;}
    .marquee::after {bottom: 8px; animation-delay: .6s;}
    @keyframes bulbs {50% {opacity: .35;}}
    .marquee .now {
        font-family: 'Bebas Neue', Impact, sans-serif; letter-spacing: .5em;
        color: var(--gold); font-size: .95rem;
    }
    .marquee h1 {
        font-family: 'Bebas Neue', Impact, sans-serif; font-size: 3.4rem;
        letter-spacing: .08em; margin: .1rem 0; line-height: 1.05;
        color: #fff3d0; text-shadow: 0 0 18px rgba(245,197,24,.7), 0 3px 0 #7a5a00;
    }
    .marquee p {
        font-family: 'Playfair Display', Georgia, serif; font-style: italic;
        margin: .2rem 0 0; color: #d9c9a3; font-size: 1.05rem;
    }

    /* film strip divider */
    .filmstrip {
        height: 22px; margin: 1.2rem 0; border-radius: 4px;
        background:
            repeating-linear-gradient(90deg, transparent 0 8px, #0b0b0f 8px 18px, transparent 18px 26px) 0 3px / 26px 6px repeat-x,
            repeating-linear-gradient(90deg, transparent 0 8px, #0b0b0f 8px 18px, transparent 18px 26px) 0 13px / 26px 6px repeat-x,
            #2a2a33;
        opacity: .9;
    }

    /* ---------- inputs ---------- */
    .stTextArea textarea {
        background: rgba(10,10,14,.85) !important; color: var(--text) !important;
        border: 1.5px solid rgba(245,197,24,.45) !important; border-radius: 14px !important;
        font-family: 'Playfair Display', Georgia, serif; font-size: 1.05rem;
    }
    .stTextArea textarea:focus {border-color: var(--gold) !important; box-shadow: 0 0 14px rgba(245,197,24,.4) !important;}
    .stTextArea label, .stSelectbox label, [data-testid="stFileUploader"] label {color: var(--gold-soft) !important; font-weight: 600;}

    div.stButton > button, div.stDownloadButton > button {
        border-radius: 12px; font-weight: 700; letter-spacing: .03em;
        background: rgba(255,255,255,.04); color: var(--gold-soft);
        border: 1.5px solid rgba(245,197,24,.5); transition: all .2s ease;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        border-color: var(--gold); color: #fff; box-shadow: 0 0 16px rgba(245,197,24,.45); transform: translateY(-1px);
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f5c518, #e08e0b); color: #1a1200; border: none;
        box-shadow: 0 4px 18px rgba(245,197,24,.4);
    }
    div.stButton > button[kind="primary"]:hover {color: #000; box-shadow: 0 6px 26px rgba(245,197,24,.7);}

    .stTabs [data-baseweb="tab-list"] {gap: 6px; border-bottom: 1px solid rgba(245,197,24,.25);}
    .stTabs [data-baseweb="tab"] {color: #bfae86; font-weight: 600; border-radius: 10px 10px 0 0;}
    .stTabs [aria-selected="true"] {color: var(--gold) !important;}
    .stTabs [data-baseweb="tab-highlight"] {background: var(--gold) !important;}

    [data-testid="stMetric"] {
        background: var(--card); border: 1px solid rgba(245,197,24,.25);
        padding: .8rem 1rem; border-radius: 14px;
    }
    [data-testid="stMetricLabel"] {color: #bfae86;}
    [data-testid="stMetricValue"] {color: var(--gold-soft); font-family: 'Bebas Neue', Impact, sans-serif; letter-spacing: .05em;}
    h5, h4, h3 {font-family: 'Bebas Neue', Impact, sans-serif !important; letter-spacing: .08em; color: var(--gold-soft) !important;}
    [data-testid="stFileUploaderDropzone"] {background: var(--card); border: 1.5px dashed rgba(245,197,24,.5);}

    /* ---------- result "ticket" ---------- */
    .ticket {
        position: relative; text-align: center; padding: 1.6rem 1.2rem; margin: 1.2rem 0;
        border-radius: 16px; background: var(--card); backdrop-filter: blur(4px);
        border: 2px dashed var(--accent); box-shadow: 0 0 34px var(--glow);
        animation: rise .5s ease both;
    }
    .ticket::before, .ticket::after {  /* ticket notches */
        content: ""; position: absolute; top: 50%; width: 26px; height: 26px; margin-top: -13px;
        background: #0b0b0f; border-radius: 50%; border: 2px dashed var(--accent);
    }
    .ticket::before {left: -15px;} .ticket::after {right: -15px;}
    @keyframes rise {from {opacity: 0; transform: translateY(14px) scale(.98);} to {opacity: 1; transform: none;}}
    .ticket .tag {font-family: 'Bebas Neue', Impact; letter-spacing: .35em; color: var(--accent); font-size: .9rem;}
    .ticket .emoji {font-size: 3.2rem; line-height: 1.1;}
    .ticket .label {font-family: 'Bebas Neue', Impact; font-size: 2.6rem; letter-spacing: .1em; color: var(--accent); text-shadow: 0 0 16px var(--glow);}
    .ticket .stars {font-size: 1.6rem; letter-spacing: .2em; color: var(--gold);}
    .ticket .sub {color: #cdbf9d; font-size: .95rem; margin-top: .2rem;}
    .t-pos {--accent: #4ade80; --glow: rgba(74,222,128,.28);}
    .t-neg {--accent: #f87171; --glow: rgba(248,113,113,.28);}
    .t-neu {--accent: #facc15; --glow: rgba(250,204,21,.28);}

    .bar-wrap {background: #24242c; border-radius: 999px; height: 14px; overflow: hidden; border: 1px solid rgba(245,197,24,.25);}
    .bar-pos {background: linear-gradient(90deg, #f5c518, #4ade80); height: 100%; border-radius: 999px;}
    .chip {display: inline-block; padding: .28rem .8rem; margin: .18rem; border-radius: 999px; font-size: .85rem; font-weight: 600;}
    .chip-pos {background: rgba(74,222,128,.15); color: #86efac; border: 1px solid rgba(74,222,128,.5);}
    .chip-neg {background: rgba(248,113,113,.15); color: #fca5a5; border: 1px solid rgba(248,113,113,.5);}
    .hint {font-size: .8rem; color: #a99a78; margin-top: .3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Model helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading projector…")
def load_model():
    model = joblib.load(MODEL_PATH)
    # Compatibility shim for scikit-learn version differences
    clf = model.named_steps["classifier"]
    if not hasattr(clf, "multi_class"):
        clf.multi_class = "auto"
    if not hasattr(clf, "penalty"):
        clf.penalty = "l2"
    return model


def classify(model, text):
    """Return (label, tag, emoji, css_class, p_negative, p_positive)."""
    proba = model.predict_proba([text])[0]
    p_pos = float(proba[list(model.classes_).index(1)])
    p_neg = 1 - p_pos
    if abs(p_pos - 0.5) <= NEUTRAL_BAND:
        return "Mixed", "ONE FOR THE CRITICS", "🎭", "t-neu", p_neg, p_pos
    if p_pos > 0.5:
        return "Positive", "BLOCKBUSTER", "🍿", "t-pos", p_neg, p_pos
    return "Negative", "BOX-OFFICE FLOP", "🍅", "t-neg", p_neg, p_pos


def key_words(model, text, top_n=6):
    vec = model.named_steps["vectorizer"]
    clf = model.named_steps["classifier"]
    X = vec.transform([text])
    names = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    contrib = [(names[i], float(X[0, i] * coefs[i])) for i in X.nonzero()[1]]
    pos = sorted([c for c in contrib if c[1] > 0], key=lambda x: -x[1])[:top_n]
    neg = sorted([c for c in contrib if c[1] < 0], key=lambda x: x[1])[:top_n]
    return pos, neg


def predict_batch(model, texts):
    proba = model.predict_proba(list(texts))
    p_pos = proba[:, list(model.classes_).index(1)]
    labels = np.where(
        np.abs(p_pos - 0.5) <= NEUTRAL_BAND,
        "Mixed",
        np.where(p_pos > 0.5, "Positive", "Negative"),
    )
    return labels, p_pos


def stars_html(p_pos):
    n = max(1, min(5, round(p_pos * 5)))
    return "★" * n + "☆" * (5 - n)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "review_text" not in st.session_state:
    st.session_state.review_text = ""
if "history" not in st.session_state:
    st.session_state.history = []

EXAMPLES = {
    "🍿 Loved it": "Absolutely fantastic product! The quality is superb and it arrived early. Highly recommend.",
    "🍅 Hated it": "Terrible experience. It broke after two days and customer support was useless. Total waste of money.",
    "🎭 Mixed": "The design is nice but the battery life is poor. It's okay for the price, I guess.",
}


def set_example(text):
    st.session_state.review_text = text


def clear_text():
    st.session_state.review_text = ""


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="marquee">
        <div class="now">★ NOW SHOWING ★</div>
        <h1>REVIEW CINEMA</h1>
        <p>Roll the review. We'll tell you how it really played.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load `review.joblib`: {e}")
    st.info("Make sure `review.joblib` is in the same folder as `app.py`.")
    st.stop()

tab_single, tab_batch, tab_about = st.tabs(["🎬 Single review", "🎞️ Batch (CSV)", "ℹ️ About"])

# --------------------------------------------------------------------------
# Tab 1 - single review
# --------------------------------------------------------------------------
with tab_single:
    st.markdown("**Quick trailers (examples):**")
    cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        col.button(name, on_click=set_example, args=(text,), use_container_width=True)

    st.text_area(
        "Your review",
        key="review_text",
        height=150,
        placeholder="Type or paste a review here…",
    )

    b1, b2 = st.columns([3, 1])
    analyze = b1.button("🎥 Roll the verdict", type="primary", use_container_width=True)
    b2.button("🗑️ Clear", on_click=clear_text, use_container_width=True)

    if analyze:
        text = st.session_state.review_text.strip()
        if not text:
            st.warning("Please enter a review first.")
        else:
            with st.spinner("The critics are watching…"):
                label, tag, emoji, css, p_neg, p_pos = classify(model, text)
                pos_words, neg_words = key_words(model, text)

            st.markdown(
                f"""
                <div class="ticket {css}">
                    <div class="tag">{tag}</div>
                    <div class="emoji">{emoji}</div>
                    <div class="label">{label}</div>
                    <div class="stars">{stars_html(p_pos)}</div>
                    <div class="sub">Confidence: {max(p_pos, p_neg):.0%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2 = st.columns(2)
            m1.metric("👍 Positive", f"{p_pos:.1%}")
            m2.metric("👎 Negative", f"{p_neg:.1%}")
            st.markdown(
                f"""
                <div class="bar-wrap"><div class="bar-pos" style="width:{p_pos*100:.1f}%"></div></div>
                <div class="hint">Bar shows how positive the review looks</div>
                """,
                unsafe_allow_html=True,
            )

            if pos_words or neg_words:
                st.markdown("##### 🔑 Scene-stealing words")
                chips = "".join(f'<span class="chip chip-pos">+ {w}</span>' for w, _ in pos_words)
                chips += "".join(f'<span class="chip chip-neg">− {w}</span>' for w, _ in neg_words)
                st.markdown(chips, unsafe_allow_html=True)

            st.session_state.history.insert(
                0,
                {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Review": text if len(text) <= 80 else text[:77] + "…",
                    "Verdict": f"{emoji} {label}",
                    "Positive %": round(p_pos * 100, 1),
                },
            )
            st.session_state.history = st.session_state.history[:20]

    if st.session_state.history:
        st.markdown('<div class="filmstrip"></div>', unsafe_allow_html=True)
        st.markdown("##### 🕘 Recent screenings")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# --------------------------------------------------------------------------
# Tab 2 - batch CSV
# --------------------------------------------------------------------------
with tab_batch:
    st.markdown("Upload a CSV with a column of reviews and get a verdict for every row.")
    file = st.file_uploader("Choose a CSV file", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")
            df = None

        if df is not None and not df.empty:
            text_cols = [c for c in df.columns if df[c].dtype == object] or list(df.columns)
            col = st.selectbox("Which column has the reviews?", text_cols)
            if st.button("🎞️ Run batch analysis", type="primary"):
                with st.spinner(f"Screening {len(df):,} reviews…"):
                    texts = df[col].fillna("").astype(str)
                    labels, p_pos = predict_batch(model, texts)
                    out = df.copy()
                    out["sentiment"] = labels
                    out["positive_probability"] = np.round(p_pos, 3)

                counts = out["sentiment"].value_counts()
                c1, c2, c3 = st.columns(3)
                c1.metric("🍿 Positive", int(counts.get("Positive", 0)))
                c2.metric("🎭 Mixed", int(counts.get("Mixed", 0)))
                c3.metric("🍅 Negative", int(counts.get("Negative", 0)))

                st.bar_chart(counts)
                st.dataframe(out, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download results (CSV)",
                    out.to_csv(index=False).encode("utf-8"),
                    file_name="review_sentiment_results.csv",
                    mime="text/csv",
                )
        elif df is not None:
            st.warning("That CSV is empty.")

# --------------------------------------------------------------------------
# Tab 3 - about
# --------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
        **How the projector works**

        - Your text is cleaned (lower-cased; links, numbers and punctuation removed).
        - A **TF-IDF** vectorizer turns the words into numbers.
        - A **Logistic Regression** classifier decides *positive* or *negative*.

        Results close to 50/50 are shown as **Mixed**. The "scene-stealing words"
        are the words in your review that pull hardest toward each side.

        *It's a simple model, so sarcasm and very short reviews can fool it.*
        """
    )

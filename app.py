import os
from datetime import datetime

import pandas as pd
import streamlit as st
from groq import Groq

# ---------------------------------------------------------------------------
# Business setup
# ---------------------------------------------------------------------------
BRAND = "Northline Apparel"
QUESTION = "Is my new season release well received by customers?"

LABELS = {
    "Loves new season": "Positive about the new season pieces (design, fit, fabric, colors, value).",
    "Dislikes new season": "Negative about the new season pieces (design, fit, fabric, colors, value).",
    "Mixed on new season": "Likes some things about the new season pieces and dislikes others.",
    "Not about new season": "About something else: older collections, shipping, the website, or customer service.",
    "Needs review": "Unclear, sarcastic, off-topic, or serious (damaged item, refund dispute, safety, abuse).",
}
NEW_SEASON_LABELS = ["Loves new season", "Dislikes new season", "Mixed on new season"]
RATING_ONLY = "Rating only"

DATA_FILE = "feedback.csv"
COLUMNS = ["timestamp", "rating", "comment", "label"]
MODEL = st.secrets.get("GROQ_MODEL", "openai/gpt-oss-20b")


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=COLUMNS)


def save_row(rating, comment, label):
    df = load_data()
    row = pd.DataFrame(
        [{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "rating": rating,
            "comment": comment,
            "label": label,
        }]
    )
    pd.concat([df, row], ignore_index=True).to_csv(DATA_FILE, index=False)


# ---------------------------------------------------------------------------
# LLM labeling
# ---------------------------------------------------------------------------
def label_comment(comment: str) -> str:
    label_list = "\n".join(f"- {name}: {desc}" for name, desc in LABELS.items())
    prompt = f"""You label customer feedback for {BRAND}, a clothing brand that just released its new season collection.
The owner wants to know: "{QUESTION}"

Choose exactly ONE label for the comment below:
{label_list}

Reply with only the label name, exactly as written above, and nothing else.

Comment: \"\"\"{comment}\"\"\""""

    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=600,
            reasoning_effort="low",
        )
        answer = (response.choices[0].message.content or "").strip().strip(".\"'")
    except Exception as e:
        st.warning(f"Labeling failed, saved for manual review. ({e})")
        return "Needs review"

    # Anything that isn't an exact label goes to manual review.
    for name in LABELS:
        if answer.lower() == name.lower():
            return name
    return "Needs review"


# ---------------------------------------------------------------------------
# Customer page
# ---------------------------------------------------------------------------
def customer_page():
    st.title(f"🧥 {BRAND}")
    st.subheader("Tell us what you think of our new season collection")

    st.write("How would you rate it?")
    stars = st.feedback("stars")
    comment = st.text_area("Your comment (optional)", placeholder="What did you love, or what could be better?")

    if st.button("Submit feedback", type="primary"):
        comment = comment.strip()
        rating = stars + 1 if stars is not None else None

        if rating is None and not comment:
            st.error("Please leave a star rating, a comment, or both.")
            return

        if comment:
            with st.spinner("Saving your feedback..."):
                label = label_comment(comment)
        else:
            label = RATING_ONLY

        save_row(rating, comment, label)
        st.success("Thank you! Your feedback was saved.")


# ---------------------------------------------------------------------------
# Owner page
# ---------------------------------------------------------------------------
def owner_page():
    st.title("📊 Owner dashboard")
    st.markdown(f"**Question:** {QUESTION}")

    df = load_data()
    if df.empty:
        st.info("No feedback yet. Submit some on the customer page.")
        return

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    labeled = df[df["label"] != RATING_ONLY]
    new_season = labeled[labeled["label"].isin(NEW_SEASON_LABELS)]
    review = df[df["label"] == "Needs review"]

    # Key numbers
    if len(new_season):
        positive_pct = (new_season["label"] == "Loves new season").mean() * 100
        positive_text = f"{positive_pct:.0f}%"
    else:
        positive_text = "–"
    avg = df["rating"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Positive about new season", positive_text,
              help="Share of comments about the new season that are 'Loves new season'.")
    c2.metric("Average star rating", f"{avg:.1f} ★" if pd.notna(avg) else "–")
    c3.metric("Need manual review", len(review))

    # Graphs
    st.subheader("How customers feel about the new season")
    counts = labeled["label"].value_counts().reindex(LABELS.keys(), fill_value=0)
    st.bar_chart(counts, horizontal=True, x_label="Comments", y_label="")

    st.subheader("Average star rating by label")
    rated = labeled.dropna(subset=["rating"])
    if len(rated):
        avg_by_label = rated.groupby("label")["rating"].mean().reindex(LABELS.keys()).dropna()
        st.bar_chart(avg_by_label, x_label="", y_label="Average stars")
    else:
        st.caption("No comments with star ratings yet.")

    # Manual review list
    st.subheader(f"Comments that need manual review ({len(review)})")
    if review.empty:
        st.caption("Nothing to review right now.")
    for _, row in review.iloc[::-1].iterrows():
        stars = f"{int(row['rating'])} ★" if pd.notna(row["rating"]) else "No rating"
        with st.container(border=True):
            st.caption(f"{row['timestamp']} · {stars}")
            st.write(row["comment"])

    with st.expander("Label definitions"):
        for name, desc in LABELS.items():
            st.markdown(f"**{name}**: {desc}")


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
st.set_page_config(page_title=f"{BRAND} Feedback", page_icon="🧥")
page = st.sidebar.radio("Page", ["Leave feedback", "Owner dashboard"])
if page == "Leave feedback":
    customer_page()
else:
    owner_page()


import html
import joblib
import streamlit as st
 
st.set_page_config(page_title="Guest Reviews", page_icon="⭐", layout="centered")
 
 
# ------------------------------------------------------------------
# Load trained model and vectorizer (your original paths)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("C:/Users/ADEEL/Desktop/Restaurent_reviews_sentiment/Models/svm_model.pkl")
    vectorizer = joblib.load("C:/Users/ADEEL/Desktop/Restaurent_reviews_sentiment/Models/tfidf_vectorizer.pkl")
    return model, vectorizer
 
 
model, vectorizer = load_artifacts()
 
 
def run_prediction(text: str):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    liked = prediction == 1
 
    # Confidence: use probabilities if the model has them, otherwise the SVM margin.
    confidence = None
    strong = False
    if hasattr(model, "predict_proba"):
        try:
            confidence = float(model.predict_proba(X).max())
            strong = confidence >= 0.85
        except Exception:
            pass
    if confidence is None and hasattr(model, "decision_function"):
        score = abs(float(model.decision_function(X).ravel()[0]))
        strong = score >= 1.0
        confidence = min(0.5 + score / 4, 0.99)  # rough 0.5-0.99 scale for display
 
    # Cosmetic star rating derived from the binary result
    if liked:
        stars = 5 if (strong or confidence is None) else 4
    else:
        stars = 1 if (strong or confidence is None) else 2
    return liked, stars, confidence
 
 
# ------------------------------------------------------------------
# Styling (internal CSS)
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=Merriweather:wght@700&display=swap');
 
:root {
  --ink: #1f2933; --muted: #6b7785; --line: #e4e7eb; --bg: #f6f7f8;
  --accent: #0b6e5f; --star: #f5a623;
  --pos-bg: #e3f5ee; --pos-fg: #0b6e5f; --neg-bg: #fdeaea; --neg-fg: #b42318;
}
html, body, [class*="css"], .stApp { font-family: 'Source Sans 3', sans-serif; color: var(--ink); }
.stApp { background: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 780px; padding-top: 1.5rem; }
 
.banner { background: linear-gradient(135deg, #0b6e5f 0%, #14463f 100%);
  border-radius: 14px 14px 0 0; padding: 38px 28px 26px; color: #fff; }
.banner h1 { font-family: 'Merriweather', serif; font-size: 1.9rem; margin: 0 0 6px; color: #fff; }
.banner .sub { opacity: .85; font-size: .95rem; }
 
.summary { background: #fff; border: 1px solid var(--line); border-top: none;
  border-radius: 0 0 14px 14px; padding: 22px 28px; display: flex; gap: 36px; flex-wrap: wrap; }
.score { text-align: center; min-width: 120px; }
.score .num { font-size: 3rem; font-weight: 700; line-height: 1; }
.stars { color: var(--star); letter-spacing: 2px; font-size: 1.1rem; }
.stars .off { color: #d5dae0; }
.score .count { color: var(--muted); font-size: .85rem; margin-top: 4px; }
.bars { flex: 1; min-width: 220px; }
.bar-row { display: flex; align-items: center; gap: 10px; font-size: .85rem; color: var(--muted); margin: 4px 0; }
.bar-row .track { flex: 1; height: 8px; background: #edf0f2; border-radius: 4px; overflow: hidden; }
.bar-row .fill { height: 100%; background: var(--accent); border-radius: 4px; }
 
.section-title { font-family: 'Merriweather', serif; font-size: 1.2rem; margin: 28px 0 10px; }
 
div[data-testid="stTextArea"] textarea { background: #fff; border: 1px solid var(--line);
  border-radius: 10px; font-size: 1rem; padding: 12px; color: var(--ink); }
div[data-testid="stTextArea"] textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(11,110,95,.15); }
.stButton > button { background: var(--accent); color: #fff; border: none; border-radius: 999px;
  padding: .55rem 1.6rem; font-weight: 600; }
.stButton > button:hover { background: #09594d; color: #fff; }
 
.review { background: #fff; border: 1px solid var(--line); border-radius: 12px;
  padding: 18px 20px; margin-bottom: 14px; }
.review .top { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
.avatar { width: 42px; height: 42px; border-radius: 50%; background: #d7ece7; color: var(--accent);
  display: flex; align-items: center; justify-content: center; font-weight: 700; }
.review .name { font-weight: 600; }
.review .meta { color: var(--muted); font-size: .8rem; }
.review .body { line-height: 1.55; margin-top: 6px; }
.review .foot { display: flex; align-items: center; gap: 14px; margin-top: 12px; }
.badge { padding: 3px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600; }
.badge.positive { background: var(--pos-bg); color: var(--pos-fg); }
.badge.negative { background: var(--neg-bg); color: var(--neg-fg); }
.review.new { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(11,110,95,.08); }
</style>
""",
    unsafe_allow_html=True,
)
 
# ------------------------------------------------------------------
# State: reviews posted in this session (starts empty)
# ------------------------------------------------------------------
if "reviews" not in st.session_state:
    st.session_state.reviews = []
reviews = st.session_state.reviews
 
 
def stars_html(n: int) -> str:
    return '<span class="stars">' + "★" * n + '<span class="off">' + "★" * (5 - n) + "</span></span>"
 
 
# ------------------------------------------------------------------
# Header + summary
# ------------------------------------------------------------------
total = len(reviews)
avg = sum(r["stars"] for r in reviews) / total if total else 0
dist = {s: sum(r["stars"] == s for r in reviews) for s in range(5, 0, -1)}
bars = "".join(
    f'<div class="bar-row"><span>{s} star</span><div class="track"><div class="fill" '
    f'style="width:{(dist[s] / total * 100) if total else 0:.0f}%"></div></div><span>{dist[s]}</span></div>'
    for s in range(5, 0, -1)
)
avg_text = f"{avg:.1f}" if total else "–"
 
st.markdown(
    f"""
<div class="banner">
  <h1>Customers reviews</h1>
  <div class="sub">Share your experience and see how it reads.</div>
</div>
<div class="summary">
  <div class="score">
    <div class="num">{avg_text}</div>
    {stars_html(round(avg))}
    <div class="count">{total} review{'s' if total != 1 else ''}</div>
  </div>
  <div class="bars">{bars}</div>
</div>
""",
    unsafe_allow_html=True,
)
 
# ------------------------------------------------------------------
# Write a review -> real-time prediction
# ------------------------------------------------------------------
st.markdown('<div class="section-title">Write a review</div>', unsafe_allow_html=True)
review = st.text_area("Enter your review:", placeholder="Tell other guests about your visit...",
                      label_visibility="collapsed", height=120)
 
if st.button("Post review"):
    if review.strip():
        liked, stars, conf = run_prediction(review)
        reviews.insert(0, {"name": "You", "date": "Just now", "text": review.strip(),
                           "stars": stars, "liked": liked, "conf": conf, "new": True})
        for r in reviews[1:]:
            r["new"] = False
        st.rerun()
    else:
        st.warning("Please enter a review.")
 
# ------------------------------------------------------------------
# Review list
# ------------------------------------------------------------------
st.markdown('<div class="section-title">Recent reviews</div>', unsafe_allow_html=True)
 
if not reviews:
    st.markdown('<div class="review"><div class="meta">No reviews yet. Be the first to write one above.</div></div>',
                unsafe_allow_html=True)
 
for r in reviews:
    cls = "positive" if r["liked"] else "negative"
    label = "👍 Liked" if r["liked"] else "👎 Not Liked"
    conf_txt = f" · {r['conf']:.0%} confidence" if r["conf"] is not None else ""
    st.markdown(
        f"""
<div class="review {'new' if r['new'] else ''}">
  <div class="top">
    <div class="avatar">Y</div>
    <div>
      <div class="name">{html.escape(r['name'])}</div>
      <div class="meta">{html.escape(r['date'])}</div>
    </div>
  </div>
  {stars_html(r['stars'])}
  <div class="body">{html.escape(r['text'])}</div>
  <div class="foot"><span class="badge {cls}">{label}{conf_txt}</span></div>
</div>
""",
        unsafe_allow_html=True,
    )
 

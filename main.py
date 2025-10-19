import os
import io
import sqlite3
import math
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# -------------------------
# Safe rerun for Streamlit
# -------------------------
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# -------------------------
# Config
# -------------------------
DB_PATH = "AI_LAB_PROJECT.db"
CSV_PATH = "udemy_courses.csv"
TFIDF_MAX_FEATURES = 5000
TOP_N_DEFAULT = 7
HYBRID_SIM_WEIGHT = 0.6   # weight for content similarity in hybrid score
HYBRID_COLLAB_WEIGHT = 0.3
HYBRID_POP_WEIGHT = 0.1

st.set_page_config(page_title="AI_LAB_PROJECT — Course Recommender", layout="wide")
st.title("🎓 AI_LAB_PROJECT — Course Recommender (Professional)")

# ==========================
# UI / THEME - CSS
# ==========================
st.markdown(
    """
    <style>
    :root{
      --primary:#2563eb; --accent:#f59e0b; --muted:#6b7280; --bg:#f8fafc; --card:#ffffff; --radius:14px;
    }
    .stApp { background: var(--bg); }
    .hero { background: linear-gradient(90deg, rgba(37,99,235,0.95), rgba(79,70,229,0.95)); color: white; padding: 20px; border-radius: 12px; margin-bottom: 16px; box-shadow: 0 6px 18px rgba(15,23,42,0.08); }
    .hero h2 { margin: 0; font-size: 22px; } .hero p { margin: 6px 0 0 0; color: rgba(255,255,255,0.92); }
    div.stButton > button:first-child { background: var(--primary); color: white; border-radius: 10px; border: none; padding: 8px 14px; font-weight: 600; }
    div.stButton > button:first-child:hover { filter: brightness(1.06); }
    .cards-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; margin-top: 8px; margin-bottom: 12px; }
    .course-card { background: var(--card); border-radius: var(--radius); padding: 14px; box-shadow: 0 6px 18px rgba(15,23,42,0.06); transition: transform 0.16s ease, box-shadow 0.16s ease; }
    .course-card:hover { transform: translateY(-6px); box-shadow: 0 10px 26px rgba(15,23,42,0.10); }
    .course-title { font-size: 16px; font-weight:700; margin-bottom:6px; color:#0f172a; }
    .course-meta { color: var(--muted); font-size:13px; margin-bottom:10px; }
    .badge { display:inline-block; padding:6px 8px; border-radius: 999px; font-weight:600; font-size:12px; margin-right:6px; }
    .lvl-beginner { background: #d1fae5; color:#065f46; }
    .lvl-intermediate { background: #fef3c7; color:#92400e; }
    .lvl-expert { background: #fee2e2; color:#7f1d1d; }
    .score-row { margin-top:8px; font-size:13px; color:#374151; }
    .small { font-size:12px; color:var(--muted); }
    section[data-testid="stSidebar"] { background: white; padding: 18px; border-radius: 8px; box-shadow: 0 8px 26px rgba(15,23,42,0.04); }
    .chart-card { background: var(--card); padding: 10px; border-radius: 12px; box-shadow: 0 6px 18px rgba(15,23,42,0.05); }
    @media (max-width:600px) { .hero h2 { font-size:18px; } }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
      <h2>AI_LAB_PROJECT — Smart Course Recommendations</h2>
      <p>Find courses matched to your interests and skill level — hybrid recommendations combining content, collaborative signals and popularity.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# DATABASE setup
# -------------------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    interest TEXT,
    skill_level TEXT,
    learning_hours INTEGER
)""")
c.execute("""CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    course_title TEXT,
    subject TEXT,
    level TEXT,
    price TEXT,
    num_subscribers INTEGER,
    UNIQUE(user_id, course_title),
    FOREIGN KEY(user_id) REFERENCES users(id)
)""")
def add_column_if_not_exists(table, column_def):
    col_name = column_def.split()[0]
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cur.fetchall()]
    if col_name not in cols:
        try:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")
            conn.commit()
        except sqlite3.OperationalError:
            pass
add_column_if_not_exists("recommendations", "similarity_score REAL DEFAULT 0")
add_column_if_not_exists("recommendations", "popularity_score REAL DEFAULT 0")
c.execute("""CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    course_title TEXT,
    rating INTEGER,
    comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, course_title),
    FOREIGN KEY(user_id) REFERENCES users(id)
)""")
c.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id)")
c.execute("CREATE INDEX IF NOT EXISTS idx_recs_user ON recommendations(user_id)")
conn.commit()

# -------------------------
# Load dataset
# -------------------------
@st.cache_data
def load_data(path):
    if not os.path.exists(path): return None
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [col for col in ["course_title","subject","level","price","num_subscribers","content_duration","description"] if col in header]
    df = pd.read_csv(path, usecols=usecols, low_memory=False).dropna(subset=["course_title","subject","level"])
    df["course_title"] = df["course_title"].astype(str)
    df["subject"] = df["subject"].astype(str)
    df["level"] = df["level"].astype(str)
    df["price"] = df["price"].astype(str) if "price" in df.columns else ""
    df["num_subscribers"] = pd.to_numeric(df.get("num_subscribers",0), errors="coerce").fillna(0).astype(int)
    df["content_duration"] = pd.to_numeric(df.get("content_duration",np.nan), errors="coerce") if "content_duration" in df.columns else np.nan
    df["description"] = df["description"].astype(str) if "description" in df.columns else ""
    return df

df = load_data(CSV_PATH)
if df is None:
    st.error(f"Dataset file `{CSV_PATH}` not found.")
    st.stop()

# -------------------------
# TF-IDF
# -------------------------
@st.cache_resource
def prepare_tfidf(dataframe):
    df_local = dataframe.copy()
    df_local["combined"] = (df_local["course_title"] + " " + df_local["subject"] + " " + df_local["level"] + " " + df_local["description"]).astype(str)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=TFIDF_MAX_FEATURES)
    matrix = vectorizer.fit_transform(df_local["combined"])
    return vectorizer, matrix
vectorizer, matrix = prepare_tfidf(df)

# -------------------------
# Collaborative filtering
# -------------------------
def build_user_course_matrix(min_ratings=1):
    recs = pd.read_sql_query("SELECT user_id, course_title FROM recommendations", conn)
    fb = pd.read_sql_query("SELECT user_id, course_title, rating FROM feedback", conn)
    if fb.shape[0] >= min_ratings:
        interactions = fb.copy()
    else:
        if recs.empty: return None, None, None
        recs["rating"] = 1
        interactions = recs
    course_to_idx = {t:i for i,t in enumerate(df["course_title"])}
    interactions = interactions[interactions["course_title"].isin(course_to_idx)]
    if interactions.empty: return None, None, None
    interactions["course_idx"] = interactions["course_title"].map(course_to_idx)
    mat = interactions.pivot_table(index="user_id", columns="course_idx", values="rating", fill_value=0)
    return mat, interactions, course_to_idx

def get_collaborative_scores_for_user(user_id):
    mat, interactions, _ = build_user_course_matrix()
    if mat is None or user_id not in mat.index: return None
    user_vec = mat.loc[user_id].values.reshape(1,-1)
    sims = cosine_similarity(user_vec, mat.values).flatten()
    sim_series = pd.Series(sims, index=mat.index)
    weighted = (mat.T * sim_series).sum(axis=1)
    if weighted.max()!=0: weighted = weighted/weighted.max()
    scores = pd.Series(0,index=range(len(df)))
    for idx,val in weighted.items(): scores.iloc[idx]=val
    return scores

# -------------------------
# Recommendation core
# -------------------------
def compute_recommendations(query_text, top_n=TOP_N_DEFAULT, price_filter="All", search_term="",
                            use_collab=True, show_explanation=True):
    q_vec = vectorizer.transform([query_text])
    sims = cosine_similarity(q_vec, matrix).flatten()
    df_local = df.copy()
    df_local["similarity_score"] = sims

    if search_term:
        mask = df_local["course_title"].str.contains(search_term, case=False, na=False) | \
               df_local["subject"].str.contains(search_term, case=False, na=False)
        df_local = df_local[mask]
    if price_filter.lower()=="free": df_local = df_local[df_local["price"].str.contains("free", case=False, na=False)]
    elif price_filter.lower()=="paid": df_local = df_local[~df_local["price"].str.contains("free", case=False, na=False)]

    df_local["popularity_score"] = df_local["num_subscribers"] / (df_local["num_subscribers"].max() or 1)
    collab_scores = get_collaborative_scores_for_user(st.session_state.user_id) if (use_collab and st.session_state.user_id) else None
    df_local["collaborative_score"] = df_local.index.map(lambda i: float(collab_scores.iloc[i]) if (collab_scores is not None and i<len(collab_scores)) else 0.0)

    df_local["final_score"] = (df_local["similarity_score"]*HYBRID_SIM_WEIGHT +
                               df_local["collaborative_score"]*HYBRID_COLLAB_WEIGHT +
                               df_local["popularity_score"]*HYBRID_POP_WEIGHT)
    df_local = df_local.sort_values("final_score", ascending=False)
    return df_local.head(top_n)

# -------------------------
# DB helpers
# -------------------------
def get_user_by_username(username):
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    return c.fetchone()
def register_user(username, interest, level, hours):
    c.execute("INSERT OR IGNORE INTO users (username, interest, skill_level, learning_hours) VALUES (?,?,?,?)",
              (username, interest, level, hours))
    conn.commit()
    return get_user_by_username(username)
def save_recommendation(user_id,row):
    try:
        c.execute("""INSERT INTO recommendations (user_id, course_title, subject, level, price, num_subscribers, similarity_score, popularity_score)
                     VALUES (?,?,?,?,?,?,?,?)""",
                  (user_id,row["course_title"],row["subject"],row["level"],str(row.get("price","")),int(row.get("num_subscribers",0)),float(row.get("similarity_score",0)),float(row.get("popularity_score",0))))
        conn.commit()
    except sqlite3.IntegrityError:
        c.execute("""UPDATE recommendations SET similarity_score=?, popularity_score=? WHERE user_id=? AND course_title=?""",
                  (float(row.get("similarity_score",0)), float(row.get("popularity_score",0)), user_id, row["course_title"]))
        conn.commit()
def fetch_saved(user_id,limit=200):
    c.execute("""SELECT course_title, subject, level, price, num_subscribers, similarity_score, popularity_score
                 FROM recommendations WHERE user_id=? ORDER BY id DESC LIMIT ?""", (user_id,limit))
    rows = c.fetchall()
    cols = ["course_title","subject","level","price","num_subscribers","similarity_score","popularity_score"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
def clear_saved(user_id):
    c.execute("DELETE FROM recommendations WHERE user_id=?", (user_id,))
    conn.commit()
def save_feedback(user_id,course_title,rating,comment=""):
    try:
        c.execute("""INSERT INTO feedback (user_id, course_title, rating, comment) VALUES (?,?,?,?)""",
                  (user_id,course_title,rating,comment))
        conn.commit()
    except sqlite3.IntegrityError:
        c.execute("""UPDATE feedback SET rating=?, comment=?, created_at=CURRENT_TIMESTAMP WHERE user_id=? AND course_title=?""",
                  (rating,comment,user_id,course_title))
        conn.commit()

# -------------------------
# Session state init
# -------------------------
if "user_id" not in st.session_state: st.session_state.user_id = None
if "username" not in st.session_state: st.session_state.username = None

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("👤 Account")
sidebar_username_input = st.sidebar.text_input("Username", key="username_input_sidebar")

saved_interest = ""; saved_level = "Beginner"; saved_hours = 5; user=None

if st.session_state.username:
    user = get_user_by_username(st.session_state.username)
    if user:
        st.sidebar.success(f"Welcome back, {st.session_state.username}!")
        # ✅ FIXED LOGOUT SECTION
        if st.sidebar.button("🚪 Logout"):
            st.session_state.user_id = None
            st.session_state.username = None
            if "username_input_sidebar" in st.session_state:
                st.session_state.pop("username_input_sidebar")
            st.sidebar.success("You have been logged out successfully.")
            safe_rerun()
        saved_interest = user[2] or ""; saved_level = user[3] or "Beginner"; saved_hours = user[4] or 5
else:
    if sidebar_username_input:
        existing = get_user_by_username(sidebar_username_input)
        if existing:
            st.sidebar.success(f"Welcome back, {sidebar_username_input}!")
            st.session_state.user_id = existing[0]; st.session_state.username = sidebar_username_input
            saved_interest = existing[2] or ""; saved_level = existing[3] or "Beginner"; saved_hours = existing[4] or 5
            safe_rerun()
        else:
            st.sidebar.info("New user. Fill preferences and Register.")
            saved_interest = st.sidebar.text_input("Your interests (comma separated):", key="reg_interest")
            saved_level = st.sidebar.selectbox("Skill Level:", ["Beginner","Intermediate","Expert"], key="reg_level")
            saved_hours = st.sidebar.slider("Hours per week:",1,40,5,key="reg_hours")
            if st.sidebar.button("Register"):
                new_user = register_user(sidebar_username_input, saved_interest, saved_level, saved_hours)
                st.sidebar.success("Registered! You are now logged in.")
                st.session_state.user_id = new_user[0]; st.session_state.username = sidebar_username_input
                safe_rerun()

st.sidebar.header("🔎 Search & Filters")
search_term = st.sidebar.text_input("Search courses (title/subject):", value="")
price_filter = st.sidebar.selectbox("Price filter:", ["All","Free","Paid"], index=0)
num_results = st.sidebar.slider("Number of recommendations:",3,20,TOP_N_DEFAULT)
use_collab = st.sidebar.checkbox("Use collaborative component (if available)", value=True)

st.sidebar.header("📊 Dataset Insights")
st.sidebar.metric("Total Courses", len(df))
free_mask = df["price"].str.contains("free",case=False,na=False)
st.sidebar.metric("Free Courses", free_mask.sum())
st.sidebar.metric("Paid Courses", (~free_mask).sum())

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["🔮 Recommendations", "📚 My Saved", "⚙️ Profile", "📈 Evaluation", "🛠️ Admin"])

# === Recommendations tab
with tabs[0]:
    if st.session_state.user_id:
        query_text = f"{saved_interest} {saved_level}"
        recs = compute_recommendations(query_text, top_n=num_results, price_filter=price_filter,
                                       search_term=search_term, use_collab=use_collab)
        if recs.empty:
            st.info("No courses found for your preferences.")
        else:
            st.markdown("<div class='cards-grid'>", unsafe_allow_html=True)
            for _, row in recs.iterrows():
                level_cls="lvl-beginner"
                if "int" in row.get("level","").lower(): level_cls="lvl-intermediate"
                if "exp" in row.get("level","").lower(): level_cls="lvl-expert"
                st.markdown(f"""
                    <div class='course-card'>
                      <div class='course-title'>{row['course_title']}</div>
                      <div class='course-meta'>{row['subject']} — {row['price']} — 👥 {int(row['num_subscribers']):,}</div>
                      <span class='badge {level_cls}'>{row['level']}</span>
                      <div class='score-row'>Similarity: {row['similarity_score']:.2f} | Pop: {row['popularity_score']:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
                save_recommendation(st.session_state.user_id,row)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Please log in or register from the sidebar to see recommendations.")

# === My Saved tab
with tabs[1]:
    if st.session_state.user_id:
        df_saved = fetch_saved(st.session_state.user_id)
        if df_saved.empty:
            st.info("You have no saved recommendations yet.")
        else:
            st.dataframe(df_saved)
            if st.button("🧹 Clear all saved"):
                clear_saved(st.session_state.user_id)
                st.success("Cleared.")
                safe_rerun()
    else:
        st.info("Please log in first.")

# === Profile tab
with tabs[2]:
    if st.session_state.user_id:
        st.subheader("Update Profile")
        new_interest = st.text_input("Interests:", value=saved_interest)
        new_level = st.selectbox("Skill Level:", ["Beginner","Intermediate","Expert"], index=["Beginner","Intermediate","Expert"].index(saved_level))
        new_hours = st.slider("Hours per week:",1,40,saved_hours)
        if st.button("💾 Save changes"):
            c.execute("UPDATE users SET interest=?, skill_level=?, learning_hours=? WHERE id=?",
                      (new_interest,new_level,new_hours,st.session_state.user_id))
            conn.commit()
            st.success("Profile updated!")
            safe_rerun()
    else:
        st.info("Please log in first.")

# === Evaluation tab
with tabs[3]:
    if st.session_state.user_id:
        st.subheader("Feedback & Ratings")
        df_saved = fetch_saved(st.session_state.user_id)
        if df_saved.empty:
            st.info("No courses saved yet.")
        else:
            chosen_course = st.selectbox("Choose a course to rate:", df_saved["course_title"].tolist())
            rating = st.slider("Rating (1-5):",1,5,3)
            comment = st.text_area("Comment (optional):")
            if st.button("💬 Submit Feedback"):
                save_feedback(st.session_state.user_id, chosen_course, rating, comment)
                st.success("Feedback saved!")

        st.subheader("Ratings Distribution")
        fb = pd.read_sql_query("SELECT rating FROM feedback", conn)
        if not fb.empty:
            fig, ax = plt.subplots()
            fb["rating"].plot(kind="hist", bins=[1,2,3,4,5,6], ax=ax)
            st.pyplot(fig)
    else:
        st.info("Please log in first.")

# === Admin tab
with tabs[4]:
    st.subheader("Admin — All Users Summary")
    users_df = pd.read_sql_query("SELECT * FROM users", conn)
    st.dataframe(users_df)
    recs_df = pd.read_sql_query("SELECT * FROM recommendations", conn)
    st.dataframe(recs_df.head(100))
    fb_df = pd.read_sql_query("SELECT * FROM feedback", conn)
    st.dataframe(fb_df)

st.caption("© 2025 AI_LAB_PROJECT — Course Recommendation System")
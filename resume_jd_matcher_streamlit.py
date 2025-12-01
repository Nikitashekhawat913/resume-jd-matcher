# resume_jd_matcher_streamlit.py - Final Version (PyPDF2 Compatible for Deployment)

import streamlit as st
import PyPDF2
import re
from collections import Counter

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

# ------------------------------------------------
# UI & Styling
# ------------------------------------------------
st.set_page_config(page_title="Resume–JD Match Analyzer", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 18px !important;
}
h1, h2 {
    font-size: 34px !important;
    font-weight: 800 !important;
}
.stFileUploader, .stTextArea label {
    font-size: 22px !important;
    font-weight: 650 !important;
}
.stButton button {
    font-size: 22px !important;
    font-weight: 700 !important;
    padding: 12px 25px !important;
    border-radius: 10px !important;
}
.stAlert {
    font-size: 18px !important;
}
</style>
""", unsafe_allow_html=True)

st.title(" Resume–JD Match Analyzer")
st.write("Upload your resume PDF and paste the job description to get a smart ATS-style match score and improvement tips.")

# ------------------------------------------------
# Skills Database
# ------------------------------------------------
COMMON_SKILLS = [
    "python","java","c","c++","c#","sql","excel","power bi","tableau",
    "nlp","machine learning","deep learning","tensorflow","pytorch",
    "git","linux","aws","azure","docker","kubernetes","data analysis",
    "data visualization","pandas","numpy","scikit-learn","html","css",
    "javascript","react","node.js","matlab","r","hadoop","spark","sas",
    "communication","management","opencv","streamlit","mongodb","flask",
    "django","postgresql","mysql","json","rest apis","cloud computing",
    "data structures","algorithms","presentation skills","leadership"
]

SKILL_SYNONYMS = {
    "ml":"machine learning",
    "dl":"deep learning",
    "js":"javascript",
    "eda":"data analysis",
    "nosql":"mongodb",
    "ai":"machine learning",
    "dsa":"data structures"
}

YEAR_PATTERN = re.compile(r"(\d+)\s*\+?\s*(?:years|year|yrs)", re.IGNORECASE)

# ------------------------------------------------
# Upload UI
# ------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("📎 Upload Resume (PDF)", type=["pdf"])
    resume_text = ""
    if uploaded_file:
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                resume_text += page.extract_text() or ""
            st.success("Resume uploaded and text extracted.")
        except Exception as e:
            st.error("❌ Unable to extract text from PDF!")

with col2:
    jd_text = st.text_area("📌 Paste Job Description (JD) here", height=260)

st.markdown("---")

# ------------------------------------------------
# Helper Functions
# ------------------------------------------------
def clean_text(t):
    return re.sub(r"\s+", " ", t.lower()).strip()

def extract_skills(text):
    found = set()
    text = text.lower()
    for s in COMMON_SKILLS:
        if s in text: found.add(s)
    for short, full in SKILL_SYNONYMS.items():
        if short in text: found.add(full)
    return sorted(found)

def extract_keywords(text, top_k=40):
    doc = nlp(text)
    words = [chunk.root.lemma_.lower() for chunk in doc.noun_chunks]
    words += [ent.text.lower() for ent in doc.ents]
    freq = Counter(words)
    return [w for w,_ in freq.most_common(top_k)]

def extract_years(text):
    matches = YEAR_PATTERN.findall(text)
    return [int(m) for m in matches if m.isdigit()]

def tfidf_sim(a,b):
    try:
        vect = TfidfVectorizer(stop_words="english")
        X = vect.fit_transform([a,b])
        return float(cosine_similarity(X[0],X[1])[0][0])
    except:
        return 0.0

# ------------------------------------------------
# Suggestion Engine
# ------------------------------------------------
def generate_suggestions(missing, fuzzy, jd_keywords, resume_keywords, exp_req, exp_have, tfidf):
    suggestions = {
        "missing_core": [],
        "missing_tools": [],
        "keyword_opt": [],
        "experience": [],
        "formatting": []
    }

    core_missing = [m for m in missing if m not in ["communication","management"]]
    if core_missing:
        suggestions["missing_core"].append(
            "Add required core skills: *" + ", ".join(core_missing) + "*")

    tool_skills = ["aws","azure","docker","git","power bi","excel","tableau","mysql","postgresql"]
    missing_tools = [t for t in missing if t in tool_skills]
    if missing_tools:
        suggestions["missing_tools"].append(
            "Add missing tools: *" + ", ".join(missing_tools) + "*")

    if fuzzy:
        suggestions["keyword_opt"].append(
            "Match JD terms more precisely: *" + ", ".join(fuzzy) + "*")

    important_kw = [k for k in jd_keywords[:8] if k not in resume_keywords][:5]
    if important_kw:
        suggestions["keyword_opt"].append(
            "Include high-value keywords: *" + ", ".join(important_kw) + "*")

    if exp_req:
        if exp_have < exp_req:
            suggestions["experience"].append(
                f"JD needs *{exp_req}+ yrs* experience. You show **{exp_have} yrs** — add project durations.")

    if tfidf < 0.30:
        suggestions["keyword_opt"].append("Increase alignment by using JD phrases in resume.")

    suggestions["formatting"].append("Ensure ATS format — avoid images, tables, icons.")
    suggestions["formatting"].append("Follow order: Skills → Experience → Projects → Education.")

    return suggestions

# ------------------------------------------------
# Match Logic
# ------------------------------------------------
def compute_match(resume, jd):
    jd_sk = extract_skills(jd)
    rs_sk = extract_skills(resume)
    jd_kw = extract_keywords(jd)
    rs_kw = extract_keywords(resume)

    exact = set(jd_sk) & set(rs_sk)
    missing = set(jd_sk) - exact

    fuzzy = set([m for m in missing if any(tok in r for tok in m.split() for r in rs_kw)])

    tfidf = tfidf_sim(resume, jd)
    jd_exp = min(extract_years(jd)) if extract_years(jd) else None
    rs_exp_list = extract_years(resume)
    rs_exp = sum(rs_exp_list)/len(rs_exp_list) if rs_exp_list else 0

    score = (
        0.6 * (len(exact)/len(jd_sk) if jd_sk else 0) +
        0.4 * tfidf
    ) * 100

    suggestions = generate_suggestions(missing, fuzzy, jd_kw, rs_kw, jd_exp, rs_exp, tfidf)

    return {
        "score": round(score,1),
        "jd_skills": jd_sk,
        "resume_skills": rs_sk,
        "suggestions": suggestions
    }

# ------------------------------------------------
# Run Button
# ------------------------------------------------
if st.button("🔍 Analyze Match"):
    if uploaded_file and jd_text.strip():
        clean_r = clean_text(resume_text)
        clean_j = clean_text(jd_text)

        with st.spinner("Analyzing..."):
            result = compute_match(clean_r, clean_j)

        # Score Display
        st.markdown(
            f"""
            <div style="background:#003300; padding:25px; border-radius:10px;
            text-align:center; font-size:50px; font-weight:900; color:#00FF7F;">
            ✨ {result['score']}% Match ✨
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Skills Overview")
        st.write(f"**JD Skills:** {', '.join(result['jd_skills'])}")
        st.write(f"**Resume Skills:** {', '.join(result['resume_skills'])}")

        st.subheader("💡 Smart Suggestions")
        for cat, items in result["suggestions"].items():
            if items:
                st.write(f"🔹 **{cat.title().replace('_',' ')}**")
                for i in items:
                    st.write("• " + i)
    else:
        st.error("⚠ Please upload resume and paste JD!")

import streamlit as st
import pandas as pd
import pdfplumber
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI ATS Resume Screener & Analyzer",
    page_icon="✨",
    layout="wide"
)

# Initialize Session State Variables to prevent data loss across tabs
if "jd_text" not in st.session_state:
    st.session_state["jd_text"] = ""
if "required_skills_input" not in st.session_state:
    st.session_state["required_skills_input"] = ""
if "ats_results" not in st.session_state:
    st.session_state["ats_results"] = pd.DataFrame()

# ---------------------------------------------------------
# Custom Light Theme CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #FAF9F6;
        font-family: 'Inter', sans-serif;
    }
    
    .header-banner {
        background: linear-gradient(135deg, #E0C3FC 0%, #8EC5FC 100%);
        padding: 22px 30px;
        border-radius: 20px;
        color: #2D3748;
        box-shadow: 0 10px 25px -5px rgba(142, 197, 252, 0.4);
        margin-bottom: 25px;
        text-align: center;
    }
    .header-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #312E81;
        margin: 0;
    }
    .header-subtitle {
        font-size: 1.05rem;
        color: #4C1D95;
        margin-top: 5px;
    }

    div[data-testid="stHorizontalBlock"] {
        background: #FFFFFF;
        padding: 10px 18px;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
        border: 1px solid #EEF2F6;
        margin-bottom: 25px;
    }

    .stat-card {
        background: white;
        padding: 16px;
        border-radius: 16px;
        border: 1px solid #F1F5F9;
        box-shadow: 0 4px 12px rgba(148, 163, 184, 0.08);
        text-align: center;
    }
    .stat-card-pink { border-top: 5px solid #F472B6; }
    .stat-card-purple { border-top: 5px solid #A78BFA; }
    .stat-card-blue { border-top: 5px solid #60A5FA; }
    .stat-card-green { border-top: 5px solid #34D399; }

    .badge-matched {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.88rem;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
    }
    .badge-missing {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.88rem;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
    }
    .badge-role {
        background-color: #EDE9FE;
        color: #5B21B6;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.88rem;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
    }

    .stButton>button {
        background: linear-gradient(135deg, #818CF8 0%, #C084FC 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 4px 14px rgba(168, 85, 247, 0.35);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        color: white;
    }

    h1, h2, h3 {
        color: #312E81 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Document Extraction Helpers
# ---------------------------------------------------------
def read_pdf(file) -> str:
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
    except Exception:
        pass
    return text.strip()

def read_docx(file) -> str:
    try:
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs]).strip()
    except Exception:
        return ""

def extract_resume_text(file) -> str:
    if file.name.endswith(".pdf"):
        return read_pdf(file)
    elif file.name.endswith(".docx"):
        return read_docx(file)
    return ""

def extract_email(text: str) -> str:
    match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    return match.group(0) if match else "Not found"

def extract_phone(text: str) -> str:
    match = re.search(r'(\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}', text)
    return match.group(0) if match else "Not found"

def extract_experience(text: str) -> float:
    pattern = r'(\d+(\.\d+)?)\+?\s*(years?|yrs?)\s*(of)?\s*(experience|exp)?'
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        years = [float(m[0]) for m in matches if float(m[0]) < 40]
        if years:
            return max(years)
    return 0.0

def extract_education(text: str) -> str:
    degrees = []
    keywords = ["Bachelor", "Master", "B.Tech", "M.Tech", "B.S.", "M.S.", "B.E.", "Ph.D", "Degree", "Diploma"]
    for line in text.split('\n'):
        for kw in keywords:
            if kw.lower() in line.lower() and len(line) < 100:
                degrees.append(line.strip())
                break
    return " | ".join(set(degrees)) if degrees else "Not specified"

def extract_skills(text: str, skill_pool: list) -> list:
    text_lower = text.lower()
    found = []
    for skill in skill_pool:
        pattern = r'\b' + re.escape(skill.strip().lower()) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill.strip())
    return list(set(found))

# ---------------------------------------------------------
# Simple Evaluation Scoring Engine
# ---------------------------------------------------------
JOB_ROLES = {
    "AI / Machine Learning Engineer": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn", "nlp", "sql", "fastapi", "docker"],
    "Full Stack Developer": ["javascript", "react", "node.js", "python", "html", "css", "sql", "mongodb", "git", "rest api"],
    "Data Analyst / Scientist": ["python", "sql", "pandas", "numpy", "power bi", "tableau", "statistics", "data visualization", "excel"],
    "DevOps & Cloud Engineer": ["docker", "kubernetes", "aws", "azure", "linux", "ci/cd", "terraform", "bash", "python", "git"],
    "Backend Engineer": ["python", "java", "sql", "fastapi", "django", "postgresql", "docker", "redis", "microservices"]
}

def recommend_best_role(resume_text: str, candidate_skills: list):
    scores = {}
    for role, skills in JOB_ROLES.items():
        matched = set(candidate_skills).intersection(set(skills))
        skill_score = (len(matched) / len(skills)) * 100 if skills else 0
        scores[role] = round(skill_score, 1)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[0], sorted_scores

def compute_match_score(resume_text: str, candidate_skills: list, jd_text: str, required_skills: list):
    if not resume_text or not jd_text:
        return 0.0, 0.0, 0.0

    if required_skills:
        matched = set(candidate_skills).intersection(set(required_skills))
        skill_score = (len(matched) / len(required_skills)) * 100
    else:
        skill_score = 100.0

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([resume_text, jd_text])
    content_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100

    final_score = (0.5 * skill_score) + (0.5 * content_score)
    
    return round(final_score, 1), round(content_score, 1), round(skill_score, 1)

default_skill_pool = ["python", "sql", "java", "c++", "machine learning", "deep learning", 
                      "fastapi", "docker", "aws", "kubernetes", "react", "javascript", 
                      "pandas", "scikit-learn", "html", "css", "git", "tableau", "power bi", "excel"]

# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="header-banner">
    <div class="header-title">✨ Smart AI Resume Screener & ATS</div>
    <div class="header-subtitle">Simple, explainable, and visual candidate evaluation platform</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Navigation Bar
# ---------------------------------------------------------
selected_tab = st.radio(
    "Navigation Menu",
    ["🔍 ATS Resume Screener", "⚔️ Candidate Comparison", "🎯 Job Role Recommender"],
    horizontal=True,
    label_visibility="collapsed"
)

st.write("")

# ---------------------------------------------------------
# TAB 1: ATS RESUME SCREENER
# ---------------------------------------------------------
if selected_tab == "🔍 ATS Resume Screener":
    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        st.markdown("### 📝 1. Job Requirements & Resumes")
        
        # Save inputs directly to st.session_state so they persist when returning
        jd_input = st.text_area("Job Description", value=st.session_state["jd_text"], height=180, placeholder="Paste job description here...")
        st.session_state["jd_text"] = jd_input

        skills_input = st.text_input("Required Skills (separated by commas)", value=st.session_state["required_skills_input"], placeholder="e.g. Python, SQL, Docker, Machine Learning")
        st.session_state["required_skills_input"] = skills_input
        
        user_skills = [s.strip().lower() for s in skills_input.split(",") if s.strip()]
        full_skills = list(set(default_skill_pool + user_skills))

        uploaded_files = st.file_uploader("Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

    with col_right:
        st.markdown("### 📊 2. Screening Results")
        
        if st.button("⚡ Screen Resumes Now", use_container_width=True):
            if not jd_input.strip():
                st.warning("Please paste a Job Description first.")
            elif not uploaded_files:
                st.warning("Please upload at least one resume file.")
            else:
                results = []
                with st.spinner("Screening candidate resumes..."):
                    for file in uploaded_files:
                        raw_text = extract_resume_text(file)
                        if not raw_text:
                            continue

                        email = extract_email(raw_text)
                        phone = extract_phone(raw_text)
                        exp = extract_experience(raw_text)
                        edu = extract_education(raw_text)
                        candidate_skills = extract_skills(raw_text, full_skills)

                        matched_skills = list(set(candidate_skills).intersection(set(user_skills)))
                        missing_skills = list(set(user_skills) - set(matched_skills))

                        overall, content, skill_score = compute_match_score(raw_text, candidate_skills, jd_input, user_skills)
                        top_role, _ = recommend_best_role(raw_text, candidate_skills)

                        results.append({
                            "Candidate File": file.name,
                            "Match Score (%)": overall,
                            "Text Match (%)": content,
                            "Skill Match (%)": skill_score,
                            "Exp (Yrs)": exp,
                            "Email": email,
                            "Phone": phone,
                            "Education": edu,
                            "Matched Skills": matched_skills,
                            "Missing Skills": missing_skills,
                            "Recommended Role": top_role[0],
                            "Raw Text": raw_text
                        })

                if results:
                    df = pd.DataFrame(results)
                    df = df.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)
                    st.session_state["ats_results"] = df

        if not st.session_state["ats_results"].empty:
            df = st.session_state["ats_results"]
            top = df.iloc[0]

            st.success(f"🏆 **Top Match:** {top['Candidate File']} ({top['Match Score (%)']}% Fit)")

            # Overall Rankings Bar Chart
            fig_bar = px.bar(
                df,
                x="Match Score (%)",
                y="Candidate File",
                orientation='h',
                color="Match Score (%)",
                color_continuous_scale=["#C4B5FD", "#818CF8", "#34D399"],
                text="Match Score (%)",
                title="Candidate Match Rankings"
            )
            fig_bar.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20),
                height=250
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.dataframe(
                df[["Candidate File", "Match Score (%)", "Exp (Yrs)", "Recommended Role", "Email"]],
                use_container_width=True,
                hide_index=True
            )

            csv = df[["Candidate File", "Match Score (%)", "Exp (Yrs)", "Email", "Phone", "Education", "Recommended Role"]].to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Screening Table (CSV)", data=csv, file_name="resume_screening_results.csv", mime="text/csv")

            st.divider()
            st.markdown("### 🔎 Candidate Inspector & Breakdown")

            selected_cand = st.selectbox("Select candidate to examine:", df["Candidate File"].tolist())
            cand = df[df["Candidate File"] == selected_cand].iloc[0]

            # Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="stat-card stat-card-purple"><div style="font-size:0.85rem; color:#6B7280;">OVERALL MATCH</div><div style="font-size:1.8rem; font-weight:800; color:#5B21B6;">{cand["Match Score (%)"]}%</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="stat-card stat-card-green"><div style="font-size:0.85rem; color:#6B7280;">SKILL MATCH SCORE</div><div style="font-size:1.8rem; font-weight:800; color:#065F46;">{cand["Skill Match (%)"]}%</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="stat-card stat-card-blue"><div style="font-size:0.85rem; color:#6B7280;">TEXT SIMILARITY</div><div style="font-size:1.8rem; font-weight:800; color:#1E40AF;">{cand["Text Match (%)"]}%</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="stat-card stat-card-pink"><div style="font-size:0.85rem; color:#6B7280;">EXPERIENCE</div><div style="font-size:1.8rem; font-weight:800; color:#9D174D;">{cand["Exp (Yrs)"]} Yrs</div></div>', unsafe_allow_html=True)

            st.write("")
            st.write(f"📧 **Email:** `{cand['Email']}` | 📞 **Phone:** `{cand['Phone']}` | 🎓 **Education:** `{cand['Education']}`")
            st.write(f"🎯 **Best Suited Role:** <span class='badge-role'>{cand['Recommended Role']}</span>", unsafe_allow_html=True)

            st.write("")

            # Progress Bars
            st.markdown("#### 📊 Evaluation Breakdown")
            st.write(f"**Skill Match Score:** {cand['Skill Match (%)']}%")
            st.progress(int(cand['Skill Match (%)']))

            st.write(f"**Text Similarity Score:** {cand['Text Match (%)']}%")
            st.progress(int(cand['Text Match (%)']))

            exp_pct = min(int((cand['Exp (Yrs)'] / 10.0) * 100), 100)
            st.write(f"**Experience Level Score:** {cand['Exp (Yrs)']} Years ({exp_pct}%)")
            st.progress(exp_pct)

            st.divider()

            # Skill Badges
            st.markdown("### 🏷️ Skill Badges")
            col_m, col_u = st.columns(2)
            with col_m:
                st.write("**✓ Matched Skills:**")
                if cand['Matched Skills']:
                    st.markdown(" ".join([f"<span class='badge-matched'>✓ {s}</span>" for s in cand['Matched Skills']]), unsafe_allow_html=True)
                else:
                    st.write("None matched.")

            with col_u:
                st.write("**✗ Missing Skills:**")
                if cand['Missing Skills']:
                    st.markdown(" ".join([f"<span class='badge-missing'>✗ {s}</span>" for s in cand['Missing Skills']]), unsafe_allow_html=True)
                else:
                    st.write("No missing skills!")

            with st.expander("📄 Show Parsed Text Content"):
                st.text_area("Extracted Text", cand["Raw Text"], height=200, disabled=True)

# ---------------------------------------------------------
# TAB 2: CANDIDATE COMPARISON
# ---------------------------------------------------------
elif selected_tab == "⚔️ Candidate Comparison":
    st.markdown("### ⚔️ Side-by-Side Candidate Comparison")

    if st.session_state["ats_results"].empty:
        st.info("👉 Please upload and screen candidates in the **🔍 ATS Resume Screener** tab first.")
    else:
        df = st.session_state["ats_results"]
        
        if len(df) < 2:
            st.warning("⚠️ You need at least **2 uploaded resumes** to use the side-by-side comparison.")
            st.write("Current Candidates Screened:")
            st.dataframe(df[["Candidate File", "Match Score (%)", "Exp (Yrs)"]], use_container_width=True)
        else:
            candidates_list = df["Candidate File"].tolist()

            c_select1, c_select2 = st.columns(2)
            with c_select1:
                cand1_name = st.selectbox("Select Candidate 1:", candidates_list, index=0)
            with c_select2:
                cand2_name = st.selectbox("Select Candidate 2:", candidates_list, index=1 if len(candidates_list) > 1 else 0)

            c1 = df[df["Candidate File"] == cand1_name].iloc[0]
            c2 = df[df["Candidate File"] == cand2_name].iloc[0]

            st.divider()

            # Comparative Visual Headers
            col1, col2 = st.columns(2, gap="large")

            with col1:
                st.markdown(f"### 👤 {c1['Candidate File']}")
                st.markdown(f'<div class="stat-card stat-card-purple"><div style="font-size:0.85rem; color:#6B7280;">OVERALL MATCH</div><div style="font-size:2rem; font-weight:800; color:#5B21B6;">{c1["Match Score (%)"]}%</div></div>', unsafe_allow_html=True)
                
                st.write("")
                st.write(f"**Experience:** `{c1['Exp (Yrs)']} Years`")
                st.write(f"**Email:** `{c1['Email']}`")
                st.write(f"**Best Suited Role:** {c1['Recommended Role']}")

                st.write("#### Score Breakdown")
                st.write(f"**Skill Match:** {c1['Skill Match (%)']}%")
                st.progress(int(c1['Skill Match (%)']))

                st.write(f"**Text Match:** {c1['Text Match (%)']}%")
                st.progress(int(c1['Text Match (%)']))

                st.write("#### Matched Skills")
                if c1['Matched Skills']:
                    st.markdown(" ".join([f"<span class='badge-matched'>{s}</span>" for s in c1['Matched Skills']]), unsafe_allow_html=True)
                else:
                    st.write("None")

            with col2:
                st.markdown(f"### 👤 {c2['Candidate File']}")
                st.markdown(f'<div class="stat-card stat-card-blue"><div style="font-size:0.85rem; color:#6B7280;">OVERALL MATCH</div><div style="font-size:2rem; font-weight:800; color:#1E40AF;">{c2["Match Score (%)"]}%</div></div>', unsafe_allow_html=True)

                st.write("")
                st.write(f"**Experience:** `{c2['Exp (Yrs)']} Years`")
                st.write(f"**Email:** `{c2['Email']}`")
                st.write(f"**Best Suited Role:** {c2['Recommended Role']}")

                st.write("#### Score Breakdown")
                st.write(f"**Skill Match:** {c2['Skill Match (%)']}%")
                st.progress(int(c2['Skill Match (%)']))

                st.write(f"**Text Match:** {c2['Text Match (%)']}%")
                st.progress(int(c2['Text Match (%)']))

                st.write("#### Matched Skills")
                if c2['Matched Skills']:
                    st.markdown(" ".join([f"<span class='badge-matched'>{s}</span>" for s in c2['Matched Skills']]), unsafe_allow_html=True)
                else:
                    st.write("None")

# ---------------------------------------------------------
# TAB 3: JOB ROLE RECOMMENDER
# ---------------------------------------------------------
elif selected_tab == "🎯 Job Role Recommender":
    st.markdown("### 🎯 Automatic Job Role Recommender")
    st.caption("Paste candidate resume text to see which tech job profile suits them best.")

    resume_input = st.text_area("Paste Resume Text", height=220, placeholder="Paste resume text here...")

    if st.button("🎯 Analyze Suitable Roles", use_container_width=True):
        if not resume_input.strip():
            st.warning("Please paste resume text.")
        else:
            cand_skills = extract_skills(resume_input, default_skill_pool)
            top_role, all_roles = recommend_best_role(resume_input, cand_skills)

            st.success(f"🏆 Best Recommended Role Profile: **{top_role[0]}** ({top_role[1]}% Match)")

            st.divider()
            
            roles_df = pd.DataFrame(all_roles, columns=["Job Profile", "Fit Score (%)"])
            
            fig_roles = px.bar(
                roles_df,
                x="Fit Score (%)",
                y="Job Profile",
                orientation='h',
                color="Fit Score (%)",
                color_continuous_scale=["#C4B5FD", "#F472B6"],
                text="Fit Score (%)",
                title="Job Profile Match Distribution"
            )
            fig_roles.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=320
            )
            st.plotly_chart(fig_roles, use_container_width=True)

            st.dataframe(roles_df, use_container_width=True, hide_index=True)
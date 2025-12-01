📄 Resume–JD Match Analyzer
🚀 Smart ATS Match Using NLP & Machine Learning

A Streamlit-based web application that analyzes a candidate’s resume against a Job Description (JD) using Natural Language Processing (NLP).
It provides:

✔ ATS Match Score (in %)
✔ Missing Skills & Tools
✔ Keyword Optimization Suggestions
✔ Experience Requirement Check
✔ Resume Formatting Tips
✔ Clean, Dark-Theme User Interface

This tool helps job seekers improve resume alignment before applying — increasing chances of ATS selection.

✨ Key Features
Feature	Description
Resume Upload	Accepts PDF resume and extracts content automatically
JD Input	User pastes job description text in the app
Smart Skill Matching	Exact + fuzzy matching using NLP
TF-IDF Based Similarity	Measures keyword alignment
ATS-Friendly Suggestions	Core skills, tools, keywords, formatting
Experience Check	Compares JD’s required experience vs resume
Fully Client-Based	No database — privacy friendly
🧠 Tech Stack
Component	Technology
Language	Python
UI Framework	Streamlit
NLP	spaCy
Machine Learning	Scikit-learn (TF-IDF + Cosine Similarity)
PDF Extraction	pdfplumber
📌 Installation
# Clone repository
git clone https://github.com/YOUR-USERNAME/resume-jd-matcher.git
cd resume-jd-matcher

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the app
streamlit run resume_jd_matcher_streamlit.py

🎯 How It Works

1️⃣ Upload resume PDF
2️⃣ Paste JD text
3️⃣ Click Analyze Match
4️⃣ Suggest ATS formate + improvements
5️⃣ Update resume accordingly ✔

📍 Output Preview
Section	Example
Match Score	78% Match
Skill Insights	JD vs Resume skills
Suggestions	Missing keywords / tools / experience
ATS Formatting Tips	Improve structure and readability
📌 Project Status

🟢 Completed — ready for deployment & demonstration
✨ Future plan: add PDF highlight & report download

🙌 Author

Nikita Shekhawat
📍 Data Science Engineering Student
🚀 Passionate about NLP, Resume Analytics & Career Tech

⭐ Support the Project

If you like this project, please ⭐ the repo!

Every star motivates further improvements 😃

!pip install gradio pandas scikit-learn pdfminer.six matplotlib
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import os

from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Extract resume text
def extract_resume_text(file):
    return extract_text(file.name)


# Ranking logic
def rank_resumes(files, job_description):

    resume_texts = []
    resume_names = []

    for file in files:

        text = extract_resume_text(file)
        resume_texts.append(text)

        # Extract only file name
        name = os.path.basename(file.name)
        resume_names.append(name)

    documents = [job_description] + resume_texts

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity_scores = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    ).flatten()

    results = pd.DataFrame({
        "Resume Name": resume_names,
        "Match Score": similarity_scores
    })

    results = results.sort_values(
        by="Match Score",
        ascending=False
    )

    # Best candidate name
    best_candidate = results.iloc[0]["Resume Name"]

    # Visualization
    plt.figure(figsize=(8,5))

    plt.bar(
        results["Resume Name"],
        results["Match Score"],
        color="#1f77ff"
    )

    plt.title("Candidate Ranking")

    plt.xlabel("Resume")

    plt.ylabel("Match Score")

    plt.xticks(rotation=30)

    plt.tight_layout()

    return results, f"Best Candidate: {best_candidate}", plt


# Dark Blue UI
custom_css = """

body {
    background: linear-gradient(135deg, #000000, #0f2027, #1f3c88);
}

h1 {
    color: #4da6ff;
    text-align: center;
}

button {
    background-color: #1f77ff !important;
    color: white !important;
    border-radius: 10px !important;
}

textarea {
    background-color: #111 !important;
    color: white !important;
}

label {
    color: white !important;
}

"""


# Build Dashboard
with gr.Blocks(css=custom_css) as demo:

    gr.Markdown("# AI Resume Screening System")

    with gr.Row():

        resumes = gr.File(
            file_count="multiple",
            label="Upload Resumes (PDF)"
        )

        job_desc = gr.Textbox(
            lines=6,
            label="Enter Job Description"
        )

    analyze_btn = gr.Button("Analyze Candidates")

    result_table = gr.Dataframe(
        label="Candidate Ranking"
    )

    best_candidate = gr.Textbox(
        label="Best Candidate"
    )

    chart = gr.Plot(
        label="Ranking Visualization"
    )

    analyze_btn.click(
        fn=rank_resumes,
        inputs=[resumes, job_desc],
        outputs=[result_table, best_candidate, chart]
    )


demo.launch()

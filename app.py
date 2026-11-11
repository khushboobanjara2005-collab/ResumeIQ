"""ResumeIQ: a session-based resume keyword analyzer built with Streamlit."""

from __future__ import annotations

from datetime import datetime
from html import escape

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from modules.job_matching import (
    compare_resume_to_job,
    comparison_input_diagnostics,
    comparison_inputs_match,
)
from modules.keyword_extraction import extract_keywords
from modules.reporting import build_results_csv
from modules.comparison_storage import (
    delete_comparison,
    get_comparison,
    initialize_database,
    list_comparisons,
    save_comparison,
    set_shortlisted,
)
from modules.text_preprocessing import extract_text_from_upload


NAVIGATION = ("Home", "Analyze Resume", "Job Match", "Results", "History", "About")

st.set_page_config(
    page_title="ResumeIQ — Resume Keyword Analyzer",
    page_icon="🪻",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --background: #F8F6FF;
        --sidebar: #EEE8FF;
        --primary: #8B5CF6;
        --hover-purple: #E3D7FF;
        --active-purple: #DDD0FF;
        --light-purple: #EDE5FF;
        --main-text: #21194D;
        --secondary-text: #655B85;
        --card: #FFFFFF;
        --border: #E6DDF8;
    }
    .stApp { background: var(--background); color: var(--main-text); }
    [data-testid="stHeader"] { background: var(--background); }
    [data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] hr { border-color: var(--border); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: var(--secondary-text);
    }
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }
    .brand {
        color: var(--main-text);
        font-size: 1.55rem;
        font-weight: 750;
    }
    .sidebar-subtitle, .sidebar-footer {
        color: var(--secondary-text);
        font-size: .88rem;
    }
    .subtitle {
        color: var(--secondary-text);
        margin-top: -.65rem;
        font-size: 1.08rem;
    }
    h1, h2, h3, h4, p, label { color: var(--main-text); }
    h1 { letter-spacing: -.025em; }
    div[data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-top: 3px solid #CBB8F5;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 3px 12px rgba(33, 25, 77, .045);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: var(--secondary-text);
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--main-text);
    }
    div.stButton > button, div.stDownloadButton > button {
        min-height: 2.65rem;
        border-radius: 10px;
        border: 1px solid var(--primary);
        color: #fff;
        background: var(--primary);
        font-weight: 650;
        padding: .55rem 1rem;
        transition: background-color .15s ease, border-color .15s ease;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background: #7545DF;
        border-color: #7545DF;
        color: #fff;
    }
    button:focus-visible, a:focus-visible, input:focus-visible,
    textarea:focus-visible, [role="radio"]:focus-visible {
        outline: 3px solid #8B5CF6 !important;
        outline-offset: 2px !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, .18) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
        gap: .35rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input[type="radio"]) {
        position: relative;
        display: flex;
        align-items: center;
        width: 100%;
        min-height: 2.7rem;
        padding: .45rem .75rem;
        margin: 0;
        border: 1px solid transparent;
        border-radius: 10px;
        color: var(--main-text);
        cursor: pointer;
        transition: background-color .15s ease, border-color .15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label > div > div:first-child {
        display: none;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label > div {
        display: flex;
        align-items: center;
        width: 100%;
        padding-left: .15rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input[type="radio"]):hover {
        background: var(--hover-purple);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input[type="radio"]:focus-visible) {
        outline: 3px solid var(--primary);
        outline-offset: 2px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input[type="radio"]:checked) {
        background: var(--active-purple);
        border-color: #CBB8F5;
        box-shadow: inset 3px 0 0 var(--primary);
        color: var(--main-text);
        font-weight: 650;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
        position: absolute;
        width: 1px;
        height: 1px;
        margin: 0;
        opacity: 0;
        overflow: hidden;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        margin: 0;
        color: var(--main-text);
    }
    textarea, input[type="text"] {
        color: var(--main-text) !important;
        background: var(--card) !important;
        border: 1px solid #CDBFEA !important;
        border-radius: 12px !important;
    }
    textarea {
        padding: .8rem .9rem !important;
        line-height: 1.55 !important;
    }
    textarea::placeholder, input::placeholder {
        color: #746A91 !important;
        opacity: 1 !important;
    }
    textarea:focus, input[type="text"]:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, .16) !important;
    }
    [data-testid="stFileUploader"] section {
        background: var(--card);
        border: 1px dashed #BCA9E6;
        border-radius: 12px;
    }
    [data-testid="stAlert"] {
        background: #F0EBFF;
        border: 1px solid var(--border);
        border-radius: 12px;
        color: var(--main-text);
    }
    [data-testid="stAlert"] p { color: var(--main-text); }
    [data-testid="stAlert"] svg { fill: var(--primary); }
    [data-testid="stExpander"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
    }
    .keyword-chip {
        display: inline-block;
        background: var(--light-purple);
        color: var(--main-text);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: .38rem .75rem;
        margin: .22rem .24rem .22rem 0;
        font-size: .9rem;
    }
    [data-testid="stExpander"] [data-testid="stText"] span {
        color: #21194D !important;
        background: #FFFFFF;
        border: 1px solid #E6DDF8;
        border-radius: 10px;
        padding: 1rem;
        margin: 0;
        font-family: inherit;
        font-size: .95rem;
        line-height: 1.55;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }
    .info-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.15rem;
        margin: .45rem 0;
    }
    [data-testid="stVerticalBlock"]:has(> [data-testid="stLayoutWrapper"]) {
        background: var(--card);
        border-color: var(--border) !important;
        border-radius: 16px !important;
        padding: .65rem .8rem;
        box-shadow: 0 2px 9px rgba(33, 25, 77, .025);
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--secondary-text);
    }
    @media (max-width: 700px) {
        .block-container { padding: 1.2rem 1rem 2rem; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input[type="radio"]) {
            min-height: 2.9rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    """Set up session-only state without replacing existing analysis data."""
    defaults = {
        "navigation": "Home",
        "resume_analysis": None,
        "match_result": None,
        "match_inputs_changed": False,
        "analysis_history": [],
        "job_description": "",
        "job_title": "",
        "batch_comparisons": None,
        "selected_history_id": None,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def navigate_to(page: str) -> None:
    """Move to a valid navigation tab on the next Streamlit rerun."""
    if page in NAVIGATION:
        st.session_state["navigation"] = page


def clear_match_result() -> None:
    """Prevent a previous comparison from appearing current after input edits."""
    st.session_state["match_inputs_changed"] = (
        st.session_state["match_result"] is not None
        or st.session_state["match_inputs_changed"]
    )
    st.session_state["match_result"] = None
    st.session_state["batch_comparisons"] = None


def save_job_description() -> None:
    """Keep the text-area value when its page is not currently rendered."""
    st.session_state["job_description"] = st.session_state[
        "_job_description_input"
    ]
    clear_match_result()


def save_job_title() -> None:
    """Keep the optional job title when its page is not currently rendered."""
    st.session_state["job_title"] = st.session_state["_job_title_input"]
    clear_match_result()


def clear_batch_comparisons() -> None:
    """Avoid showing batch results after the uploaded resumes change."""
    st.session_state["batch_comparisons"] = None


def update_shortlist(comparison_id: int, widget_key: str) -> None:
    """Save a shortlist checkbox change to SQLite."""
    set_shortlisted(comparison_id, st.session_state[widget_key])


def get_current_match_result(resume_text: str, job_description: str) -> dict | None:
    """Return a saved comparison only while its exact inputs remain current."""
    result = st.session_state["match_result"]
    if result is not None and not comparison_inputs_match(
        result, resume_text, job_description.strip()
    ):
        st.session_state["match_inputs_changed"] = True
        st.session_state["match_result"] = None
        return None
    return result


def show_keywords(keywords: list[str], empty_message: str = "No keywords to show yet.") -> None:
    """Render safe, readable keyword chips."""
    if not keywords:
        st.caption(empty_message)
        return
    chips = "".join(
        f'<span class="keyword-chip">{escape(keyword)}</span>' for keyword in keywords
    )
    st.markdown(chips, unsafe_allow_html=True)


def render_sidebar() -> str:
    """Render navigation items and return the active page."""
    with st.sidebar:
        st.markdown('<div class="brand">ResumeIQ</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-subtitle">Resume Keyword Analyzer</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        selected_page = st.radio(
            "Navigation",
            NAVIGATION,
            key="navigation",
            label_visibility="collapsed",
            on_change=clear_batch_comparisons,
        )
        st.divider()
        st.markdown(
            '<div class="sidebar-footer">Rule-Based NLP Resume Analysis</div>',
            unsafe_allow_html=True,
        )
    return selected_page


def render_home() -> None:
    st.title("ResumeIQ")
    st.markdown(
        '<p class="subtitle">Analyze your resume. Discover your strengths.</p>',
        unsafe_allow_html=True,
    )

    history = st.session_state["analysis_history"]
    latest_match = st.session_state["match_result"]
    analyzed_count = sum(
        item["type"] == "Resume analyzed" for item in history
    )
    keyword_count = (
        len(st.session_state["resume_analysis"]["keywords"])
        if st.session_state["resume_analysis"]
        else 0
    )
    last_score = (
        f'{latest_match["keyword_coverage"]:.1f}%'
        if latest_match is not None and latest_match["keyword_coverage"] is not None
        else "Not available"
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Resumes Analyzed", analyzed_count)
    metric_columns[1].metric("Keywords Extracted", keyword_count)
    metric_columns[2].metric("Last Match Score", last_score)

    st.write("")
    st.button(
        "Analyze New Resume",
        type="primary",
        on_click=navigate_to,
        args=("Analyze Resume",),
    )

    st.subheader("Recent Activity")
    if history:
        for event in reversed(history[-5:]):
            st.markdown(
                f'<div class="info-card"><strong>{escape(event["type"])}</strong>'
                f'<br><small>{escape(event["time"])}</small></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="info-card" style="background:#EDE5FF;color:#655B85;">'
            "Your recent analyses will appear here.</div>",
            unsafe_allow_html=True,
        )


def render_analyze_resume() -> None:
    st.title("Analyze Resume")
    st.write("Choose to paste resume text or upload a `.txt` or `.pdf` file.")
    st.caption("Files are analyzed in this session only and are not sent to an external service.")

    input_method = st.radio(
        "Resume input method",
        ("Paste text", "Upload file"),
        horizontal=True,
        label_visibility="collapsed",
    )
    uploaded_file = None
    resume_text = ""
    resume_name = "Pasted resume"
    if input_method == "Paste text":
        resume_text = st.text_area(
            "Resume text",
            height=220,
            placeholder="Paste the resume text you want to analyze...",
            key="resume_text_input",
        )
    else:
        uploaded_file = st.file_uploader(
            "Choose a resume file",
            type=["txt", "pdf"],
            help="TXT or PDF, up to 5 MB; PDFs may contain up to 10 pages.",
            key="resume_file_input",
        )

    if st.button("Analyze Resume", type="primary"):
        try:
            if input_method == "Paste text" and resume_text.strip():
                source_text = resume_text.strip()
            elif input_method == "Upload file" and uploaded_file is not None:
                resume_name = uploaded_file.name
                source_text = extract_text_from_upload(
                    uploaded_file.name, uploaded_file.getvalue()
                )
            else:
                st.error("Please upload a resume or paste resume text before analyzing.")
                return

            if not source_text.strip():
                st.error("No readable text was found. Please check the resume and try again.")
                return

            keyword_frame = extract_keywords(source_text)
            keywords = keyword_frame["keyword"].tolist()
            if not keywords:
                st.warning(
                    "The text was analyzed, but no meaningful keywords were found. "
                    "Try adding more resume detail."
                )

            analyzed_at = datetime.now().strftime("%b %d, %Y at %I:%M %p")
            if st.session_state["match_result"] is not None:
                st.session_state["match_inputs_changed"] = True
            st.session_state["resume_analysis"] = {
                "original_text": source_text,
                "resume_name": resume_name,
                "keywords": keywords,
                "keyword_scores": keyword_frame.to_dict(orient="records"),
                "analyzed_at": analyzed_at,
            }
            st.session_state["match_result"] = None
            st.session_state["analysis_history"].append(
                {"type": "Resume analyzed", "time": analyzed_at}
            )
            st.success("Resume analysis complete.")
        except ValueError as error:
            st.error(str(error))

    analysis = st.session_state["resume_analysis"]
    if analysis:
        st.subheader("Extracted Keywords")
        st.metric("Total keywords extracted", len(analysis["keywords"]))
        show_keywords(analysis["keywords"], "No keywords were extracted from this resume.")
        with st.expander("Preview analyzed resume text"):
            st.text(analysis["original_text"])


def render_job_match() -> None:
    st.title("Job Match")
    analysis = st.session_state["resume_analysis"]
    if "_job_description_input" not in st.session_state:
        st.session_state["_job_description_input"] = st.session_state[
            "job_description"
        ]
    if "_job_title_input" not in st.session_state:
        st.session_state["_job_title_input"] = st.session_state["job_title"]

    st.write("Compare one or more resumes with a job description.")
    job_title = st.text_input(
        "Job title (optional)",
        key="_job_title_input",
        on_change=save_job_title,
        placeholder="For example, Python Developer",
    )
    job_description = st.text_area(
        "Job description",
        height=230,
        placeholder="Paste the job description here...",
        key="_job_description_input",
        on_change=save_job_description,
    )

    if analysis:
        if st.button("Compare Resume", type="primary"):
            if not job_description.strip():
                st.error("Please enter a job description before comparing.")
            else:
                active_resume_text = analysis["original_text"]
                current_job_description = job_description.strip()
                result = compare_resume_to_job(
                    active_resume_text, current_job_description
                )
                result["input_diagnostics"] = comparison_input_diagnostics(
                    result,
                    active_resume_text,
                    current_job_description,
                    analysis["original_text"],
                    job_description,
                )
                result["compared_at"] = datetime.now().strftime(
                    "%b %d, %Y at %I:%M %p"
                )
                result["history_id"] = save_comparison(
                    analysis.get("resume_name", "Analyzed resume"),
                    job_title,
                    result,
                )
                st.session_state["match_result"] = result
                st.session_state["match_inputs_changed"] = False
                st.session_state["analysis_history"].append(
                    {
                        "type": "Job description compared",
                        "time": result["compared_at"],
                    }
                )
                st.success("Comparison complete and saved to history.")

        result = get_current_match_result(
            analysis["original_text"], job_description
        )
        if result:
            diagnostics = result.get("input_diagnostics", {})
            provenance_verified = (
                diagnostics.get("fingerprints_match") is True
                and diagnostics.get("resume_from_active_analysis") is True
                and diagnostics.get("job_from_current_text_area") is True
            )
            if not provenance_verified:
                st.error("Input provenance could not be verified. Compare again.")
            else:
                st.success(
                    "Results correspond to the current analyzed resume and job description."
                )
                _render_current_comparison(result, analysis, job_description)
        elif st.session_state["match_inputs_changed"]:
            st.warning("Inputs changed — compare again.")
        else:
            st.info("Compare the analyzed resume with this job description.")
    else:
        st.info(
            "Analyze a resume first to compare it here, or upload multiple resumes below."
        )
        st.button(
            "Go to Analyze Resume",
            on_click=navigate_to,
            args=("Analyze Resume",),
        )

    st.divider()
    st.subheader("Compare Multiple Resumes")
    st.caption(
        "Each supported PDF or TXT file is compared separately against the job "
        "description above. Resume contents are processed in memory and are not saved."
    )
    uploaded_resumes = st.file_uploader(
        "Choose resume files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        key="batch_resume_files",
        on_change=clear_batch_comparisons,
        help="TXT or PDF, up to 5 MB; PDFs may contain up to 10 pages.",
    )
    if st.button("Compare All Resumes", key="compare_all_resumes"):
        if not job_description.strip():
            st.error("Please enter a job description before comparing resumes.")
        elif not uploaded_resumes:
            st.error("Choose at least one PDF or TXT resume to compare.")
        else:
            batch_comparisons = []
            for uploaded_resume in uploaded_resumes:
                try:
                    resume_text = extract_text_from_upload(
                        uploaded_resume.name, uploaded_resume.getvalue()
                    )
                    if not resume_text.strip():
                        raise ValueError(
                            "No readable text was found in this file."
                        )
                    comparison = compare_resume_to_job(
                        resume_text, job_description.strip()
                    )
                    history_id = save_comparison(
                        uploaded_resume.name, job_title, comparison
                    )
                    batch_comparisons.append(
                        {
                            "resume_name": uploaded_resume.name,
                            "result": comparison,
                            "history_id": history_id,
                        }
                    )
                except ValueError as error:
                    message = f"{uploaded_resume.name}: {error}"
                    if (
                        "larger than 5 MB" in str(error)
                        or "more than 10 pages" in str(error)
                    ):
                        st.error(message)
                    else:
                        st.warning(message)

            st.session_state["batch_comparisons"] = batch_comparisons
            if batch_comparisons:
                st.success(
                    f"Compared and saved {len(batch_comparisons)} resume(s)."
                )
            else:
                st.error("No valid resumes were available to compare.")

    _render_batch_comparisons()


def _render_current_comparison(
    result: dict, analysis: dict, job_description: str
) -> None:
    """Render the active, provenance-checked comparison."""
    diagnostics = result["input_diagnostics"]
    if diagnostics["inputs_identical"]:
        st.warning(
            "The resume and job description inputs are identical. "
            "Verify the job description before interpreting these results."
        )
    st.caption(
        "Keyword match is the percentage of extracted job-description keywords "
        "also found in the resume. Cosine similarity is a separate text comparison."
    )
    summary_columns = st.columns(3)
    job_keyword_count = len(result["job_keywords"])
    matched_keyword_count = len(result["matched_keywords"])
    st.caption(
        f"Matched {matched_keyword_count} of {job_keyword_count} unique "
        f"job-description keywords. Extraction is limited to "
        f"{result['job_keyword_limit']} keywords."
    )
    summary_columns[0].metric(
        "Keyword Match",
        (
            f"{matched_keyword_count / job_keyword_count * 100:.1f}%"
            if job_keyword_count
            else "Not available"
        ),
    )
    summary_columns[1].metric(
        "Matched Keywords", f"{matched_keyword_count} / {job_keyword_count}"
    )
    summary_columns[2].metric(
        "Cosine Similarity", f'{result["cosine_similarity"]:.1f}%'
    )
    if job_keyword_count == 0:
        st.info(result["message"])

    left_column, right_column = st.columns(2)
    with left_column:
        with st.container(border=True):
            st.subheader("Matched Keywords")
            show_keywords(
                result["matched_keywords"], "No extracted keywords matched."
            )
    with right_column:
        with st.container(border=True):
            st.subheader("Missing Keywords")
            show_keywords(
                result["missing_keywords"], "No missing keywords identified."
            )
    st.caption(
        "Missing keywords are suggestions to review—not proof that you lack those skills."
    )
    st.caption(
        "These transparent text metrics are not an official ATS score or a prediction "
        "of hiring success."
    )
def _render_batch_comparisons() -> None:
    """Show results from the latest multi-resume comparison and shortlist them."""
    batch_comparisons = st.session_state["batch_comparisons"]
    if batch_comparisons is None:
        return
    if not batch_comparisons:
        st.info("There are no valid batch comparison results to display.")
        return

    st.subheader("Multiple Resume Results")
    st.caption(
        "These results belong to the currently loaded batch. Changing job inputs, "
        "uploads, or navigation clears this list; saved entries remain in History."
    )
    table_rows = []
    for item in batch_comparisons:
        result = item["result"]
        table_rows.append(
            {
                "Resume": item["resume_name"],
                "Keyword Match": (
                    f'{result["keyword_coverage"]:.1f}%'
                    if result["keyword_coverage"] is not None
                    else "Not available"
                ),
                "Cosine Similarity": f'{result["cosine_similarity"]:.1f}%',
                "Matched Skills": ", ".join(result["matched_keywords"]),
                "Missing Skills": ", ".join(result["missing_keywords"]),
            }
        )
    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)
    for item in batch_comparisons:
        comparison_id = item["history_id"]
        widget_key = f"batch_shortlist_{comparison_id}"
        saved = get_comparison(comparison_id)
        if saved is None:
            st.warning(f"Saved history entry for {item['resume_name']} was removed.")
            continue
        if widget_key not in st.session_state:
            st.session_state[widget_key] = saved["shortlisted"]
        st.checkbox(
            f"Shortlist {item['resume_name']}",
            key=widget_key,
            on_change=update_shortlist,
            args=(comparison_id, widget_key),
        )


def render_history() -> None:
    """Browse, reopen, shortlist, and delete saved comparison records."""
    st.title("Comparison History")
    st.caption(
        "Saved results are historical snapshots, not current comparisons. Resume "
        "text and job-description text are not stored."
    )
    selected_filter = st.radio(
        "Candidates",
        ("All comparisons", "Shortlisted only"),
        horizontal=True,
        key="history_filter",
    )
    entries = list_comparisons(shortlisted_only=selected_filter == "Shortlisted only")
    if not entries:
        st.info(
            "No shortlisted comparisons yet."
            if selected_filter == "Shortlisted only"
            else "Completed comparisons will appear here."
        )
        return

    entry_ids = {entry["id"] for entry in entries}
    if st.session_state["selected_history_id"] not in entry_ids:
        st.session_state["selected_history_id"] = None

    for entry in entries:
        title = entry["job_title"] or "Job title not provided"
        st.markdown(
            f"**{escape(entry['resume_name'])}** · {escape(title)} · "
            f"{escape(entry['compared_at'])}"
        )
        columns = st.columns([1.3, 1, 1, 1])
        columns[0].metric(
            "Keyword Match",
            (
                f"{entry['keyword_match_percentage']:.1f}%"
                if entry["keyword_match_percentage"] is not None
                else "Not available"
            ),
        )
        columns[1].metric("Cosine Similarity", f"{entry['cosine_similarity']:.1f}%")
        columns[2].metric("Matched", len(entry["matched_keywords"]))
        columns[3].metric("Missing", len(entry["missing_keywords"]))
        action_columns = st.columns([1, 1, 4])
        if action_columns[0].button(
            "Open results", key=f"open_history_{entry['id']}"
        ):
            st.session_state["selected_history_id"] = entry["id"]
        if action_columns[1].button(
            "Delete", key=f"delete_history_{entry['id']}"
        ):
            delete_comparison(entry["id"])
            if st.session_state["selected_history_id"] == entry["id"]:
                st.session_state["selected_history_id"] = None
            st.rerun()

        widget_key = f"history_shortlist_{entry['id']}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = entry["shortlisted"]
        action_columns[2].checkbox(
            "Shortlist",
            key=widget_key,
            on_change=update_shortlist,
            args=(entry["id"], widget_key),
        )
        st.divider()

    selected_id = st.session_state["selected_history_id"]
    if selected_id is not None:
        selected = get_comparison(selected_id)
        if selected is not None:
            st.subheader("Saved Historical Results")
            st.warning(
                "Historical snapshot — this result is not recalculated against "
                "the current resume or job description."
            )
            st.write(
                f"Resume: **{escape(selected['resume_name'])}** · "
                f"Job title: **{escape(selected['job_title'] or 'Not provided')}** · "
                f"Compared: **{escape(selected['compared_at'])}**"
            )
            historical_columns = st.columns(2)
            historical_columns[0].metric(
                "Keyword Match",
                (
                    f"{selected['keyword_match_percentage']:.1f}%"
                    if selected["keyword_match_percentage"] is not None
                    else "Not available"
                ),
            )
            historical_columns[1].metric(
                "Cosine Similarity", f"{selected['cosine_similarity']:.1f}%"
            )
            keyword_columns = st.columns(2)
            with keyword_columns[0]:
                st.markdown("**Matched Keywords**")
                show_keywords(selected["matched_keywords"], "None")
            with keyword_columns[1]:
                st.markdown("**Missing Keywords**")
                show_keywords(selected["missing_keywords"], "None")


def render_results() -> None:
    st.title("Results")
    analysis = st.session_state["resume_analysis"]
    if not analysis:
        st.info("Your results will appear here after you analyze a resume.")
        return

    match_result = get_current_match_result(
        analysis["original_text"],
        st.session_state["job_description"],
    )
    st.success(f'Latest resume analysis: Complete · {analysis["analyzed_at"]}')
    matched_count = len(match_result["matched_keywords"]) if match_result else None
    missing_count = len(match_result["missing_keywords"]) if match_result else None
    coverage = (
        f'{match_result["keyword_coverage"]:.1f}%'
        if match_result and match_result["keyword_coverage"] is not None
        else "Not available"
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Keywords Extracted", len(analysis["keywords"]))
    metric_columns[1].metric(
        "Matched Keywords", matched_count if matched_count is not None else "Not available"
    )
    metric_columns[2].metric(
        "Missing Keywords", missing_count if missing_count is not None else "Not available"
    )
    metric_columns[3].metric("Keyword Match", coverage)

    if match_result and match_result["job_keywords"]:
        st.subheader("Keyword Comparison")
        st.caption("Missing keywords are suggestions, not proof that you lack those skills.")
        left_column, right_column = st.columns(2)
        with left_column:
            with st.container(border=True):
                st.markdown("**Matched Keywords**")
                show_keywords(match_result["matched_keywords"])
        with right_column:
            with st.container(border=True):
                st.markdown("**Missing Keywords**")
                show_keywords(match_result["missing_keywords"])

        chart_frame = pd.DataFrame(
            {
                "Category": ["Matched", "Missing"],
                "Keywords": [matched_count, missing_count],
            }
        )
        figure, axis = plt.subplots(figsize=(6, 3.2))
        axis.bar(chart_frame["Category"], chart_frame["Keywords"], color=["#8B5CF6", "#C9B8F5"])
        axis.set_ylabel("Number of keywords")
        axis.set_title("Job description keyword coverage")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylim(bottom=0)
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
    elif match_result:
        st.info(match_result["message"])
    elif st.session_state["match_inputs_changed"]:
        st.warning("Inputs changed — compare again.")
    else:
        st.info("Compare a job description to see matched and missing keyword results.")

    st.download_button(
        "Download Results",
        data=build_results_csv(analysis, match_result),
        file_name="resumeiq_results.csv",
        mime="text/csv",
    )


def render_about() -> None:
    st.title("About ResumeIQ")
    with st.container(border=True):
        st.subheader("Project Overview")
        st.write(
            "ResumeIQ — Resume Keyword Analyzer is a Python-based application "
            "that analyzes resume text, extracts important keywords, and compares "
            "them with job descriptions using basic Natural Language Processing."
        )

    with st.container(border=True):
        st.subheader("Project Objectives")
        st.markdown(
            "- Understand text preprocessing\n"
            "- Implement tokenization\n"
            "- Apply stopword removal\n"
            "- Extract relevant keywords\n"
            "- Compare resume and job-description text\n"
            "- Visualize analysis results"
        )

    with st.container(border=True):
        st.subheader("NLP Techniques")
        st.markdown(
            "- **Text preprocessing:** lowercases text, normalizes whitespace, and removes "
            "unnecessary punctuation while preserving technical terms.\n"
            "- **Tokenization:** splits text into smaller units, such as words and technical "
            "terms.\n"
            "- **Stopword removal:** filters common English words that add little meaning.\n"
            "- **TF-IDF:** ranks terms based on their frequency and how distinctive they are "
            "across sentence-like text segments.\n"
            "- **Keyword matching:** compares normalized extracted job-description keywords "
            "with extracted resume keywords.\n"
            "- **Cosine similarity:** compares the direction of the two TF-IDF text vectors; "
            "it is separate from keyword coverage."
        )

    with st.container(border=True):
        st.subheader("Technologies Used")
        st.write("Python · Streamlit · NLTK · scikit-learn · Pandas · Matplotlib · PyMuPDF")

    with st.container(border=True):
        st.subheader("Limitations and Future Improvements")
        st.write(
            "ResumeIQ uses simple text matching and cannot understand context, synonyms, "
            "skill proficiency, or whether a claimed skill is accurate. PDF extraction "
            "depends on selectable text, so scanned documents may need OCR. Possible "
            "improvements include synonym-aware matching, multilingual support, and more "
            "robust PDF/OCR handling. Its keyword results are not an official ATS "
            "evaluation or a prediction of hiring success."
        )


def main() -> None:
    initialize_database()
    initialize_state()
    page = render_sidebar()
    pages = {
        "Home": render_home,
        "Analyze Resume": render_analyze_resume,
        "Job Match": render_job_match,
        "Results": render_results,
        "History": render_history,
        "About": render_about,
    }
    pages[page]()


if __name__ == "__main__":
    main()

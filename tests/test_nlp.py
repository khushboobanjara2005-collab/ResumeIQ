"""Small unit tests for ResumeIQ's local NLP workflow."""

import io
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch
from pathlib import Path

import pandas as pd
import pymupdf as fitz

from modules import job_matching
from modules import comparison_storage
from modules.job_matching import (
    comparison_input_diagnostics,
    compare_resume_to_job,
    comparison_inputs_match,
)
from modules.keyword_extraction import extract_keywords
from modules.reporting import build_results_csv
from modules.text_preprocessing import (
    MAX_PDF_PAGES,
    MAX_UPLOAD_SIZE_BYTES,
    extract_text_from_upload,
    normalize_text,
    preprocess_text,
)


class TextPreprocessingTests(unittest.TestCase):
    def test_normalization_and_stopword_removal(self):
        self.assertEqual(normalize_text("  DATA,   Analysis! "), "data analysis")
        self.assertEqual(preprocess_text("The data is useful"), ["data", "useful"])

    def test_upload_validation_and_text_decoding(self):
        self.assertEqual(extract_text_from_upload("resume.txt", b"Hello"), "Hello")
        with self.assertRaises(ValueError):
            extract_text_from_upload("resume.docx", b"content")
        with self.assertRaises(ValueError):
            extract_text_from_upload("resume.txt", b"\xff")

    def test_pdf_extraction_and_invalid_pdf_handling(self):
        document = fitz.open()
        document.new_page().insert_text((72, 72), "Python resume")
        pdf_bytes = document.tobytes()
        document.close()

        self.assertIn("Python resume", extract_text_from_upload("resume.pdf", pdf_bytes))
        with self.assertRaises(ValueError):
            extract_text_from_upload("resume.pdf", b"not a valid PDF")

    def test_upload_size_and_pdf_page_limits_are_enforced(self):
        oversized_bytes = b"x" * (MAX_UPLOAD_SIZE_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "larger than 5 MB"):
            extract_text_from_upload("resume.txt", oversized_bytes)

        document = fitz.open()
        for _ in range(MAX_PDF_PAGES + 1):
            document.new_page()
        pdf_bytes = document.tobytes()
        document.close()
        with self.assertRaisesRegex(ValueError, "more than 10 pages"):
            extract_text_from_upload("resume.pdf", pdf_bytes)


class KeywordExtractionTests(unittest.TestCase):
    def test_compound_terms_stay_single_keywords(self):
        keywords = extract_keywords(
            "CI/CD, Spring Boot, Node.js, React.js, Next.js, Vue.js, "
            "ASP.NET, Power BI, REST APIs, and machine learning."
        )["keyword"].tolist()

        for compound in (
            "ci/cd",
            "spring boot",
            "node",
            "react",
            "next.js",
            "vue",
            "asp.net",
            "power bi",
            "rest apis",
            "machine learning",
        ):
            self.assertIn(compound, keywords)
        self.assertNotIn("ci", keywords)
        self.assertNotIn("cd", keywords)
        self.assertNotIn("spring", keywords)
        self.assertNotIn("boot", keywords)

    def test_extracts_supported_multiword_phrases(self):
        result = extract_keywords(
            "Built a machine learning system for data analysis and web development."
        )
        self.assertIn("machine learning", result["keyword"].tolist())
        self.assertIn("data analysis", result["keyword"].tolist())
        self.assertIn("web development", result["keyword"].tolist())

    def test_empty_and_stopword_only_inputs_return_no_keywords(self):
        self.assertTrue(extract_keywords("").empty)
        self.assertTrue(extract_keywords("the and is to").empty)

    def test_configured_stopwords_do_not_remove_supported_skill_phrases(self):
        result = extract_keywords(
            "An undergraduate studying computer applications "
            "and coursework. Skills include computer vision and Python."
        )
        keywords = result["keyword"].tolist()

        for generic_word in (
            "computer",
            "applications",
            "coursework",
            "undergraduate",
        ):
            self.assertNotIn(generic_word, keywords)
        self.assertIn("computer vision", keywords)
        self.assertIn("python", keywords)

    def test_generic_actions_and_contact_metadata_are_not_extracted_as_skills(self):
        resume = (
            "Jordan Lee | jordan@example.com | (555) 123-4567\n"
            "Created and analyzed reports for a personal portfolio.\n"
            "Python SQL Pandas NumPy Excel data analysis database management."
        )
        keywords = extract_keywords(resume)["keyword"].tolist()

        for noise in (
            "jordan",
            "lee",
            "email",
            "personal",
            "created",
            "analyzed",
            "reports",
        ):
            self.assertNotIn(noise, keywords)
        for skill in (
            "python",
            "sql",
            "pandas",
            "numpy",
            "excel",
            "data analysis",
            "database management",
        ):
            self.assertIn(skill, keywords)

    def test_generic_job_verbs_are_filtered_but_skill_phrases_remain(self):
        text = (
            "Creating and generating reports, analyzing data, and important tasks. "
            "Python SQL support data analysis, machine learning, responsive design, "
            "and communication skills."
        )
        keywords = set(extract_keywords(text)["keyword"].tolist())

        self.assertTrue(
            {
                "creating",
                "create",
                "generating",
                "generate",
                "analyze",
                "analyzing",
                "analyst",
                "important",
                "tasks",
                "support",
            }.isdisjoint(keywords)
        )
        self.assertTrue(
            {
                "python",
                "sql",
                "data analysis",
                "machine learning",
                "responsive design",
                "communication skills",
            }.issubset(keywords)
        )

    def test_filters_generic_job_words_and_keeps_technical_terms(self):
        text = (
            "Junior developer with strong preferred skills in Python, SQL, CSS, "
            "Pandas, NumPy, scikit-learn, Git, GitHub, TF-IDF, machine learning, "
            "data analysis, and natural language processing."
        )
        keywords = set(extract_keywords(text)["keyword"].tolist())

        self.assertTrue(
            {
                "python",
                "sql",
                "css",
                "pandas",
                "numpy",
                "scikit-learn",
                "git",
                "github",
                "tf-idf",
                "machine learning",
                "data analysis",
                "natural language processing",
            }.issubset(keywords)
        )
        self.assertTrue(
            {"junior", "developer", "strong", "preferred"}.isdisjoint(keywords)
        )

    def test_filters_need_and_skilled_but_keeps_meaningful_phrases_and_tools(self):
        keywords = extract_keywords(
            "We need a skilled Python developer with SQL, data analysis, "
            "problem solving, and communication skills."
        )["keyword"].tolist()

        self.assertTrue(
            {
                "python",
                "sql",
                "data analysis",
                "problem solving",
                "communication skills",
            }.issubset(keywords)
        )
        self.assertTrue(
            {"need", "needs", "needed", "skilled", "skillful", "communication"}.isdisjoint(
                keywords
            )
        )

    def test_python_sql_job_filters_generic_terms_without_losing_skills(self):
        keywords = extract_keywords(
            "Junior Python Developer with SQL. We need a skilled person "
            "with data analysis skills."
        )["keyword"].tolist()

        self.assertEqual(set(keywords), {"python", "sql", "data analysis"})

    def test_front_end_job_keeps_phrases_and_omits_hyphen_fragments(self):
        resume = (
            "Built responsive design and web development projects using HTML and CSS."
        )
        job_description = (
            "Front-End Developer: We need a skilled developer with front-end, "
            "responsive design, web development, communication skills, and "
            "entry-level experience. It requires building responsive pages; "
            "desirable skills include HTML and CSS."
        )

        keywords = extract_keywords(job_description)["keyword"].tolist()
        result = compare_resume_to_job(resume, job_description)

        self.assertIn("front end", keywords)
        self.assertIn("communication skills", keywords)
        self.assertIn("responsive design", keywords)
        self.assertIn("web development", keywords)
        self.assertNotIn("end", keywords)
        self.assertNotIn("front", keywords)
        self.assertNotIn("entry", keywords)
        self.assertNotIn("level", keywords)
        self.assertTrue(
            {
                "need",
                "skilled",
                "building",
                "requires",
                "desirable",
            }.isdisjoint(result["missing_keywords"])
        )
        self.assertFalse(
            set(result["matched_keywords"]) & set(result["missing_keywords"])
        )

    def test_keywords_are_unique_across_case_and_repetition(self):
        keywords = extract_keywords("Python python PYTHON and SQL, sql.")["keyword"].tolist()
        self.assertEqual(keywords.count("python"), 1)
        self.assertEqual(keywords.count("sql"), 1)

    def test_filters_desirable_without_dropping_requested_tools(self):
        text = (
            "Desirable experience with HTML, CSS, Streamlit, GitHub, Python, SQL, "
            "Pandas, and scikit-learn."
        )
        keywords = set(extract_keywords(text)["keyword"].tolist())
        self.assertNotIn("desirable", keywords)
        self.assertTrue(
            {
                "html",
                "css",
                "streamlit",
                "github",
                "python",
                "sql",
                "pandas",
                "scikit-learn",
            }.issubset(keywords)
        )
        self.assertIn(
            "communication skills",
            extract_keywords("Desirable communication skills. Python and SQL.")[
                "keyword"
            ].tolist(),
        )

    def test_filters_standalone_text_but_keeps_meaningful_phrases(self):
        generic_terms = extract_keywords(
            "Text is desirable. Python is required."
        )["keyword"].tolist()
        phrase_terms = extract_keywords(
            "Text preprocessing, natural language processing, and machine learning."
        )["keyword"].tolist()

        self.assertNotIn("text", generic_terms)
        self.assertIn("text preprocessing", phrase_terms)
        self.assertIn("natural language processing", phrase_terms)
        self.assertIn("machine learning", phrase_terms)

    def test_extracts_tools_across_sections_and_preserves_technical_spellings(self):
        resume = """
        Front-end: HTML, CSS
        Frameworks: Streamlit
        Collaboration: GitHub
        Data tools: SQL, Pandas, NumPy, scikit-learn
        Methods: TF-IDF, C++, C#, .NET
        """
        keywords = set(extract_keywords(resume)["keyword"].tolist())
        expected = {
            "html",
            "css",
            "streamlit",
            "github",
            "sql",
            "pandas",
            "numpy",
            "scikit-learn",
            "tf-idf",
            "c++",
            "c#",
            ".net",
        }

        self.assertTrue(expected.issubset(keywords))

    def test_front_end_test_resume_extracts_skills_not_resume_metadata(self):
        source_text = """Jordan Lee
jordan.lee@example.com | (555) 010-1234
SUMMARY
Front-End Developer
TECHNICAL SKILLS
HTML, CSS, JavaScript, React, Git, GitHub
EXPERIENCE
Built responsive web layouts. Front-end development and software development projects used Python, SQL, and NLP.
EDUCATION
Technical Skills"""

        displayed_skills = extract_keywords(source_text)["keyword"].tolist()

        self.assertTrue(
            {
                "html",
                "css",
                "javascript",
                "react",
                "git",
                "github",
                "front end",
                "front end development",
                "responsive web layouts",
                "software development",
                "python",
                "sql",
                "natural language processing",
            }.issubset(displayed_skills)
        )
        self.assertTrue(
            {
                "summary",
                "education",
                "experience",
                "technical",
                "skills",
                "project",
                "jordan",
                "lee",
                "jordan lee",
                "example.com",
                "02",
                "file",
                "used",
            }.isdisjoint(displayed_skills)
        )
        self.assertEqual(len(displayed_skills), len(set(displayed_skills)))
        self.assertIn("Jordan Lee", source_text)
        self.assertIn("jordan.lee@example.com", source_text)
        self.assertIn("Technical Skills", source_text)

    def test_test_file_02_phrases_are_preserved_without_standalone_web(self):
        resume = """JavaScript, React, HTML, CSS, Git, and responsive web layouts.
Interested in creating accessible, interactive web interfaces.
Communication skills."""
        job_description = (
            "Front-End Developer with responsive design, accessible web interfaces, "
            "and communication skills."
        )

        resume_keywords = extract_keywords(resume)["keyword"].tolist()
        job_keywords = extract_keywords(job_description)["keyword"].tolist()
        result = compare_resume_to_job(resume, job_description)

        self.assertIn("responsive web layouts", resume_keywords)
        self.assertIn("web interfaces", resume_keywords)
        self.assertIn("communication skills", resume_keywords)
        self.assertIn("responsive design", job_keywords)
        self.assertIn("web interfaces", job_keywords)
        self.assertIn("communication skills", job_keywords)
        self.assertNotIn("web", resume_keywords)
        self.assertNotIn("web", job_keywords)
        self.assertNotIn("accessible", resume_keywords)
        self.assertNotIn("interactive", resume_keywords)
        self.assertNotIn("interfaces", resume_keywords)
        self.assertIn("web interfaces", result["matched_keywords"])
        self.assertIn("communication skills", result["matched_keywords"])
        self.assertIn("responsive design", result["missing_keywords"])
        self.assertNotIn("web", result["missing_keywords"])
        self.assertFalse(
            set(result["matched_keywords"]) & set(result["missing_keywords"])
        )
        self.assertEqual(
            len(result["matched_keywords"]) + len(result["missing_keywords"]),
            len(result["job_keywords"]),
        )

    def test_candidate_name_before_contact_line_is_not_a_resume_skill(self):
        resume = """RESUMEIQ TEST FILE 02
Sample Resume - Front-End Development
Purpose: Test matching with a different skill set.
Jordan Lee
Front-End Developer Intern | jordan.lee@example.test
JavaScript, React, HTML, CSS, Git, and responsive web layouts."""

        keywords = extract_keywords(resume)["keyword"].tolist()

        self.assertNotIn("jordan", keywords)
        self.assertNotIn("lee", keywords)
        self.assertNotIn("jordan lee", keywords)
        self.assertIn("javascript", keywords)
        self.assertIn("web interfaces", extract_keywords(
            "accessible, interactive web interfaces."
        )["keyword"].tolist())


class JobMatchingTests(unittest.TestCase):
    def test_aliases_match_across_resume_and_job_description(self):
        react_result = compare_resume_to_job(
            "Experienced ReactJS developer.", "React.js experience required."
        )
        kubernetes_result = compare_resume_to_job(
            "Managed Kubernetes clusters.", "Experience with k8s required."
        )
        html_ml_result = compare_resume_to_job("HTML development.", "ML skills.")

        self.assertEqual(react_result["job_keywords"], ["react"])
        self.assertEqual(react_result["matched_keywords"], ["react"])
        self.assertEqual(kubernetes_result["job_keywords"], ["kubernetes"])
        self.assertEqual(kubernetes_result["matched_keywords"], ["kubernetes"])
        self.assertEqual(html_ml_result["job_keywords"], ["machine learning"])
        self.assertEqual(html_ml_result["matched_keywords"], [])

    def test_comparison_fingerprints_track_resume_and_job_inputs(self):
        resume = "Python SQL Pandas"
        job_description = "Python SQL NumPy"
        first_result = compare_resume_to_job(resume, job_description)
        second_result = compare_resume_to_job(resume, job_description)

        self.assertEqual(
            first_result["input_fingerprints"],
            second_result["input_fingerprints"],
        )
        self.assertTrue(
            comparison_inputs_match(first_result, resume, job_description)
        )
        self.assertFalse(
            comparison_inputs_match(first_result, resume, "Python SQL Excel")
        )
        self.assertFalse(
            comparison_inputs_match(first_result, "Python SQL Excel", job_description)
        )
        self.assertEqual(
            first_result["keyword_coverage"],
            round(
                len(first_result["matched_keywords"])
                / len(first_result["job_keywords"])
                * 100,
                1,
            ),
        )
        self.assertEqual(first_result["job_keyword_limit"], 25)

    def test_identical_inputs_are_explicitly_detected_in_safe_diagnostics(self):
        text = "Python SQL Pandas data analysis"
        result = compare_resume_to_job(text, text)
        diagnostics = comparison_input_diagnostics(
            result, text, text, text, text
        )

        self.assertEqual(
            diagnostics,
            {
                "inputs_identical": True,
                "fingerprints_match": True,
                "resume_from_active_analysis": True,
                "job_from_current_text_area": True,
            },
        )

    def test_distinct_resume_and_job_inputs_have_correct_matches_and_metrics(self):
        resume = "Python SQL Pandas data analysis"
        job_description = "Python SQL HTML CSS communication skills"
        result = compare_resume_to_job(resume, job_description)
        diagnostics = comparison_input_diagnostics(
            result, resume, job_description, resume, job_description
        )

        self.assertFalse(diagnostics["inputs_identical"])
        self.assertTrue(diagnostics["fingerprints_match"])
        self.assertTrue(diagnostics["resume_from_active_analysis"])
        self.assertTrue(diagnostics["job_from_current_text_area"])
        self.assertEqual(
            result["job_keywords"],
            ["python", "sql", "html", "css", "communication skills"],
        )
        self.assertEqual(result["matched_keywords"], ["python", "sql"])
        self.assertEqual(
            set(result["missing_keywords"]),
            {"html", "css", "communication skills"},
        )
        self.assertEqual(result["keyword_coverage"], 40.0)
        self.assertAlmostEqual(result["cosine_similarity"], 22.6, delta=0.1)

    def test_whitespace_around_job_input_is_normalized_for_provenance(self):
        resume = "Python SQL Pandas"
        current_job_description = "  Python SQL HTML  "
        compared_job_description = current_job_description.strip()
        result = compare_resume_to_job(resume, compared_job_description)

        self.assertTrue(
            comparison_inputs_match(result, resume, current_job_description)
        )
        diagnostics = comparison_input_diagnostics(
            result,
            resume,
            compared_job_description,
            resume,
            current_job_description,
        )
        self.assertTrue(diagnostics["fingerprints_match"])
        self.assertTrue(diagnostics["job_from_current_text_area"])

    def test_distinct_inputs_pass_provenance_validation(self):
        resume = "Python SQL Pandas data analysis"
        job_description = "Python SQL HTML CSS communication skills"
        result = compare_resume_to_job(resume, job_description)
        diagnostics = comparison_input_diagnostics(
            result, resume, job_description, resume, job_description
        )

        self.assertTrue(diagnostics["fingerprints_match"])
        self.assertTrue(
            all(
                diagnostics[key]
                for key in (
                    "resume_from_active_analysis",
                    "job_from_current_text_area",
                )
            )
        )

    def test_changed_inputs_after_comparison_fail_provenance_checks(self):
        resume = "Python SQL Pandas"
        job_description = "Python SQL NumPy"
        result = compare_resume_to_job(resume, job_description)

        self.assertTrue(comparison_inputs_match(result, resume, job_description))
        self.assertFalse(
            comparison_inputs_match(result, resume, "Python SQL HTML")
        )
        self.assertFalse(
            comparison_inputs_match(result, "Python SQL Excel", job_description)
        )

    def test_same_resume_and_job_description_matches_without_duplicates(self):
        text = (
            "Python, SQL, data analysis, problem solving, and communication skills. "
            "Python and SQL."
        )
        result = compare_resume_to_job(text, text)

        self.assertEqual(result["keyword_coverage"], 100.0)
        self.assertEqual(result["cosine_similarity"], 100.0)
        self.assertEqual(result["missing_keywords"], [])
        self.assertEqual(len(result["job_keywords"]), len(set(result["job_keywords"])))

    def test_generic_words_do_not_dilute_partial_keyword_match(self):
        result = compare_resume_to_job(
            "Python and SQL.",
            "We need a skilled developer with Python, SQL, HTML, and communication skills.",
        )

        self.assertEqual(result["matched_keywords"], ["python", "sql"])
        self.assertNotIn("need", result["job_keywords"])
        self.assertNotIn("skilled", result["job_keywords"])
        self.assertNotIn("communication", result["job_keywords"])
        self.assertIn("communication skills", result["job_keywords"])
        self.assertIn("html", result["missing_keywords"])
        self.assertEqual(
            set(result["matched_keywords"]) & set(result["missing_keywords"]),
            set(),
        )

    def test_match_and_nonmatching_terms_are_separated(self):
        result = compare_resume_to_job(
            "Python developer with machine learning experience",
            "Python and machine learning skills with cloud computing",
        )
        self.assertIn("python", result["matched_keywords"])
        self.assertIn("cloud computing", result["missing_keywords"])
        self.assertGreater(result["keyword_coverage"], 0)

    def test_no_overlap_is_zero_and_empty_description_is_rejected(self):
        no_match = compare_resume_to_job("Python developer", "Java database engineer")
        self.assertEqual(no_match["keyword_coverage"], 0.0)
        self.assertEqual(no_match["matched_keywords"], [])

        with self.assertRaisesRegex(ValueError, "Job description must not be empty"):
            compare_resume_to_job("Python developer", "")

    def test_job_match_uses_job_keywords_against_full_resume_text(self):
        result = compare_resume_to_job(
            "SUMMARY\nPython SQL Pandas NumPy Git data visualization.",
            "Required: Python, SQL, Pandas, NumPy, Git, and data visualization.",
        )

        self.assertEqual(
            set(result["matched_keywords"]),
            {"python", "sql", "pandas", "numpy", "git", "data visualization"},
        )
        self.assertEqual(result["missing_keywords"], [])
        self.assertEqual(result["keyword_coverage"], 100.0)

    def test_different_inputs_use_separate_cosine_vectors_and_job_keyword_denominator(self):
        resume = "Python SQL Pandas project portfolio experience"
        job_description = "Python SQL NumPy Excel database management"
        original_cosine = job_matching.cosine_similarity
        captured_vectors = []

        def inspect_vectors(resume_vector, job_vector):
            captured_vectors.extend(
                (resume_vector.toarray(), job_vector.toarray())
            )
            return original_cosine(resume_vector, job_vector)

        with patch.object(job_matching, "cosine_similarity", side_effect=inspect_vectors):
            result = compare_resume_to_job(resume, job_description)

        self.assertEqual(
            set(result["job_keywords"]),
            {"python", "sql", "numpy", "database management", "excel"},
        )
        self.assertEqual(result["matched_keywords"], ["python", "sql"])
        self.assertEqual(
            set(result["missing_keywords"]),
            {"numpy", "database management", "excel"},
        )
        self.assertEqual(result["keyword_coverage"], 40.0)
        self.assertEqual(len(captured_vectors), 2)
        self.assertNotEqual(captured_vectors[0].tolist(), captured_vectors[1].tolist())
        self.assertGreaterEqual(result["cosine_similarity"], 0.0)
        self.assertLess(result["cosine_similarity"], 100.0)

    def test_job_match_can_find_resume_terms_beyond_extraction_limit(self):
        resume = (
            "Python SQL HTML CSS JavaScript React NLP Streamlit GitHub Git Pandas "
            "NumPy scikit-learn TF-IDF C++ C# .NET"
        )
        resume_keywords = extract_keywords(resume)["keyword"].tolist()
        result = compare_resume_to_job(resume, "Experience with .NET.")

        self.assertIn(".net", resume_keywords)
        self.assertEqual(result["matched_keywords"], [".net"])
        self.assertEqual(result["missing_keywords"], [])
        self.assertEqual(result["keyword_coverage"], 100.0)

    def test_job_match_with_no_relevant_keywords_has_unavailable_coverage(self):
        result = compare_resume_to_job(
            "Alex Morgan has undergraduate coursework in computer applications.",
            "Junior developer with preferred coursework and computer applications.",
        )

        self.assertEqual(result["job_keywords"], [])
        self.assertEqual(result["matched_keywords"], [])
        self.assertEqual(result["missing_keywords"], [])
        self.assertIsNone(result["keyword_coverage"])
        self.assertIn("not available", result["message"].lower())

    def test_multiword_phrase_matches_without_missing_component_words(self):
        result = compare_resume_to_job(
            "Experience with problem solving and Python.",
            "Problem solving, problem-solving, and Python are required.",
        )

        self.assertIn("problem solving", result["matched_keywords"])
        self.assertNotIn("problem", result["missing_keywords"])
        self.assertNotIn("solving", result["missing_keywords"])
        self.assertEqual(result["job_keywords"].count("problem solving"), 1)
        self.assertEqual(result["keyword_coverage"], 100.0)

    def test_generic_role_words_do_not_count_as_job_keywords(self):
        result = compare_resume_to_job(
            "Python, SQL, and data analysis.",
            "Junior developer with strong preferred Python and SQL skills.",
        )

        self.assertNotIn("junior", result["job_keywords"])
        self.assertNotIn("developer", result["job_keywords"])
        self.assertNotIn("strong", result["job_keywords"])
        self.assertNotIn("preferred", result["job_keywords"])
        self.assertEqual(result["job_keywords"], ["python", "sql"])
        self.assertEqual(result["keyword_coverage"], 100.0)

    def test_repeated_job_keywords_count_once_for_coverage(self):
        result = compare_resume_to_job(
            "Python developer with SQL.",
            "Python, PYTHON, and SQL; Python skills preferred.",
        )

        self.assertEqual(result["job_keywords"].count("python"), 1)
        self.assertEqual(result["job_keywords"].count("sql"), 1)
        self.assertEqual(result["keyword_coverage"], 100.0)

    def test_full_half_and_zero_keyword_coverage_use_unique_job_keywords(self):
        job = "Python, SQL, Git, GitHub; python preferred."

        full = compare_resume_to_job("Python SQL Git GitHub", job)
        half = compare_resume_to_job("Python SQL", job)
        none = compare_resume_to_job("HTML CSS", job)

        self.assertEqual(
            set(full["job_keywords"]), {"python", "sql", "github", "git"}
        )
        self.assertEqual(full["keyword_coverage"], 100.0)
        self.assertEqual(half["keyword_coverage"], 50.0)
        self.assertEqual(none["keyword_coverage"], 0.0)
        for result in (full, half, none):
            self.assertEqual(
                set(result["matched_keywords"]) & set(result["missing_keywords"]),
                set(),
            )
            self.assertEqual(
                len(result["matched_keywords"]) + len(result["missing_keywords"]),
                len(result["job_keywords"]),
            )

    def test_capitalization_is_normalized_but_git_and_github_are_distinct(self):
        result = compare_resume_to_job("Python and Git.", "python, PYTHON, Git, GitHub.")

        self.assertEqual(result["job_keywords"].count("python"), 1)
        self.assertIn("python", result["matched_keywords"])
        self.assertIn("git", result["matched_keywords"])
        self.assertIn("github", result["missing_keywords"])
        self.assertNotIn("github", result["matched_keywords"])

    def test_cosine_similarity_is_one_hundred_for_identical_documents(self):
        text = "Python SQL Pandas data analysis."
        result = compare_resume_to_job(text, text)

        self.assertEqual(result["cosine_similarity"], 100.0)

    def test_cosine_similarity_uses_same_meaningful_unigram_vocabulary(self):
        resume = "Python SQL Pandas"
        job_description = "Pandas, SQL, and Python."
        result = compare_resume_to_job(resume, job_description)

        self.assertEqual(result["cosine_similarity"], 100.0)

    def test_cosine_similarity_is_zero_for_unrelated_documents(self):
        result = compare_resume_to_job(
            "Python SQL Pandas",
            "Gardening cooking furniture",
        )

        self.assertEqual(result["cosine_similarity"], 0.0)

    def test_cosine_similarity_is_between_zero_and_one_hundred_for_partial_overlap(self):
        result = compare_resume_to_job(
            "Python SQL Pandas",
            "Python SQL machine learning cloud computing",
        )

        self.assertGreater(result["cosine_similarity"], 0.0)
        self.assertLess(result["cosine_similarity"], 100.0)
        self.assertEqual(result["keyword_coverage"], 50.0)
        self.assertNotEqual(result["cosine_similarity"], result["keyword_coverage"])

    def test_phrases_and_technical_terms_normalize_consistently(self):
        result = compare_resume_to_job(
            "Problem-solving, MACHINE LEARNING, and natural language processing. "
            "Python, SQL, Pandas, Streamlit, GitHub, HTML, CSS, scikit learn.",
            "problem solving; machine-learning; Natural Language Processing. "
            "Python, python, SQL, Pandas, Streamlit, GitHub, HTML, CSS, "
            "scikit-learn, and desirable.",
        )

        expected = {
            "problem solving",
            "machine learning",
            "natural language processing",
            "python",
            "sql",
            "pandas",
            "streamlit",
            "github",
            "html",
            "css",
            "scikit-learn",
        }
        self.assertEqual(set(result["job_keywords"]), expected)
        self.assertEqual(set(result["matched_keywords"]), expected)
        self.assertEqual(result["missing_keywords"], [])
        self.assertEqual(result["keyword_coverage"], 100.0)
        self.assertIn("cosine_similarity", result)
        self.assertNotEqual(result["cosine_similarity"], result["keyword_coverage"])

    def test_skills_are_matched_across_sections_and_missing_skill_stays_missing(self):
        resume = """
        Web:
        HTML
        CSS
        Streamlit
        GitHub

        Data:
        SQL
        Pandas
        NumPy
        scikit-learn
        TF-IDF
        """
        job_description = (
            "HTML CSS Streamlit GitHub SQL Pandas NumPy scikit learn TF IDF "
            "and Kubernetes."
        )

        result = compare_resume_to_job(resume, job_description)
        matched = set(result["matched_keywords"])
        missing = set(result["missing_keywords"])

        self.assertTrue(
            {
                "html",
                "css",
                "streamlit",
                "github",
                "sql",
                "pandas",
                "numpy",
                "scikit-learn",
                "tf-idf",
            }.issubset(matched)
        )
        self.assertIn("kubernetes", missing)
        self.assertFalse(matched & missing)

    def test_empty_resume_and_description_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "Resume text must not be empty"):
            compare_resume_to_job("", "HTML, CSS, and Kubernetes are required.")
        with self.assertRaisesRegex(ValueError, "Job description must not be empty"):
            compare_resume_to_job("HTML and CSS", "")
        with self.assertRaisesRegex(ValueError, "Resume text must not be empty"):
            compare_resume_to_job(" \n ", " \t ")

    def test_matcher_excludes_generic_keywords_from_unique_coverage_set(self):
        extracted = pd.DataFrame(
            [
                ("desirable", 1.0),
                ("communication skills", 0.9),
                ("python", 0.8),
                ("python", 0.7),
            ],
            columns=["keyword", "score"],
        )
        with patch("modules.job_matching.extract_keywords", return_value=extracted):
            result = compare_resume_to_job(
                "Python and communication skills.",
                "Python and communication skills.",
            )

        self.assertNotIn("desirable", result["job_keywords"])
        self.assertEqual(
            result["job_keywords"], ["communication skills", "python"]
        )
        self.assertEqual(result["keyword_coverage"], 100.0)
        self.assertIn("cosine_similarity", result)

    def test_standalone_text_is_filtered_but_text_preprocessing_matches(self):
        result = compare_resume_to_job(
            "Text preprocessing experience.",
            "Text and text preprocessing are desirable skills.",
        )

        self.assertNotIn("text", result["job_keywords"])
        self.assertIn("text preprocessing", result["matched_keywords"])
        self.assertNotIn("text preprocessing", result["missing_keywords"])


class ReportingTests(unittest.TestCase):
    def test_csv_contains_actual_resume_and_match_results(self):
        analysis = {
            "keywords": ["python", "machine learning"],
            "keyword_scores": [
                {"keyword": "python", "score": 1.2},
                {"keyword": "machine learning", "score": 0.9},
            ],
        }
        match_result = {
            "matched_keywords": ["python"],
            "missing_keywords": ["cloud computing"],
            "keyword_coverage": 50.0,
            "cosine_similarity": 37.5,
        }

        report = pd.read_csv(io.BytesIO(build_results_csv(analysis, match_result)))
        self.assertIn("machine learning", report["keyword"].tolist())
        self.assertIn("cloud computing", report["keyword"].tolist())
        coverage_row = report.loc[
            report["keyword"] == "keyword_match_percentage", "value"
        ].iloc[0]
        self.assertEqual(float(coverage_row), 50.0)


class ComparisonStorageTests(unittest.TestCase):
    def test_history_persists_metrics_shortlist_and_delete(self):
        resume_text = "Python SQL Pandas data analysis"
        job_description = "Python SQL HTML"
        result = compare_resume_to_job(resume_text, job_description)
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "history.sqlite3"
            comparison_id = comparison_storage.save_comparison(
                "resume.txt", "Python Developer", result, database_path
            )

            reopened = comparison_storage.get_comparison(
                comparison_id, database_path
            )
            self.assertIsNotNone(reopened)
            self.assertEqual(reopened["resume_name"], "resume.txt")
            self.assertEqual(reopened["job_title"], "Python Developer")
            self.assertEqual(
                reopened["keyword_match_percentage"], result["keyword_coverage"]
            )
            self.assertEqual(
                reopened["cosine_similarity"], result["cosine_similarity"]
            )
            self.assertEqual(
                reopened["matched_keywords"], result["matched_keywords"]
            )
            self.assertEqual(
                reopened["missing_keywords"], result["missing_keywords"]
            )

            comparison_storage.set_shortlisted(
                comparison_id, True, database_path
            )
            self.assertEqual(
                [entry["id"] for entry in comparison_storage.list_comparisons(
                    shortlisted_only=True, database_path=database_path
                )],
                [comparison_id],
            )
            comparison_storage.set_shortlisted(
                comparison_id, False, database_path
            )
            self.assertEqual(
                comparison_storage.list_comparisons(
                    shortlisted_only=True, database_path=database_path
                ),
                [],
            )
            comparison_storage.delete_comparison(comparison_id, database_path)
            self.assertIsNone(
                comparison_storage.get_comparison(comparison_id, database_path)
            )

    def test_history_does_not_store_resume_or_job_text(self):
        resume_text = "Private resume text Python SQL"
        job_description = "Confidential job description Python SQL"
        result = compare_resume_to_job(resume_text, job_description)

        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "history.sqlite3"
            comparison_storage.save_comparison(
                "private-resume.pdf", "", result, database_path
            )
            with closing(sqlite3.connect(database_path)) as connection:
                saved_row = repr(
                    connection.execute(
                        "SELECT * FROM comparisons"
                    ).fetchone()
                )
            self.assertNotIn(resume_text, saved_row)
            self.assertNotIn(job_description, saved_row)


class AppNavigationStateTests(unittest.TestCase):
    def test_multiple_resume_comparison_shortlist_and_history_ui(self):
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parents[1] / "app.py"
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"RESUMEIQ_DATABASE": str(Path(directory) / "history.sqlite3")},
        ):
            database_path = Path(directory) / "history.sqlite3"
            app_test = AppTest.from_file(
                str(app_path), default_timeout=20
            ).run(timeout=20)
            app_test.radio[0].set_value("Job Match").run(timeout=20)
            app_test.file_uploader[0].set_value(
                [
                    ("python_resume.txt", b"Python SQL Pandas", "text/plain"),
                    ("web_resume.txt", b"HTML CSS React", "text/plain"),
                    ("invalid.pdf", b"not a real PDF", "application/pdf"),
                ]
            ).run(timeout=20)
            next(
                button
                for button in app_test.button
                if button.label == "Compare All Resumes"
            ).click().run(timeout=20)
            self.assertIn(
                "Please enter a job description before comparing resumes.",
                [error.value for error in app_test.error],
            )
            app_test.text_area[0].set_value(
                "Python SQL HTML CSS communication skills"
            ).run(timeout=20)

            compare_all = next(
                button
                for button in app_test.button
                if button.label == "Compare All Resumes"
            )
            compare_all.click().run(timeout=20)
            self.assertEqual(len(app_test.session_state["batch_comparisons"]), 2)
            self.assertTrue(app_test.warning)
            self.assertEqual(
                len(comparison_storage.list_comparisons(database_path=database_path)),
                2,
            )

            app_test.radio[0].set_value("History").run(timeout=20)
            self.assertIsNone(app_test.session_state["batch_comparisons"])
            app_test.checkbox[0].check().run(timeout=20)
            shortlisted = comparison_storage.list_comparisons(
                shortlisted_only=True, database_path=database_path
            )
            self.assertEqual(len(shortlisted), 1)

            app_test.radio[0].set_value("Shortlisted only").run(timeout=20)
            open_button = next(
                button
                for button in app_test.button
                if button.label == "Open results"
            )
            open_button.click().run(timeout=20)
            self.assertTrue(
                any(
                    "Historical snapshot" in warning.value
                    for warning in app_test.warning
                )
            )

    def test_job_description_and_comparison_survive_navigation(self):
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parents[1] / "app.py"
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"RESUMEIQ_DATABASE": str(Path(directory) / "history.sqlite3")},
        ):
            app_test = AppTest.from_file(
                str(app_path), default_timeout=20
            ).run(timeout=20)
            app_test.session_state["resume_analysis"] = {
                "original_text": "Python SQL Pandas data analysis",
                "keywords": ["python", "sql", "pandas", "data analysis"],
                "keyword_scores": [],
                "analyzed_at": "Oct 22, 2026 at 11:00 PM",
            }
            app_test.run(timeout=20)

            app_test.radio[0].set_value("Job Match").run(timeout=20)
            job_description = "Python SQL HTML"
            app_test.text_area[0].set_value(job_description).run(timeout=20)
            app_test.button[0].click().run(timeout=20)
            self.assertIsNotNone(app_test.session_state["match_result"])

            app_test.radio[0].set_value("Results").run(timeout=20)
            self.assertEqual(
                app_test.session_state["job_description"], job_description
            )
            self.assertIsNotNone(app_test.session_state["match_result"])

            app_test.radio[0].set_value("Job Match").run(timeout=20)
            self.assertEqual(app_test.text_area[0].value, job_description)
            changed_description = "Python gardening"
            app_test.text_area[0].set_value(changed_description).run(timeout=20)
            self.assertEqual(
                app_test.session_state["job_description"], changed_description
            )
            self.assertIsNone(app_test.session_state["match_result"])


if __name__ == "__main__":
    unittest.main()

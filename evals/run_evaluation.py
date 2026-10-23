"""
End-to-end evaluation runner for the existing Hybrid RAG project.

Run from the project root:
    python -m evals.run_evaluation

Optional semantic answer judge (uses additional Groq calls):
    python -m evals.run_evaluation --llm-judge

Outputs:
    evals/results/rag_evaluation_results.csv
    evals/results/rag_evaluation_report.md

This script initializes the same components as main.py. It does not modify the
production RAG pipeline or its retrieval settings.
"""
import argparse
import csv
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from evals.golden_dataset import GOLDEN_CASES
from evals.metrics import token_f1, retrieval_relevance_at_k, abstention_correct

from src.embeddings.embedder import EmbeddingModel
from src.vectorstore.chroma_store import ChromaVectorStore
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.pipeline import RetrievalPipeline
from src.generation.generator import get_generator
from src.generation.rag_generator import RAGGenerator
from src.generation.citation_judge import CitationJudge
from src.rag_pipeline import RAGPipeline


def build_rag_pipeline():
    """Build the same pipeline configuration used by main.py."""
    embedding_model = EmbeddingModel(model_name="BAAI/bge-small-en-v1.5")
    vector_store = ChromaVectorStore(
        persist_directory="./data/chroma",
        collection_name="internal_docs",
    )
    count = vector_store.count()
    if count == 0:
        raise RuntimeError(
            "ChromaDB collection 'internal_docs' is empty. "
            "Run the ingestion pipeline before evaluation."
        )

    dense_retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )
    sparse_retriever = BM25Retriever(vector_store=vector_store)
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60,
    )
    reranker = Reranker(model_name="BAAI/bge-reranker-base")
    retrieval_pipeline = RetrievalPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
    )
    generator = get_generator(prefer="groq")
    rag_generator = RAGGenerator(generator=generator)
    citation_judge = CitationJudge(generator=generator)
    rag_pipeline = RAGPipeline(
        retrieval_pipeline=retrieval_pipeline,
        rag_generator=rag_generator,
        citation_judge=citation_judge,
    )
    return rag_pipeline, generator, count


def judge_answer(generator, question: str, expected: str, actual: str) -> str:
    """LLM judge: returns CORRECT, PARTIAL, or INCORRECT."""
    prompt = f"""Evaluate whether the candidate answer correctly answers the question
using the reference answer as the rubric. Do not require exact wording.
Do not reward unsupported extra claims. Return exactly one label:
CORRECT, PARTIAL, or INCORRECT.

QUESTION:
{question}

REFERENCE ANSWER:
{expected}

CANDIDATE ANSWER:
{actual}

LABEL:"""
    raw = generator.generate(
        prompt=prompt,
        max_new_tokens=128,
        system="You are a strict, concise answer-quality evaluator.",
    )
    normalized = (raw or "").strip().upper()
    # Prefer exact labels, while tolerating a short explanatory suffix.
    for label in ("INCORRECT", "PARTIAL", "CORRECT"):
        if re.search(rf"\b{label}\b", normalized):
            return label
    return "UNKNOWN"


def chunk_source(chunk: Dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {}) or {}
    source = (
        metadata.get("source_file")
        or metadata.get("source")
        or metadata.get("file_name")
        or metadata.get("filename")
        or ""
    )
    return Path(str(source)).name if source else ""


def run_evaluation(use_llm_judge: bool = False) -> List[Dict[str, Any]]:
    rag_pipeline, generator, chunk_count = build_rag_pipeline()
    print(f"Connected to ChromaDB: {chunk_count} chunks")
    print(f"Evaluating {len(GOLDEN_CASES)} golden questions")
    if use_llm_judge:
        print("LLM answer judge is ON; this makes extra model calls.")

    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(GOLDEN_CASES, start=1):
        print(f"[{index:02d}/{len(GOLDEN_CASES)}] {case.case_id}: {case.question}")
        started = time.perf_counter()
        try:
            result = rag_pipeline.ask(case.question)
            latency = time.perf_counter() - started
            answer = str(result.get("answer", "") or "")
            chunks = result.get("retrieved_chunks", []) or []
            answer_lower = answer.lower().strip()

            refusal_phrases = [
                "i don't have enough information",
                "i do not have enough information",
                "the provided documents do not contain",
                "the documents do not provide enough information",
                "i cannot answer this question from the provided documents",
                "i can't answer this question from the provided documents",
            ]
            
            explicit_refusal = any(
                phrase in answer_lower for phrase in refusal_phrases
            )
            
            abstained = bool(result.get("abstained", False)) or explicit_refusal

            # Evaluate only the top 5 retrieved chunks.
            top_chunks = chunks[:5]
            
            # Keep source names for display, but use unique filenames for document metrics.
            source_names = [chunk_source(c) for c in top_chunks if chunk_source(c)]
            retrieved_sources = set(source_names)
            expected_sources = {
                Path(source).name for source in case.source_files if source
            }
            
            if expected_sources:
                matched_sources = retrieved_sources.intersection(expected_sources)
            
                # 1 if at least one expected document was retrieved; otherwise 0.
                source_hit_at_5 = float(bool(matched_sources))
            
                # Fraction of expected documents found. This can never exceed 1.0.
                source_recall_at_5 = len(matched_sources) / len(expected_sources)
            
                # Fraction of unique retrieved documents that are expected documents.
                source_precision_at_5 = (
                    len(matched_sources) / len(retrieved_sources)
                    if retrieved_sources else 0.0
                )
            else:
                # No expected source labels are supplied for this question.
                source_recall_at_5 = ""
                source_hit_at_5 = ""
                source_precision_at_5 = ""

            f1 = token_f1(answer, case.golden_answer)
            abstention_ok = abstention_correct(case.answerable, abstained)
            verifications = result.get("claim_verifications", []) or []
            cited_claims = [v for v in verifications if v.get("citations")]
            supported_claims = sum(v.get("supported") is True for v in cited_claims)
            observed_citation_accuracy = (
                supported_claims / len(cited_claims) if cited_claims else 0.0
            )
            observed_citation_coverage = float(result.get("citation_coverage", 0.0) or 0.0)

            if use_llm_judge:
                answer_judgment = judge_answer(generator, case.question, case.golden_answer, answer)
            else:
                answer_judgment = "NOT_RUN"

            # Faithfulness proxy from the pipeline's claim verifier.
            # Treat no cited claims as unverified rather than declaring the answer faithful.
            faithfulness_proxy = (
                supported_claims / len(cited_claims) if cited_claims else 0.0
            )

            rows.append({
                "case_id": case.case_id,
                "category": case.category,
                "question": case.question,
                "answerable_in_corpus": case.answerable,
                "expected_sources": ";".join(case.source_files),
                "answer": answer,
                "abstained": abstained,
                "abstention_correct": abstention_ok,
                "answer_token_f1": round(f1, 4),
                "answer_llm_judgment": answer_judgment,
                "source_hit_at_5": source_hit_at_5,
                "source_recall_at_5": source_recall_at_5,
                "source_precision_at_5": source_precision_at_5,
                "retrieval_relevance_pipeline": result.get("retrieval_relevance", ""),
                "citation_coverage_pipeline": observed_citation_coverage,
                "citation_accuracy_pipeline": result.get("citation_accuracy", ""),
                "citation_accuracy_from_claims": round(observed_citation_accuracy, 4),
                "faithfulness_proxy": round(faithfulness_proxy, 4),
                "invalid_citations_count": len(result.get("invalid_citations", []) or []),
                "retrieved_sources": ";".join(source_names),
                "latency_seconds": round(latency, 3),
                "error": "",
            })
        except Exception as exc:
            latency = time.perf_counter() - started
            rows.append({
                "case_id": case.case_id,
                "category": case.category,
                "question": case.question,
                "answerable_in_corpus": case.answerable,
                "expected_sources": ";".join(case.source_files),
                "answer": "",
                "abstained": "",
                "abstention_correct": "",
                "answer_token_f1": "",
                "answer_llm_judgment": "ERROR" if use_llm_judge else "NOT_RUN",
                "source_hit_at_5": "",
                "source_recall_at_5": "",
                "source_precision_at_5": "",
                "retrieval_relevance_pipeline": "",
                "citation_coverage_pipeline": "",
                "citation_accuracy_pipeline": "",
                "citation_accuracy_from_claims": "",
                "faithfulness_proxy": "",
                "invalid_citations_count": "",
                "retrieved_sources": "",
                "latency_seconds": round(latency, 3),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    return rows


def mean_numeric(rows, key):
    vals = []
    for row in rows:
        value = row.get(key, "")
        try:
            if value != "":
                vals.append(float(value))
        except (TypeError, ValueError):
            pass
    return statistics.mean(vals) if vals else 0.0


def write_reports(rows: List[Dict[str, Any]], use_llm_judge: bool) -> None:
    """Write a concise Excel workbook with summary and individual results."""

    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    out_dir = Path("evals") / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    excel_path = out_dir / "rag_evaluation_results.xlsx"

    successful = [row for row in rows if not row.get("error")]
    failed = [row for row in rows if row.get("error")]

    answerable = [
        row for row in successful
        if row.get("answerable_in_corpus") is True
    ]
    unanswerable = [
        row for row in successful
        if row.get("answerable_in_corpus") is False
    ]

    correct_abstentions = sum(
        row.get("abstention_correct") is True
        for row in successful
    )

    correct_unanswerable_refusals = sum(
        row.get("abstention_correct") is True
        and row.get("answerable_in_corpus") is False
        for row in successful
    )

    workbook = Workbook()

    # -------------------------
    # Sheet 1: Summary
    # -------------------------
    summary = workbook.active
    summary.title = "Summary"

    summary.append(["HYBRID RAG EVALUATION SUMMARY", "Value"])
    summary.append(["Total test questions", len(rows)])
    summary.append(["Successfully evaluated", len(successful)])
    summary.append(["Failed questions", len(failed)])
    summary.append([
        "Source Hit@5 (answerable questions)",
        round(mean_numeric(answerable, "source_hit_at_5"), 3),
    ])
    summary.append([
        "Source Recall@5 (answerable questions)",
        round(mean_numeric(answerable, "source_recall_at_5"), 3),
    ])
    summary.append([
        "Citation Accuracy (pipeline estimate)",
        round(mean_numeric(successful, "citation_accuracy_pipeline"), 3),
    ])
    summary.append([
        "Citation Coverage (pipeline estimate)",
        round(mean_numeric(successful, "citation_coverage_pipeline"), 3),
    ])
    summary.append([
        "Correct abstention decisions",
        f"{correct_abstentions}/{len(successful)}",
    ])
    summary.append([
        "Correct refusals on unanswerable questions",
        f"{correct_unanswerable_refusals}/{len(unanswerable)}",
    ])
    summary.append([
        "Average latency (seconds)",
        round(mean_numeric(successful, "latency_seconds"), 2),
    ])
    summary.append([
        "LLM answer judge enabled",
        "Yes" if use_llm_judge else "No",
    ])

    # -------------------------
    # Sheet 2: Results
    # -------------------------
    results_sheet = workbook.create_sheet("Results")

    if rows:
        headers = list(rows[0].keys())
        results_sheet.append(headers)

        for row in rows:
            results_sheet.append([row.get(header, "") for header in headers])

    # -------------------------
    # Basic spreadsheet formatting
    # -------------------------
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions

        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(
                fill_type="solid",
                fgColor="244062",
            )
            cell.alignment = Alignment(
                vertical="center",
                wrap_text=True,
            )

        for column_cells in sheet.columns:
            max_length = 0
            column_letter = get_column_letter(column_cells[0].column)

            for cell in column_cells:
                cell.alignment = Alignment(
                    vertical="top",
                    wrap_text=True,
                )

                if cell.value is not None:
                    max_length = max(
                        max_length,
                        min(len(str(cell.value)), 55),
                    )

            sheet.column_dimensions[column_letter].width = min(
                max(max_length + 2, 14),
                55,
            )

    summary.column_dimensions["A"].width = 43
    summary.column_dimensions["B"].width = 25

    workbook.save(excel_path)
    print(f"\nExcel evaluation results saved to: {excel_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate the Hybrid RAG pipeline on the golden dataset.")
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Use an additional LLM call to judge answer correctness for each question.",
    )
    args = parser.parse_args()
    rows = run_evaluation(use_llm_judge=args.llm_judge)
    write_reports(rows, use_llm_judge=args.llm_judge)


if __name__ == "__main__":
    main()

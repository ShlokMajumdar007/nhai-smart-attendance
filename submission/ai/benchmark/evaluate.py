"""
benchmark/evaluate.py
======================
Evaluation suite for the NHAI Face Authentication recognition pipeline.

Computes industry-standard biometric and ML metrics:
    - Accuracy, Precision, Recall, F1 Score
    - False Acceptance Rate (FAR)
    - False Rejection Rate (FRR)
    - Equal Error Rate (EER)
    - ROC AUC
    - Confusion matrix

Accepts a CSV of labelled verification pairs:
    genuine_embedding_path, impostor_embedding_path, label (1=genuine/0=impostor)

Or runs a self-contained demo with synthetically generated embeddings when
no dataset is supplied.

Usage:
    # Self-contained demo (no dataset required)
    python -m benchmark.evaluate --output eval_report.json

    # With a real pairs CSV
    python -m benchmark.evaluate --pairs pairs.csv --output eval_report.json

    # Sweep thresholds and find EER
    python -m benchmark.evaluate --sweep --output eval_report.json
"""

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ThresholdMetrics:
    """All metrics at a single cosine similarity threshold."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    far: float            # False Acceptance Rate = FP / (FP + TN)
    frr: float            # False Rejection Rate  = FN / (FN + TP)
    tp: int
    tn: int
    fp: int
    fn: int
    confusion_matrix: List[List[int]]


@dataclass
class EvaluationReport:
    """Full evaluation output."""

    timestamp: str
    n_genuine_pairs: int
    n_impostor_pairs: int
    total_pairs: int
    threshold_used: float
    metrics_at_threshold: ThresholdMetrics
    eer: float                          # Equal Error Rate
    eer_threshold: float                # Threshold at EER
    roc_auc: float
    sweep_results: List[ThresholdMetrics] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

class FaceAuthEvaluator:
    """
    Evaluates the face recognition pipeline on a labelled pair dataset.

    Each pair consists of two embeddings (genuine or impostor) and a ground-
    truth label (1 = same person, 0 = different people).  The evaluator
    computes cosine similarity for each pair and classifies it against a
    configurable threshold.

    Args:
        threshold: Default cosine similarity threshold for binary decisions.
    """

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold

    def evaluate(
        self,
        embeddings_a: np.ndarray,   # (N, 128)
        embeddings_b: np.ndarray,   # (N, 128)
        labels: np.ndarray,         # (N,) int  1=genuine 0=impostor
        sweep: bool = False,
    ) -> EvaluationReport:
        """
        Run the full evaluation.

        Args:
            embeddings_a: First embeddings in each pair.
            embeddings_b: Second embeddings in each pair.
            labels:       Ground-truth labels (1 genuine / 0 impostor).
            sweep:        If True, evaluate across a range of thresholds.

        Returns:
            EvaluationReport
        """
        assert len(embeddings_a) == len(embeddings_b) == len(labels), \
            "All arrays must have the same length."

        similarities = self._batch_cosine_similarity(embeddings_a, embeddings_b)
        n_genuine = int(labels.sum())
        n_impostor = int((labels == 0).sum())

        logger.info(
            "Dataset — %d pairs (%d genuine, %d impostor)",
            len(labels), n_genuine, n_impostor,
        )

        # Metrics at the configured threshold
        metrics = self._compute_metrics(similarities, labels, self.threshold)

        # ROC AUC
        roc_auc = self._compute_roc_auc(similarities, labels)

        # EER
        eer, eer_threshold = self._compute_eer(similarities, labels)

        # Optional sweep
        sweep_results: List[ThresholdMetrics] = []
        if sweep:
            thresholds = np.linspace(0.30, 0.95, 66)
            for t in thresholds:
                sweep_results.append(self._compute_metrics(similarities, labels, float(t)))
            logger.info("Threshold sweep complete (%d points)", len(sweep_results))

        report = EvaluationReport(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            n_genuine_pairs=n_genuine,
            n_impostor_pairs=n_impostor,
            total_pairs=len(labels),
            threshold_used=self.threshold,
            metrics_at_threshold=metrics,
            eer=round(eer, 6),
            eer_threshold=round(eer_threshold, 4),
            roc_auc=round(roc_auc, 6),
            sweep_results=sweep_results if sweep else [],
        )

        self._print_report(report)
        return report

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _compute_metrics(
        self, similarities: np.ndarray, labels: np.ndarray, threshold: float
    ) -> ThresholdMetrics:
        predictions = (similarities >= threshold).astype(int)

        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions, zero_division=0)
        rec = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        cm = confusion_matrix(labels, predictions, labels=[0, 1])

        # cm layout: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return ThresholdMetrics(
            threshold=round(threshold, 4),
            accuracy=round(float(acc), 6),
            precision=round(float(prec), 6),
            recall=round(float(rec), 6),
            f1=round(float(f1), 6),
            far=round(float(far), 6),
            frr=round(float(frr), 6),
            tp=int(tp),
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            confusion_matrix=cm.tolist(),
        )

    def _compute_roc_auc(self, similarities: np.ndarray, labels: np.ndarray) -> float:
        try:
            return float(roc_auc_score(labels, similarities))
        except ValueError as exc:
            logger.warning("ROC AUC computation failed: %s", exc)
            return 0.0

    def _compute_eer(
        self, similarities: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute Equal Error Rate (EER) — the threshold where FAR ≈ FRR.
        Uses the ROC curve from scikit-learn.
        """
        try:
            fpr, tpr, thresholds = roc_curve(labels, similarities, pos_label=1)
            fnr = 1.0 - tpr
            # Find index where |FAR - FRR| is minimised
            idx = int(np.argmin(np.abs(fpr - fnr)))
            eer = float((fpr[idx] + fnr[idx]) / 2.0)
            eer_threshold = float(thresholds[idx]) if idx < len(thresholds) else self.threshold
            logger.info("EER = %.4f at threshold = %.4f", eer, eer_threshold)
            return eer, eer_threshold
        except Exception as exc:
            logger.warning("EER computation failed: %s", exc)
            return 0.0, self.threshold

    @staticmethod
    def _batch_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Vectorised cosine similarity between two arrays of embeddings.
        Both inputs are L2-normalised before computing dot products.
        """
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.clip((a_norm * b_norm).sum(axis=1), -1.0, 1.0)

    # ------------------------------------------------------------------
    # Report formatting
    # ------------------------------------------------------------------

    def _print_report(self, report: EvaluationReport):
        m = report.metrics_at_threshold
        bar = "=" * 66
        print(f"\n{bar}")
        print("  NHAI Face Authentication — Evaluation Report")
        print(f"  {report.timestamp}")
        print(bar)
        print(f"\n  Dataset: {report.total_pairs} pairs "
              f"({report.n_genuine_pairs} genuine / {report.n_impostor_pairs} impostor)")
        print(f"  Threshold: {report.threshold_used:.4f}")
        print()
        print(f"  {'Accuracy':<20} {m.accuracy * 100:.2f}%  (target > 95%)")
        print(f"  {'Precision':<20} {m.precision * 100:.2f}%")
        print(f"  {'Recall':<20} {m.recall * 100:.2f}%")
        print(f"  {'F1 Score':<20} {m.f1 * 100:.2f}%")
        print(f"  {'FAR':<20} {m.far * 100:.4f}%")
        print(f"  {'FRR':<20} {m.frr * 100:.4f}%")
        print(f"  {'EER':<20} {report.eer * 100:.4f}%")
        print(f"  {'EER Threshold':<20} {report.eer_threshold:.4f}")
        print(f"  {'ROC AUC':<20} {report.roc_auc:.6f}")
        print()
        print("  Confusion Matrix (rows=actual, cols=predicted):")
        print("                  Pred:Impostor  Pred:Genuine")
        print(f"  Actual:Impostor    {m.tn:>6}         {m.fp:>6}")
        print(f"  Actual:Genuine     {m.fn:>6}         {m.tp:>6}")
        print(f"\n{bar}\n")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_pairs_from_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load evaluation pairs from a CSV file.

    Expected columns:
        embedding_a_path, embedding_b_path, label

    Where embedding files are .npy files of shape (128,).

    Returns:
        (embeddings_a, embeddings_b, labels) as numpy arrays.
    """
    embeddings_a, embeddings_b, labels = [], [], []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emb_a = np.load(row["embedding_a_path"]).astype(np.float32)
            emb_b = np.load(row["embedding_b_path"]).astype(np.float32)
            label = int(row["label"])
            embeddings_a.append(emb_a)
            embeddings_b.append(emb_b)
            labels.append(label)

    return (
        np.array(embeddings_a),
        np.array(embeddings_b),
        np.array(labels, dtype=np.int32),
    )


def generate_synthetic_dataset(
    n_genuine: int = 500,
    n_impostor: int = 500,
    embedding_dim: int = 128,
    genuine_sim_mean: float = 0.82,
    impostor_sim_mean: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic evaluation dataset for demo / CI purposes.

    Genuine pairs are sampled close together in embedding space;
    impostor pairs are sampled far apart.
    """
    rng = np.random.default_rng(42)

    def _random_embedding():
        v = rng.standard_normal(embedding_dim).astype(np.float32)
        return v / np.linalg.norm(v)

    emb_a_list, emb_b_list, label_list = [], [], []

    # Genuine pairs — perturb a shared base embedding
    for _ in range(n_genuine):
        base = _random_embedding()
        noise = rng.standard_normal(embedding_dim).astype(np.float32) * 0.18
        perturbed = base + noise
        perturbed /= np.linalg.norm(perturbed)
        emb_a_list.append(base)
        emb_b_list.append(perturbed)
        label_list.append(1)

    # Impostor pairs — independent random embeddings
    for _ in range(n_impostor):
        emb_a_list.append(_random_embedding())
        emb_b_list.append(_random_embedding())
        label_list.append(0)

    return (
        np.array(emb_a_list),
        np.array(emb_b_list),
        np.array(label_list, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate face recognition accuracy for the NHAI system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pairs",
        default=None,
        help="Path to pairs CSV (embedding_a_path, embedding_b_path, label). "
             "If omitted, runs synthetic demo.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Cosine similarity threshold for genuine/impostor decision.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep thresholds from 0.30 to 0.95 and include results in report.",
    )
    parser.add_argument(
        "--output",
        default="eval_report.json",
        help="Path for the JSON report output file.",
    )
    parser.add_argument(
        "--genuine",
        type=int,
        default=500,
        help="(synthetic mode) Number of genuine pairs.",
    )
    parser.add_argument(
        "--impostor",
        type=int,
        default=500,
        help="(synthetic mode) Number of impostor pairs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.pairs:
        logger.info("Loading pairs from %s", args.pairs)
        embeddings_a, embeddings_b, labels = load_pairs_from_csv(args.pairs)
    else:
        logger.info(
            "No pairs CSV supplied — using synthetic dataset "
            "(%d genuine, %d impostor)", args.genuine, args.impostor
        )
        embeddings_a, embeddings_b, labels = generate_synthetic_dataset(
            n_genuine=args.genuine,
            n_impostor=args.impostor,
        )

    evaluator = FaceAuthEvaluator(threshold=args.threshold)
    report = evaluator.evaluate(
        embeddings_a=embeddings_a,
        embeddings_b=embeddings_b,
        labels=labels,
        sweep=args.sweep,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info("Evaluation report saved to %s", output_path)


if __name__ == "__main__":
    main()
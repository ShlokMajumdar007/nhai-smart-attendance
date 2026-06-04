"""
benchmark/benchmark.py
=======================
Performance benchmarking suite for the NHAI Face Authentication System.

Measures:
    - Model load time
    - Enrollment latency (per-frame embedding + DB write)
    - Recognition latency (detection + embedding + similarity)
    - Liveness challenge latency (per-frame update)
    - Average FPS on synthetic frames
    - Peak and average memory usage (RSS)
    - CPU utilisation during inference
    - On-disk model size

Outputs a human-readable report to stdout and a machine-readable JSON file.

Usage:
    python -m benchmark.benchmark --model-dir ai/models --output benchmark_report.json
    python -m benchmark.benchmark --help
"""

import argparse
import json
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import psutil

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LatencyStat:
    """Statistics for a repeated latency measurement (all values in ms)."""

    name: str
    n_samples: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float

    @classmethod
    def from_samples(cls, name: str, samples: List[float]) -> "LatencyStat":
        arr = np.array(samples, dtype=np.float64)
        return cls(
            name=name,
            n_samples=len(arr),
            mean_ms=float(np.mean(arr)),
            median_ms=float(np.median(arr)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            std_ms=float(np.std(arr)),
        )

    def summary_line(self) -> str:
        return (
            f"{self.name:<30}  "
            f"mean={self.mean_ms:6.1f}ms  "
            f"p95={self.p95_ms:6.1f}ms  "
            f"p99={self.p99_ms:6.1f}ms  "
            f"min={self.min_ms:5.1f}ms  "
            f"max={self.max_ms:5.1f}ms  "
            f"n={self.n_samples}"
        )


@dataclass
class ResourceStat:
    """Memory and CPU measurements."""

    peak_memory_mb: float
    avg_memory_mb: float
    avg_cpu_percent: float
    process_rss_mb: float       # Resident Set Size at report time


@dataclass
class ModelInfo:
    """On-disk model metadata."""

    path: str
    size_mb: float
    exists: bool


@dataclass
class BenchmarkReport:
    """Top-level benchmark output."""

    timestamp: str
    platform: str
    python_version: str
    model_info: ModelInfo
    model_load_ms: float
    fps_synthetic: float
    latencies: List[LatencyStat]
    resources: ResourceStat
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Synthetic frame generation
# ---------------------------------------------------------------------------

def _make_synthetic_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """
    Generate a realistic synthetic BGR frame for benchmarking.
    Fills the frame with a skin-tone gradient so face quality checks
    are more representative than pure noise.
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Skin-tone base: BGR (80, 120, 180)
    frame[:, :, 0] = 80
    frame[:, :, 1] = 120
    frame[:, :, 2] = 180
    # Add gradient
    for c, base in enumerate([80, 120, 180]):
        frame[:, :, c] = np.clip(
            base + np.linspace(-20, 20, w, dtype=np.float32), 0, 255
        ).astype(np.uint8)
    return frame


def _make_synthetic_aligned_face() -> np.ndarray:
    """
    112×112 float32 face tensor in [-1, 1] (MobileFaceNet input format).
    """
    rng = np.random.default_rng(42)
    face = rng.uniform(-1.0, 1.0, (112, 112, 3)).astype(np.float32)
    return face


def _make_synthetic_landmarks(n: int = 468) -> np.ndarray:
    """
    Realistic-looking (n, 3) landmark array in pixel space (480×640 frame).
    """
    rng = np.random.default_rng(7)
    lm = rng.uniform(0, 1, (n, 3)).astype(np.float32)
    lm[:, 0] *= 640   # x
    lm[:, 1] *= 480   # y
    lm[:, 2] *= 0.05  # z (small)
    return lm


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:

    def __init__(self, model_dir: str, n_warmup: int = 5, n_samples: int = 50):
        self._model_dir = model_dir
        self._n_warmup = n_warmup
        self._n_samples = n_samples
        self._process = psutil.Process(os.getpid())
        self._memory_samples: List[float] = []
        self._cpu_samples: List[float] = []
        self._notes: List[str] = []

    def run(self) -> BenchmarkReport:
        logger.info("=" * 60)
        logger.info("NHAI Face Auth — Benchmark Suite")
        logger.info("=" * 60)

        model_info = self._measure_model_info()
        model_load_ms = self._measure_model_load()

        embedder = self._load_embedder()
        latencies: List[LatencyStat] = []

        if embedder is not None:
            latencies.append(self._bench_embedding(embedder))
            latencies.append(self._bench_cosine_similarity())
            latencies.append(self._bench_liveness_blink())
            latencies.append(self._bench_liveness_smile())
            latencies.append(self._bench_liveness_head_turn())
            latencies.append(self._bench_full_pipeline(embedder))
        else:
            self._notes.append(
                "Embedding benchmarks skipped — mobilefacenet.tflite not found. "
                "Place the model file in ai/models/ to enable full benchmarking."
            )
            latencies.append(self._bench_cosine_similarity())
            latencies.append(self._bench_liveness_blink())
            latencies.append(self._bench_liveness_smile())
            latencies.append(self._bench_liveness_head_turn())

        fps = self._bench_fps(embedder)
        resources = self._collect_resources()

        report = BenchmarkReport(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
            python_version=sys.version.split()[0],
            model_info=model_info,
            model_load_ms=model_load_ms,
            fps_synthetic=fps,
            latencies=latencies,
            resources=resources,
            notes=self._notes,
        )

        self._print_report(report)
        return report

    # ------------------------------------------------------------------
    # Individual benchmarks
    # ------------------------------------------------------------------

    def _measure_model_info(self) -> ModelInfo:
        model_path = os.path.join(self._model_dir, "mobilefacenet.tflite")
        exists = os.path.isfile(model_path)
        size_mb = os.path.getsize(model_path) / (1024 * 1024) if exists else 0.0
        logger.info("Model path  : %s", model_path)
        logger.info("Model exists: %s", exists)
        if exists:
            logger.info("Model size  : %.2f MB", size_mb)
        return ModelInfo(path=model_path, size_mb=round(size_mb, 3), exists=exists)

    def _measure_model_load(self) -> float:
        model_path = os.path.join(self._model_dir, "mobilefacenet.tflite")
        if not os.path.isfile(model_path):
            self._notes.append("Model load time skipped — .tflite file absent.")
            return 0.0

        try:
            t0 = time.monotonic()
            from ai.embedding.mobilefacenet import MobileFaceNet
            MobileFaceNet(model_dir=self._model_dir)
            elapsed = (time.monotonic() - t0) * 1000
            logger.info("Model load time: %.1f ms", elapsed)
            return round(elapsed, 2)
        except Exception as exc:
            logger.warning("Model load benchmark failed: %s", exc)
            self._notes.append(f"Model load failed: {exc}")
            return 0.0

    def _load_embedder(self):
        model_path = os.path.join(self._model_dir, "mobilefacenet.tflite")
        if not os.path.isfile(model_path):
            return None
        try:
            from ai.embedding.mobilefacenet import MobileFaceNet
            return MobileFaceNet(model_dir=self._model_dir)
        except Exception as exc:
            logger.warning("Could not load embedder: %s", exc)
            return None

    def _bench_embedding(self, embedder) -> LatencyStat:
        face = _make_synthetic_aligned_face()
        samples: List[float] = []

        # Warmup
        for _ in range(self._n_warmup):
            embedder.get_embedding(face)

        for _ in range(self._n_samples):
            t0 = time.monotonic()
            embedder.get_embedding(face)
            samples.append((time.monotonic() - t0) * 1000)
            self._sample_resources()

        stat = LatencyStat.from_samples("embedding_inference", samples)
        logger.info(stat.summary_line())
        return stat

    def _bench_cosine_similarity(self) -> LatencyStat:
        from ai.recognition.similarity import cosine_similarity
        rng = np.random.default_rng(1)
        a = rng.standard_normal(128).astype(np.float32)
        b = rng.standard_normal(128).astype(np.float32)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)

        # Build a mock database of 1000 enrolled embeddings
        enrolled = [(f"emp_{i}", rng.standard_normal(128).astype(np.float32)) for i in range(1000)]

        from ai.recognition.similarity import find_best_match
        samples: List[float] = []

        for _ in range(self._n_warmup):
            find_best_match(a, enrolled)

        for _ in range(self._n_samples):
            t0 = time.monotonic()
            find_best_match(a, enrolled)
            samples.append((time.monotonic() - t0) * 1000)

        stat = LatencyStat.from_samples("cosine_match_1k_employees", samples)
        logger.info(stat.summary_line())
        return stat

    def _bench_liveness_blink(self) -> LatencyStat:
        from ai.liveness.blink import BlinkDetector
        detector = BlinkDetector()
        landmarks = _make_synthetic_landmarks()
        samples: List[float] = []

        for _ in range(self._n_warmup):
            detector.update(landmarks)

        for _ in range(self._n_samples):
            t0 = time.monotonic()
            detector.update(landmarks)
            samples.append((time.monotonic() - t0) * 1000)

        stat = LatencyStat.from_samples("liveness_blink_per_frame", samples)
        logger.info(stat.summary_line())
        return stat

    def _bench_liveness_smile(self) -> LatencyStat:
        from ai.liveness.smile import SmileDetector
        detector = SmileDetector()
        landmarks = _make_synthetic_landmarks()
        samples: List[float] = []

        for _ in range(self._n_warmup + 20):   # calibration needs ~20 frames
            detector.calibrate(landmarks)

        for _ in range(self._n_samples):
            t0 = time.monotonic()
            detector.update(landmarks)
            samples.append((time.monotonic() - t0) * 1000)

        stat = LatencyStat.from_samples("liveness_smile_per_frame", samples)
        logger.info(stat.summary_line())
        return stat

    def _bench_liveness_head_turn(self) -> LatencyStat:
        from ai.liveness.head_turn import HeadTurnDetector
        detector = HeadTurnDetector(direction="left")
        pose = {"yaw": -25.0, "pitch": 0.0, "roll": 0.0}
        samples: List[float] = []

        for _ in range(self._n_samples):
            t0 = time.monotonic()
            detector.update(pose)
            samples.append((time.monotonic() - t0) * 1000)

        stat = LatencyStat.from_samples("liveness_head_turn_per_frame", samples)
        logger.info(stat.summary_line())
        return stat

    def _bench_full_pipeline(self, embedder) -> LatencyStat:
        """
        Synthetic end-to-end: embedding + cosine match.
        (Face detection excluded — requires real camera frame + GPU path.)
        """
        from ai.recognition.similarity import find_best_match
        rng = np.random.default_rng(3)

        face = _make_synthetic_aligned_face()
        enrolled = [(f"emp_{i}", rng.standard_normal(128).astype(np.float32)) for i in range(500)]

        samples: List[float] = []

        for _ in range(self._n_warmup):
            emb = embedder.get_embedding(face)
            find_best_match(emb, enrolled)

        for _ in range(self._n_samples):
            t0 = time.monotonic()
            emb = embedder.get_embedding(face)
            find_best_match(emb, enrolled)
            samples.append((time.monotonic() - t0) * 1000)
            self._sample_resources()

        stat = LatencyStat.from_samples("full_pipeline_embedding+match", samples)
        logger.info(stat.summary_line())
        return stat

    def _bench_fps(self, embedder) -> float:
        """
        Simulate processing N frames and compute throughput.
        Uses only the embedding step (no camera I/O).
        """
        if embedder is None:
            self._notes.append("FPS benchmark skipped — embedder unavailable.")
            return 0.0

        face = _make_synthetic_aligned_face()
        n_frames = 100

        t0 = time.monotonic()
        for _ in range(n_frames):
            embedder.get_embedding(face)
        elapsed = time.monotonic() - t0

        fps = n_frames / elapsed
        logger.info("Synthetic FPS (embedding only): %.1f", fps)
        return round(fps, 2)

    # ------------------------------------------------------------------
    # Resource monitoring
    # ------------------------------------------------------------------

    def _sample_resources(self):
        try:
            mem_mb = self._process.memory_info().rss / (1024 * 1024)
            cpu = self._process.cpu_percent(interval=None)
            self._memory_samples.append(mem_mb)
            self._cpu_samples.append(cpu)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def _collect_resources(self) -> ResourceStat:
        rss_mb = self._process.memory_info().rss / (1024 * 1024)
        mem_arr = np.array(self._memory_samples) if self._memory_samples else np.array([rss_mb])
        cpu_arr = np.array(self._cpu_samples) if self._cpu_samples else np.array([0.0])

        stat = ResourceStat(
            peak_memory_mb=round(float(mem_arr.max()), 2),
            avg_memory_mb=round(float(mem_arr.mean()), 2),
            avg_cpu_percent=round(float(cpu_arr.mean()), 2),
            process_rss_mb=round(rss_mb, 2),
        )
        logger.info(
            "Memory — peak=%.1fMB avg=%.1fMB rss=%.1fMB  CPU avg=%.1f%%",
            stat.peak_memory_mb, stat.avg_memory_mb,
            stat.process_rss_mb, stat.avg_cpu_percent,
        )
        return stat

    # ------------------------------------------------------------------
    # Report formatting
    # ------------------------------------------------------------------

    def _print_report(self, report: BenchmarkReport):
        bar = "=" * 70
        print(f"\n{bar}")
        print("  NHAI Face Authentication — Benchmark Report")
        print(f"  {report.timestamp}")
        print(f"  {report.platform}  •  Python {report.python_version}")
        print(bar)
        print(f"\n  Model: {report.model_info.path}")
        print(f"  Size : {report.model_info.size_mb:.2f} MB  (target < 20 MB)")
        print(f"  Load : {report.model_load_ms:.1f} ms")
        print(f"  FPS  : {report.fps_synthetic:.1f}  (synthetic, embedding-only)")
        print("\n  Latencies:")
        for stat in report.latencies:
            print(f"    {stat.summary_line()}")
        print("\n  Resources:")
        r = report.resources
        print(f"    Peak memory : {r.peak_memory_mb:.1f} MB  (target < 300 MB)")
        print(f"    Avg memory  : {r.avg_memory_mb:.1f} MB")
        print(f"    Avg CPU     : {r.avg_cpu_percent:.1f}%")
        if report.notes:
            print("\n  Notes:")
            for note in report.notes:
                print(f"    • {note}")
        print(f"\n{bar}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the NHAI Face Authentication pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default="ai/models",
        help="Directory containing mobilefacenet.tflite",
    )
    parser.add_argument(
        "--output",
        default="benchmark_report.json",
        help="Path for the JSON report output file",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of measurement samples per benchmark",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (excluded from stats)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    runner = BenchmarkRunner(
        model_dir=args.model_dir,
        n_warmup=args.warmup,
        n_samples=args.samples,
    )

    report = runner.run()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info("Report saved to %s", output_path)


if __name__ == "__main__":
    main()
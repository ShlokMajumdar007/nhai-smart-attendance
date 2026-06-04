"""
tests/test_verification.py
===========================
Pytest unit tests for ai/recognition/verify.py and ai/recognition/similarity.py

Tests cover:
    - cosine_similarity range and symmetry
    - find_best_match returns correct identity
    - find_best_match rejects below-threshold scores
    - rank_matches ordering
    - FaceVerifier.verify() with liveness passed/failed
    - FaceVerifier multi-frame buffer accumulation
    - FaceVerifier.reset() clears buffer
    - VerificationResult fields
    - 1:1 verify_pair
"""

import numpy as np
import pytest

from ai.recognition.similarity import (
    cosine_similarity,
    euclidean_distance,
    find_best_match,
    rank_matches,
)
from ai.recognition.verify import FaceVerifier, VerificationResult


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(99)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-10)


def _random_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(128).astype(np.float32)
    return _unit(v)


def _perturbed(base: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
    """Return an embedding close to base (high similarity)."""
    noise = RNG.standard_normal(128).astype(np.float32) * noise_scale
    return _unit(base + noise)


def _random_face() -> np.ndarray:
    """112×112×3 float32 in [-1, 1] — synthetic MobileFaceNet input."""
    return RNG.uniform(-1.0, 1.0, (112, 112, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# Stub embedder — returns deterministic embeddings without a real model file
# ---------------------------------------------------------------------------

class _StubEmbedder:
    """
    Deterministic stub replacing MobileFaceNet for unit testing.
    Returns a fixed embedding for any input.
    """

    def __init__(self, fixed_embedding: np.ndarray):
        self._embedding = _unit(fixed_embedding)

    def get_embedding(self, face: np.ndarray) -> np.ndarray:
        return self._embedding.copy()

    def get_average_embedding(self, faces) -> np.ndarray:
        return self._embedding.copy()

    @property
    def embedding_dim(self) -> int:
        return 128

    @property
    def similarity_threshold(self) -> float:
        return 0.65


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:

    def test_identical_embeddings_score_one(self):
        v = _random_embedding(1)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_embeddings_score_near_zero(self):
        a = np.zeros(128, dtype=np.float32)
        b = np.zeros(128, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        score = cosine_similarity(a, b)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_score_bounded_minus_one_to_one(self):
        for seed in range(20):
            a = _random_embedding(seed)
            b = _random_embedding(seed + 100)
            score = cosine_similarity(a, b)
            assert -1.0 <= score <= 1.0

    def test_symmetry(self):
        a = _random_embedding(5)
        b = _random_embedding(6)
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a), abs=1e-6)

    def test_close_embeddings_high_score(self):
        base = _random_embedding(10)
        close = _perturbed(base, noise_scale=0.05)
        score = cosine_similarity(base, close)
        assert score > 0.90, f"Close embeddings should score > 0.90, got {score:.4f}"

    def test_distant_embeddings_low_score(self):
        a = _random_embedding(20)
        b = _random_embedding(21)
        score = cosine_similarity(a, b)
        assert score < 0.80, f"Random embeddings typically score < 0.80, got {score:.4f}"

    def test_zero_vector_does_not_raise(self):
        z = np.zeros(128, dtype=np.float32)
        v = _random_embedding(3)
        score = cosine_similarity(z, v)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# find_best_match
# ---------------------------------------------------------------------------

class TestFindBestMatch:

    def _enrolled(self):
        """Build a small enrolled gallery."""
        return [
            ("alice", _random_embedding(1)),
            ("bob",   _random_embedding(2)),
            ("carol", _random_embedding(3)),
        ]

    def test_returns_correct_identity_for_genuine(self):
        enrolled = self._enrolled()
        alice_emb = enrolled[0][1]
        # Query close to alice
        query = _perturbed(alice_emb, noise_scale=0.05)
        person_id, score = find_best_match(query, enrolled, threshold=0.60)
        assert person_id == "alice", f"Expected 'alice', got '{person_id}' (score={score:.4f})"
        assert score >= 0.60

    def test_returns_none_for_impostor_below_threshold(self):
        enrolled = self._enrolled()
        # A completely different embedding — should not match
        impostor = _random_embedding(999)
        person_id, score = find_best_match(impostor, enrolled, threshold=0.99)
        assert person_id is None

    def test_empty_gallery_returns_none(self):
        query = _random_embedding(1)
        person_id, score = find_best_match(query, [], threshold=0.65)
        assert person_id is None
        assert score == 0.0

    def test_score_returned_on_no_match(self):
        enrolled = self._enrolled()
        impostor = _random_embedding(500)
        _, score = find_best_match(impostor, enrolled, threshold=0.99)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_threshold_boundary_accepted(self):
        enrolled = self._enrolled()
        alice_emb = enrolled[0][1]
        query = _perturbed(alice_emb, noise_scale=0.02)
        score_actual = cosine_similarity(query, alice_emb)
        # Set threshold just below actual score
        person_id, _ = find_best_match(query, enrolled, threshold=score_actual - 0.01)
        assert person_id == "alice"

    def test_threshold_boundary_rejected(self):
        enrolled = self._enrolled()
        alice_emb = enrolled[0][1]
        query = _perturbed(alice_emb, noise_scale=0.02)
        # Set threshold to 1.0 — nothing should match
        person_id, _ = find_best_match(query, enrolled, threshold=1.0)
        assert person_id is None

    def test_large_gallery_correct_match(self):
        """Verify correct match is found in a gallery of 1000 identities."""
        gallery = [(f"person_{i}", _random_embedding(i)) for i in range(1000)]
        target_id = "person_42"
        target_emb = gallery[42][1]
        query = _perturbed(target_emb, noise_scale=0.03)
        person_id, score = find_best_match(query, gallery, threshold=0.60)
        assert person_id == target_id, f"Expected person_42, got {person_id} (score={score:.4f})"


# ---------------------------------------------------------------------------
# rank_matches
# ---------------------------------------------------------------------------

class TestRankMatches:

    def test_ranked_descending(self):
        enrolled = [(f"p{i}", _random_embedding(i)) for i in range(10)]
        query = _random_embedding(0)
        ranked = rank_matches(query, enrolled, top_k=5)
        assert len(ranked) == 5
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True), "Results must be in descending order"

    def test_top_k_respected(self):
        enrolled = [(f"p{i}", _random_embedding(i)) for i in range(20)]
        query = _random_embedding(0)
        for k in (1, 5, 10):
            ranked = rank_matches(query, enrolled, top_k=k)
            assert len(ranked) == k


# ---------------------------------------------------------------------------
# FaceVerifier
# ---------------------------------------------------------------------------

class TestFaceVerifier:

    @pytest.fixture
    def alice_embedding(self):
        return _random_embedding(seed=1)

    @pytest.fixture
    def verifier(self, alice_embedding):
        enrolled = [
            ("alice", alice_embedding),
            ("bob",   _random_embedding(seed=2)),
        ]
        embedder = _StubEmbedder(alice_embedding)
        return FaceVerifier(
            detector=None,       # not needed for unit tests
            embedder=embedder,
            enrolled_embeddings=enrolled,
            threshold=0.65,
            verification_frames=3,
        )

    def test_verify_passes_with_liveness_and_genuine_face(self, verifier, alice_embedding):
        face = _random_face()
        # Fill buffer manually (bypasses detector)
        for _ in range(3):
            verifier._frame_buffer.append(face)

        result = verifier.verify(liveness_passed=True)
        assert result.success is True
        assert result.person_id == "alice"
        assert result.liveness_passed is True
        assert result.confidence >= 0.65

    def test_verify_fails_when_liveness_not_passed(self, verifier):
        face = _random_face()
        for _ in range(3):
            verifier._frame_buffer.append(face)

        result = verifier.verify(liveness_passed=False)
        assert result.success is False
        assert result.person_id is None
        assert result.rejection_reason == "liveness_failed"

    def test_verify_fails_with_empty_buffer(self, verifier):
        result = verifier.verify(liveness_passed=True)
        assert result.success is False
        assert result.rejection_reason == "no_frames"

    def test_buffer_cleared_after_verify(self, verifier):
        face = _random_face()
        for _ in range(3):
            verifier._frame_buffer.append(face)
        verifier.verify(liveness_passed=True)
        assert len(verifier._frame_buffer) == 0

    def test_reset_clears_buffer(self, verifier):
        face = _random_face()
        verifier._frame_buffer.append(face)
        verifier.reset()
        assert len(verifier._frame_buffer) == 0

    def test_add_frame_returns_true_when_buffer_full(self, verifier):
        face = _random_face()
        returned = []
        for _ in range(3):
            returned.append(verifier.add_frame(face))
        assert returned[-1] is True
        assert returned[0] is False

    def test_verification_result_fields(self, verifier):
        face = _random_face()
        for _ in range(3):
            verifier._frame_buffer.append(face)

        result = verifier.verify(liveness_passed=True)

        assert isinstance(result, VerificationResult)
        assert isinstance(result.success, bool)
        assert isinstance(result.confidence, float)
        assert isinstance(result.liveness_passed, bool)
        assert isinstance(result.latency_ms, float)
        assert result.latency_ms >= 0.0

    def test_latency_measured(self, verifier):
        face = _random_face()
        for _ in range(3):
            verifier._frame_buffer.append(face)
        result = verifier.verify(liveness_passed=True)
        assert result.latency_ms > 0.0

    def test_verify_pair_genuine(self, alice_embedding):
        embedder = _StubEmbedder(alice_embedding)
        verifier = FaceVerifier(
            detector=None,
            embedder=embedder,
            enrolled_embeddings=[],
            threshold=0.65,
        )
        close = _perturbed(alice_embedding, noise_scale=0.02)
        matched, score = verifier.verify_pair(alice_embedding, close)
        assert matched is True
        assert score >= 0.65

    def test_verify_pair_impostor(self, alice_embedding):
        embedder = _StubEmbedder(alice_embedding)
        verifier = FaceVerifier(
            detector=None,
            embedder=embedder,
            enrolled_embeddings=[],
            threshold=0.99,   # impossible threshold
        )
        other = _random_embedding(seed=777)
        matched, score = verifier.verify_pair(alice_embedding, other)
        assert matched is False
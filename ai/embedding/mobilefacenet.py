"""
mobilefacenet.py

TensorFlow Lite inference wrapper for MobileFaceNet.

Features:
- Dynamic embedding dimension detection
- Automatic model validation
- L2-normalized embeddings
- Batch inference support
- Multi-frame averaging support
"""

import json
import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try lightweight runtime first
try:
    import tflite_runtime.interpreter as tflite

    TFLITE_RUNTIME = True
except ImportError:
    import tensorflow as tf

    tflite = tf.lite
    TFLITE_RUNTIME = False


class MobileFaceNet:
    """
    MobileFaceNet TensorFlow Lite wrapper.

    Expected input:
        (1, 112, 112, 3)

    Output:
        (1, embedding_dim)

    embedding_dim is detected automatically.
    """

    MODEL_FILE = "mobilefacenet.tflite"
    METADATA_FILE = "metadata.json"

    def __init__(
        self,
        model_dir: str,
        num_threads: int = 4,
    ):
        self.model_dir = model_dir
        self.num_threads = num_threads

        self._interpreter = None
        self._input_details = None
        self._output_details = None

        self._metadata = {}

        self.input_shape = None
        self.output_shape = None
        self._embedding_dim = None

        self._load_metadata()
        self._load_model()

    # ==========================================================
    # Metadata
    # ==========================================================

    def _load_metadata(self):
        metadata_path = os.path.join(
            self.model_dir,
            self.METADATA_FILE,
        )

        if not os.path.exists(metadata_path):
            logger.warning(
                "Metadata file not found: %s",
                metadata_path,
            )
            return

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

            logger.info(
                "Loaded metadata for model: %s",
                self._metadata.get("model_name", "unknown"),
            )

        except Exception as e:
            logger.exception(
                "Failed to load metadata: %s",
                e,
            )

    # ==========================================================
    # Model Loading
    # ==========================================================

    def _load_model(self):
        model_path = os.path.join(
            self.model_dir,
            self.MODEL_FILE,
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        logger.info("Loading model: %s", model_path)

        self._interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=self.num_threads,
        )

        self._interpreter.allocate_tensors()

        self._input_details = (
            self._interpreter.get_input_details()
        )

        self._output_details = (
            self._interpreter.get_output_details()
        )

        self.input_shape = tuple(
            self._input_details[0]["shape"]
        )

        self.output_shape = tuple(
            self._output_details[0]["shape"]
        )

        self._embedding_dim = int(
            self.output_shape[-1]
        )

        logger.info(
            "Model loaded successfully | "
            "input=%s output=%s embedding_dim=%d",
            self.input_shape,
            self.output_shape,
            self._embedding_dim,
        )

        self._validate_model()

    def _validate_model(self):
        expected_input = (1, 112, 112, 3)

        if self.input_shape != expected_input:
            raise RuntimeError(
                f"Unexpected model input shape. "
                f"Expected {expected_input}, "
                f"got {self.input_shape}"
            )

        logger.info(
            "Model validation passed."
        )

    # ==========================================================
    # Inference
    # ==========================================================

    def get_embedding(
        self,
        face: np.ndarray,
    ) -> np.ndarray:
        """
        Extract a single embedding.

        Parameters
        ----------
        face : np.ndarray
            Shape (112,112,3)
            Float32 normalized to [-1,1]

        Returns
        -------
        np.ndarray
            L2-normalized embedding.
        """

        if face.shape != (112, 112, 3):
            raise ValueError(
                f"Expected (112,112,3), "
                f"got {face.shape}"
            )

        if face.dtype != np.float32:
            face = face.astype(np.float32)

        input_tensor = np.expand_dims(
            face,
            axis=0,
        )

        self._interpreter.set_tensor(
            self._input_details[0]["index"],
            input_tensor,
        )

        self._interpreter.invoke()

        output = self._interpreter.get_tensor(
            self._output_details[0]["index"]
        )

        embedding = output[0]

        return self._l2_normalize(
            embedding.astype(np.float32)
        )

    def get_batch_embeddings(
        self,
        faces: List[np.ndarray],
    ) -> np.ndarray:
        """
        Extract embeddings for multiple faces.
        """

        embeddings = [
            self.get_embedding(face)
            for face in faces
        ]

        return np.array(
            embeddings,
            dtype=np.float32,
        )

    def get_average_embedding(
        self,
        faces: List[np.ndarray],
    ) -> np.ndarray:
        """
        Average embeddings across frames.
        """

        if not faces:
            raise ValueError(
                "No faces provided."
            )

        embeddings = self.get_batch_embeddings(
            faces
        )

        avg_embedding = np.mean(
            embeddings,
            axis=0,
        )

        return self._l2_normalize(
            avg_embedding
        )

    # ==========================================================
    # Utilities
    # ==========================================================

    @staticmethod
    def _l2_normalize(
        embedding: np.ndarray,
    ) -> np.ndarray:
        norm = np.linalg.norm(
            embedding
        )

        if norm < 1e-10:
            return embedding

        return embedding / norm

    # ==========================================================
    # Properties
    # ==========================================================

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def similarity_threshold(self) -> float:
        return float(
            self._metadata.get(
                "similarity_threshold",
                0.72,
            )
        )

    @property
    def model_info(self) -> dict:
        return {
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "embedding_dim": self.embedding_dim,
            "backend": (
                "tflite-runtime"
                if TFLITE_RUNTIME
                else "tensorflow"
            ),
        }
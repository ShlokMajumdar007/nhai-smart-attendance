"""
mobilefacenet.py
TensorFlow Lite inference wrapper for MobileFaceNet.
Extracts 128-dim L2-normalized face embeddings.
"""

import numpy as np
import os
import logging
from typing import Optional, List
import json

logger = logging.getLogger(__name__)

# Try TFLite runtime first (smaller), fall back to full TF
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_RUNTIME = True
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    TFLITE_RUNTIME = False


class MobileFaceNet:
    """
    TFLite inference engine for MobileFaceNet.

    Input : (1, 112, 112, 3) float32 normalized to [-1, 1]
    Output: (1, 128) float32 L2-normalized embedding
    """

    MODEL_FILE = "mobilefacenet.tflite"
    METADATA_FILE = "metadata.json"

    def __init__(self, model_dir: str, num_threads: int = 4):
        self.model_dir = model_dir
        self.num_threads = num_threads
        self._interpreter: Optional[object] = None
        self._input_details = None
        self._output_details = None
        self._metadata = {}

        self._load_metadata()
        self._load_model()

    def _load_metadata(self):
        meta_path = os.path.join(self.model_dir, self.METADATA_FILE)
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self._metadata = json.load(f)
        logger.info(f"Model metadata: {self._metadata.get('model_name', 'unknown')}")

    def _load_model(self):
        model_path = os.path.join(self.model_dir, self.MODEL_FILE)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"TFLite model not found at {model_path}. "
                "Download MobileFaceNet from https://github.com/sirius-ai/MobileFaceNet_TF "
                "and convert to .tflite format."
            )

        if TFLITE_RUNTIME:
            self._interpreter = tflite.Interpreter(
                model_path=model_path,
                num_threads=self.num_threads,
            )
        else:
            self._interpreter = tflite.Interpreter(
                model_path=model_path,
                num_threads=self.num_threads,
            )

        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        logger.info(
            f"MobileFaceNet loaded | "
            f"input={self._input_details[0]['shape']} "
            f"output={self._output_details[0]['shape']}"
        )

    def get_embedding(self, face: np.ndarray) -> np.ndarray:
        """
        Run inference on a single preprocessed face.

        Args:
            face: np.ndarray shape (112, 112, 3) dtype float32 in [-1, 1]

        Returns:
            embedding: np.ndarray shape (128,) L2-normalized
        """
        if face.shape != (112, 112, 3):
            raise ValueError(f"Expected (112,112,3), got {face.shape}")

        input_data = np.expand_dims(face, axis=0).astype(np.float32)

        self._interpreter.set_tensor(
            self._input_details[0]["index"], input_data
        )
        self._interpreter.invoke()

        output = self._interpreter.get_tensor(
            self._output_details[0]["index"]
        )  # (1, 128)

        embedding = output[0]
        return self._l2_normalize(embedding)

    def get_batch_embeddings(self, faces: List[np.ndarray]) -> np.ndarray:
        """
        Get embeddings for a list of faces.
        Returns shape (N, 128).
        """
        embeddings = []
        for face in faces:
            emb = self.get_embedding(face)
            embeddings.append(emb)
        return np.array(embeddings)

    def get_average_embedding(self, faces: List[np.ndarray]) -> np.ndarray:
        """
        Average multiple embeddings and re-normalize.
        Used for multi-frame verification and enrollment.
        """
        if not faces:
            raise ValueError("No faces provided")

        batch = self.get_batch_embeddings(faces)
        avg = batch.mean(axis=0)
        return self._l2_normalize(avg)

    @staticmethod
    def _l2_normalize(embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            return embedding
        return embedding / norm

    @property
    def embedding_dim(self) -> int:
        return int(self._metadata.get("embedding_dim", 128))

    @property
    def similarity_threshold(self) -> float:
        return float(self._metadata.get("similarity_threshold", 0.65))
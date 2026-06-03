import sys
from pathlib import Path

import numpy as np

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.embedding.mobilefacenet import MobileFaceNet


def main():
    print("Loading model...")

    model = MobileFaceNet(
        model_dir="ai/models"
    )

    print("\nModel Info")
    print("-" * 40)
    print("Embedding Dimension:", model.embedding_dim)

    if hasattr(model, "model_info"):
        print(model.model_info)

    # Create dummy face input
    face = np.random.uniform(
        -1.0,
        1.0,
        (112, 112, 3)
    ).astype(np.float32)

    print("\nRunning inference...")

    embedding = model.get_embedding(face)

    print("\nResults")
    print("-" * 40)
    print("Embedding Shape :", embedding.shape)
    print("Embedding Dtype :", embedding.dtype)
    print("Embedding Norm  :", np.linalg.norm(embedding))
    print("First 10 Values :")
    print(embedding[:10])


if __name__ == "__main__":
    main()
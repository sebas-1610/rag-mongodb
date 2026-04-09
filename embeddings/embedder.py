"""
embeddings/embedder.py
======================
S — Única responsabilidad: generar embeddings de texto.
D — El pipeline depende de EmbedderBase, no de SentenceTransformer directamente.
O — Para usar otro modelo: crear nueva clase que herede EmbedderBase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


# =============================================================================
# INTERFAZ BASE  (Principio I y D)
# =============================================================================


class EmbedderBase(ABC):
    """I — Interfaz mínima: solo embed() y dimension."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos.
        Retorna lista de vectores float (no numpy, para serializar a MongoDB).
        """
        ...

    @abstractmethod
    def embed_single(self, text: str) -> List[float]:
        """Embedding para un solo texto (útil en queries)."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...


# =============================================================================
# IMPLEMENTACIÓN PRINCIPAL — all-MiniLM-L6-v2
# =============================================================================


class MiniLMEmbedder(EmbedderBase):
    """
    S — Solo envuelve SentenceTransformer y normaliza la salida.
    O — Para cambiar modelo: crear MultilingualEmbedder sin tocar esta clase.

    Modelo: all-MiniLM-L6-v2
      Dimensión : 384
      Velocidad : ~14K oraciones/seg en CPU
      Uso       : texto en inglés / español básico
    """

    def __init__(self, model_id: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        print(f"[Embedder] Cargando '{model_id}'...")
        self._model = SentenceTransformer(model_id)
        self._model_id = model_id
        self._batch_size = batch_size
        print(
            f"[Embedder] Listo. Dimensión: {self._model.get_sentence_embedding_dimension()}"
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors: np.ndarray = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,  # normalizar para similitud coseno
        )
        # Convertir a lista de Python (serializable a MongoDB / JSON)
        return vectors.tolist()

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_id


# =============================================================================
# IMPLEMENTACIÓN MULTILINGÜE (para texto en español)
# =============================================================================


class MultilingualEmbedder(EmbedderBase):
    """
    O — Nueva implementación sin modificar MiniLMEmbedder.
    Modelo: paraphrase-multilingual-MiniLM-L12-v2
      Soporta 50+ idiomas incluyendo español.
      Más preciso para textos en español que all-MiniLM-L6-v2.
    """

    def __init__(
        self,
        model_id: str = "paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 32,
    ):
        from sentence_transformers import SentenceTransformer

        print(f"[Embedder] Cargando modelo multilingüe '{model_id}'...")
        self._model = SentenceTransformer(model_id)
        self._model_id = model_id
        self._batch_size = batch_size

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_id


# =============================================================================
# FACTORY
# =============================================================================


class EmbedderFactory:
    """O — Para añadir modelo nuevo: solo agregar entrada al dict."""

    @staticmethod
    def create(model_name: str = "all-MiniLM-L6-v2") -> EmbedderBase:
        modelos = {
            "all-MiniLM-L6-v2": lambda: MiniLMEmbedder("all-MiniLM-L6-v2"),
            "paraphrase-multilingual-MiniLM-L12-v2": lambda: MultilingualEmbedder(),
        }
        if model_name not in modelos:
            raise ValueError(f"Modelo '{model_name}' no registrado.")
        return modelos[model_name]()


# =============================================================================
# SINGLETON GLOBAL (se carga una sola vez al iniciar la app)
# =============================================================================
_embedder_instance: EmbedderBase | None = None


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> EmbedderBase:
    """
    Retorna la instancia singleton del embedder.
    D — La app entera usa este getter, no instancia EmbedderBase directamente.
    """
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = EmbedderFactory.create(model_name)
    return _embedder_instance

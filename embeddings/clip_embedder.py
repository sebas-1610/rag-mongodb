"""
embeddings/clip_embedder.py
============================
S — Embedder para imágenes y texto usando CLIP (OpenAI).
D — Depende de sentence-transformers y PIL.
O — Extiende EmbedderBase para mantener compatibilidad.

CLIP (Contrastive Language-Image Pretraining) mapea imágenes y texto
al mismo espacio vectorial (512 dimensiones para ViT-B-32).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union
from abc import abstractmethod

from embeddings.embedder import EmbedderBase


class CLIPEmbedder(EmbedderBase):
    """
    Embedder CLIP para imágenes y texto.
    Model: clip-ViT-B-32 (512 dimensiones)
    
    Permite:
    - Texto → Vector (para búsqueda texto→imagen)
    - Imagen → Vector (para búsqueda imagen→imagen)
    - Zero-shot classification (para describir imágenes)
    """

    def __init__(self, model_id: str = "clip-ViT-B-32", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        print(f"[CLIP Embedder] Cargando '{model_id}'...")
        self._model = SentenceTransformer(model_id)
        self._model_id = model_id
        self._batch_size = batch_size
        self._dim = 512  # CLIP ViT-B-32 output dimension
        print(f"[CLIP Embedder] Listo. Dimensión: {self._dim}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Codifica textos al espacio vectorial de CLIP.
        Útil para búsqueda texto→imagen.
        """
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_single(self, text: str) -> List[float]:
        """Codifica un solo texto."""
        return self.embed([text])[0]

    def embed_image(self, image) -> List[float]:
        """
        Codifica una imagen al espacio vectorial de CLIP.
        
        Args:
            image: PIL.Image.Image, str (ruta), o Path
        
        Returns:
            List de 512 floats
        """
        from PIL import Image as PILImage

        if isinstance(image, (str, Path)):
            image = PILImage.open(image)

        vector = self._model.encode(
            image,
            normalize_embeddings=True,
        )
        return vector.tolist()

    def embed_images(self, images: list) -> List[List[float]]:
        """
        Codifica múltiples imágenes en batch.
        
        Args:
            images: Lista de PIL.Image.Image objects
        
        Returns:
            Lista de vectores de 512 floats
        """
        if not images:
            return []
        vectors = self._model.encode(
            images,
            batch_size=self._batch_size,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def describe_image(self, image, candidate_labels: List[str]) -> List[dict]:
        """
        Clasifica una imagen usando zero-shot con CLIP.
        
        Args:
            image: PIL.Image.Image, str (ruta), o Path
            candidate_labels: Lista de etiquetas candidatas
        
        Returns:
            Lista de dicts ordenada por confianza:
            [{"label": str, "score": float}, ...]
        """
        import torch
        from PIL import Image as PILImage

        if isinstance(image, (str, Path)):
            image = PILImage.open(image)

        # Codificar imagen
        image_embedding = self._model.encode(image)

        # Codificar etiquetas como texto (mejora precisión)
        text_embeddings = self._model.encode([
            f"a photo of a {label}" for label in candidate_labels
        ])

        # Calcular similitud coseno
        from sentence_transformers import util
        similarities = util.cos_sim(image_embedding, text_embeddings)[0]

        # Aplicar softmax con temperatura (CLIP default: 100)
        temperature = 100.0
        probabilities = torch.softmax(similarities * temperature, dim=0)

        # Ordenar por confianza
        results = [
            {"label": label, "score": round(prob.item(), 4)}
            for label, prob in zip(candidate_labels, probabilities)
        ]
        results.sort(key=lambda x: x["score"], reverse=True)

        return results

    def get_top_labels(self, image, labels: List[str], top_k: int = 3) -> List[str]:
        """
        Obtiene las top-k etiquetas más probables para una imagen.
        Útil para generar descripciones automáticas.
        """
        results = self.describe_image(image, labels)
        return [r["label"] for r in results[:top_k]]

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_id


# =============================================================================
# SINGLETON para CLIP
# =============================================================================

_clip_instance: Optional[CLIPEmbedder] = None


def get_clip_embedder(model_id: str = "clip-ViT-B-32") -> CLIPEmbedder:
    """
    Retorna instancia singleton del CLIPEmbedder.
    """
    global _clip_instance
    if _clip_instance is None:
        _clip_instance = CLIPEmbedder(model_id)
    return _clip_instance

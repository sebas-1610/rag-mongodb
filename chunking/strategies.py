"""
chunking/strategies.py
======================
Las 3 estrategias requeridas por el proyecto, con SOLID estricto.
Reutiliza la arquitectura del taller_rag.py y la adapta a MongoDB.

S — Cada clase tiene una sola responsabilidad
O — Nuevas estrategias se agregan heredando ChunkingStrategy
L — Todas las estrategias son intercambiables en el pipeline
I — Interfaces mínimas: ChunkingStrategy y SentenceDetector
D — El pipeline depende de ChunkingStrategy, no de implementaciones
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import math
from collections import Counter


# =============================================================================
# MODELOS DE DATOS
# =============================================================================


@dataclass
class ChunkDoc:
    """
    Representa exactamente el documento que va a MongoDB.
    Estructura según sección 5.3 del proyecto.

    S — Solo almacena datos de un chunk, sin lógica de negocio.
    """

    doc_id: str  # ObjectId del documento padre (como str)
    chunk_index: int
    estrategia_chunking: str  # "fixed" | "sentence-aware" | "semantic"
    chunk_texto: str
    embedding: List[float]  # se completa en la capa de embeddings
    modelo: str
    tokens: int
    fecha_ingesta: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_mongo(self) -> Dict[str, Any]:
        """Serializa para inserción en MongoDB."""
        return {
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "estrategia_chunking": self.estrategia_chunking,
            "chunk_texto": self.chunk_texto,
            "embedding": self.embedding,
            "modelo": self.modelo,
            "tokens": self.tokens,
            "fecha_ingesta": self.fecha_ingesta,
            "metadata": self.metadata,
        }


# =============================================================================
# INTERFACES  (Principio I)
# =============================================================================


class SentenceDetector(ABC):
    """I — Interfaz mínima para detección de oraciones."""

    @abstractmethod
    def detect(self, text: str) -> List[str]: ...

    @abstractmethod
    def name(self) -> str: ...


class ChunkingStrategy(ABC):
    """
    I — Interfaz mínima para estrategias de chunking.
    L — Todas las implementaciones son intercambiables.
    """

    @abstractmethod
    def split(self, text: str) -> List[str]: ...

    @abstractmethod
    def strategy_name(self) -> str: ...

    def build_chunks(
        self, text: str, doc_id: str, modelo: str = "all-MiniLM-L6-v2"
    ) -> List[ChunkDoc]:
        """
        Template method: split → construir ChunkDoc por cada fragmento.
        O — Las subclases solo sobreescriben split(), no este método.
        """
        fragmentos = self.split(text)
        return [
            ChunkDoc(
                doc_id=doc_id,
                chunk_index=i,
                estrategia_chunking=self.strategy_name(),
                chunk_texto=frag,
                embedding=[],  # se llena en embeddings/embedder.py
                modelo=modelo,
                tokens=len(frag.split()),
                metadata={"posicion": i, "total_chunks": len(fragmentos)},
            )
            for i, frag in enumerate(fragmentos)
            if frag.strip()
        ]


# =============================================================================
# ESTRATEGIA 1 — FIXED-SIZE  (chunk_size=256, overlap=32)
# =============================================================================


class FixedSizeStrategy(ChunkingStrategy):
    """
    S — Solo divide texto en ventanas de tamaño fijo con overlap.
    Parámetros recomendados por el proyecto: size=256, overlap=32.
    """

    def __init__(self, chunk_size: int = 256, overlap: int = 32):
        self._chunk_size = chunk_size
        self._overlap = overlap

    def strategy_name(self) -> str:
        return "fixed"

    def split(self, text: str) -> List[str]:
        palabras = text.split()
        chunks: List[str] = []
        inicio = 0

        while inicio < len(palabras):
            fin = min(inicio + self._chunk_size, len(palabras))
            chunks.append(" ".join(palabras[inicio:fin]))
            inicio += self._chunk_size - self._overlap

        return chunks


# =============================================================================
# ESTRATEGIA 2 — SENTENCE-AWARE  (max 5 oraciones, overlap 1)
# =============================================================================


class RegexSentenceDetector(SentenceDetector):
    """
    S — Solo detecta límites de oración con regex + lista de abreviaturas.
    Simula spaCy cuando no hay internet para descargar modelos.
    """

    _ABREVIATURAS = {
        "dr",
        "dra",
        "sr",
        "sra",
        "etc",
        "p.ej",
        "ej",
        "ing",
        "lic",
        "prof",
        "fig",
        "vs",
        "núm",
        "art",
        "pág",
    }

    def detect(self, text: str) -> List[str]:
        candidatos = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚ\"])", text)
        oraciones: List[str] = []
        buffer = ""

        for candidato in candidatos:
            buffer = (buffer + " " + candidato).strip() if buffer else candidato
            ultima = re.split(r"\s+", buffer)[-1].rstrip(".!?,;").lower()
            if ultima not in self._ABREVIATURAS:
                oraciones.append(buffer)
                buffer = ""

        if buffer:
            oraciones.append(buffer)

        return [o.strip() for o in oraciones if o.strip()]

    def name(self) -> str:
        return "regex-sentence-detector"


class NLTKSentenceDetector(SentenceDetector):
    """
    O — Implementación con NLTK real cuando hay internet.
    D — Intercambiable con RegexSentenceDetector sin tocar SentenceAwareStrategy.
    """

    def __init__(self, language: str = "spanish"):
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize

        self._tokenize = sent_tokenize
        self._language = language

    def detect(self, text: str) -> List[str]:
        return [
            s.strip()
            for s in self._tokenize(text, language=self._language)
            if s.strip()
        ]

    def name(self) -> str:
        return f"nltk-{self._language}"


class SentenceAwareStrategy(ChunkingStrategy):
    """
    S — Agrupa oraciones sin romper ninguna.
    D — Recibe el detector de oraciones como dependencia inyectada.
    Parámetros del proyecto: max_sentences=5, overlap=1.
    """

    def __init__(
        self,
        detector: SentenceDetector,
        max_sentences: int = 5,
        overlap_sentences: int = 1,
    ):
        self._detector = detector
        self._max = max_sentences
        self._overlap = overlap_sentences

    def strategy_name(self) -> str:
        return "sentence-aware"

    def split(self, text: str) -> List[str]:
        oraciones = self._detector.detect(text)
        chunks: List[str] = []
        i = 0

        while i < len(oraciones):
            grupo = oraciones[i : i + self._max]
            chunks.append(" ".join(grupo))
            i += self._max - self._overlap

        return chunks


# =============================================================================
# ESTRATEGIA 3 — SEMANTIC  (umbral coseno 0.75–0.85)
# =============================================================================


class TFIDFVectorizer:
    """
    S — Solo vectoriza texto con TF-IDF casero.
    Reemplazable por SentenceTransformer (ver embeddings/embedder.py).
    """

    def __init__(self, corpus: List[str]):
        self._vocab = sorted(
            {
                w
                for doc in corpus
                for w in re.findall(r"\b[a-záéíóúüñ]{3,}\b", doc.lower())
            }
        )
        n = len(corpus)
        self._idf = {
            w: math.log(
                (n + 1)
                / (
                    sum(
                        1
                        for d in corpus
                        if w in re.findall(r"\b[a-záéíóúüñ]{3,}\b", d.lower())
                    )
                    + 1
                )
            )
            + 1
            for w in self._vocab
        }

    def transform(self, text: str) -> List[float]:
        tokens = re.findall(r"\b[a-záéíóúüñ]{3,}\b", text.lower())
        freq = Counter(tokens)
        total = len(tokens) or 1
        return [(freq.get(w, 0) / total) * self._idf.get(w, 1.0) for w in self._vocab]


class CosineSimilarity:
    """S — Solo calcula similitud coseno entre dos vectores."""

    @staticmethod
    def compute(v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a**2 for a in v1))
        n2 = math.sqrt(sum(b**2 for b in v2))
        return dot / (n1 * n2) if n1 and n2 else 0.0


class SemanticStrategy(ChunkingStrategy):
    """
    S — Detecta quiebres semánticos comparando embeddings de oraciones adyacentes.
    D — Recibe detector de oraciones como dependencia.
    Umbral recomendado por el proyecto: 0.75–0.85.
    """

    def __init__(
        self,
        detector: SentenceDetector,
        threshold: float = 0.75,
    ):
        self._detector = detector
        self._threshold = threshold
        self._similarity = CosineSimilarity()

    def strategy_name(self) -> str:
        return "semantic"

    def split(self, text: str) -> List[str]:
        oraciones = self._detector.detect(text)
        if len(oraciones) <= 1:
            return oraciones

        # Vectorizar con TF-IDF
        vectorizer = TFIDFVectorizer(oraciones)
        vectors = [vectorizer.transform(o) for o in oraciones]

        # Detectar quiebres: similitud < umbral → nuevo chunk
        breaks = [0]
        for i in range(len(vectors) - 1):
            sim = self._similarity.compute(vectors[i], vectors[i + 1])
            if sim < self._threshold:
                breaks.append(i + 1)

        breaks.append(len(oraciones))

        # Agrupar oraciones entre quiebres
        return [
            " ".join(oraciones[breaks[i] : breaks[i + 1]])
            for i in range(len(breaks) - 1)
        ]


# =============================================================================
# FACTORY  (Principio O — extensible sin modificar código existente)
# =============================================================================


class ChunkingStrategyFactory:
    """
    O — Para añadir una nueva estrategia: solo agregar una entrada al dict.
    S — Solo crea estrategias, no las ejecuta.
    """

    @staticmethod
    def create(
        strategy: str,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        sentence_max: int = 5,
        sentence_overlap: int = 1,
        semantic_threshold: float = 0.75,
        use_nltk: bool = False,
    ) -> ChunkingStrategy:
        """
        strategy: "fixed" | "sentence-aware" | "semantic"
        """
        # Elegir detector de oraciones (D — inyección de dependencias)
        try:
            detector: SentenceDetector = (
                NLTKSentenceDetector() if use_nltk else RegexSentenceDetector()
            )
        except Exception:
            detector = RegexSentenceDetector()

        strategies = {
            "fixed": lambda: FixedSizeStrategy(chunk_size, chunk_overlap),
            "sentence-aware": lambda: SentenceAwareStrategy(
                detector, sentence_max, sentence_overlap
            ),
            "semantic": lambda: SemanticStrategy(detector, semantic_threshold),
        }

        if strategy not in strategies:
            raise ValueError(
                f"Estrategia '{strategy}' no válida. "
                f"Opciones: {list(strategies.keys())}"
            )

        return strategies[strategy]()

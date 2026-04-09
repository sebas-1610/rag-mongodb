"""
retrieval/search.py
===================
S — Única responsabilidad: buscar chunks relevantes en MongoDB.
D — Depende de EmbedderBase y MongoDBClient como abstracciones.
O — Para añadir nueva estrategia de búsqueda: heredar SearchEngine.

Tipos de búsqueda:
    - Vectorial  : $vectorSearch (Atlas Vector Search)
    - Híbrida    : $vectorSearch + filtros $match por metadata
    - Texto completo: $text (Atlas Search)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from bson import ObjectId

from embeddings.embedder import EmbedderBase
from database.mongodb import mongo


# =============================================================================
# RESULTADO DE BÚSQUEDA
# =============================================================================


@dataclass
class SearchResult:
    """S — Solo representa un resultado de búsqueda con su score."""

    chunk_id: str
    doc_id: str
    chunk_texto: str
    estrategia_chunking: str
    score: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_texto": self.chunk_texto,
            "estrategia_chunking": self.estrategia_chunking,
            "score": self.score,
            "metadata": self.metadata,
        }


# =============================================================================
# INTERFAZ BASE  (Principios I y D)
# =============================================================================


class SearchEngine(ABC):
    """I — Interfaz mínima para motores de búsqueda."""

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        estrategia: Optional[str] = None,
        filtros: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]: ...


# =============================================================================
# BÚSQUEDA VECTORIAL con Atlas $vectorSearch
# =============================================================================


class VectorSearchEngine(SearchEngine):
    """
    S — Solo ejecuta búsqueda vectorial con $vectorSearch de Atlas.
    Requiere el índice 'vector_index' creado en scripts/init_db.py.

    Pipeline de agregación:
      $vectorSearch → $match (filtros opcionales) → $project
    """

    def __init__(self, embedder: EmbedderBase):
        self._embedder = embedder

    async def search(
        self,
        query: str,
        top_k: int = 5,
        estrategia: Optional[str] = None,
        filtros: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:

        query_vector = self._embedder.embed_single(query)

        # Construir filtro pre-vectorial (se aplica ANTES del kNN)
        pre_filter: Dict[str, Any] = {}
        if estrategia:
            pre_filter["estrategia_chunking"] = estrategia
        if filtros:
            pre_filter.update(filtros)

        # Pipeline de agregación Atlas
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": top_k * 10,  # candidatos antes del reranking
                    "limit": top_k,
                    **({"filter": pre_filter} if pre_filter else {}),
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "doc_id": 1,
                    "chunk_texto": 1,
                    "estrategia_chunking": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        resultados: List[SearchResult] = []
        async for doc in mongo.chunks.aggregate(pipeline):
            resultados.append(
                SearchResult(
                    chunk_id=str(doc["_id"]),
                    doc_id=str(doc.get("doc_id", "")),
                    chunk_texto=doc["chunk_texto"],
                    estrategia_chunking=doc.get("estrategia_chunking", ""),
                    score=doc.get("score", 0.0),
                    metadata=doc.get("metadata", {}),
                )
            )

        return resultados


# =============================================================================
# BÚSQUEDA HÍBRIDA: vectorial + filtros SQL-style
# =============================================================================


class HybridSearchEngine(SearchEngine):
    """
    S — Combina búsqueda vectorial con filtros de metadata (fecha, idioma, etc.).
    O — Extiende VectorSearchEngine sin modificarlo.
    D — Recibe VectorSearchEngine como dependencia.

    Ejemplo de uso:
      engine.search("energías renovables", filtros={"idioma": "es", "categoria": "tech"})
    """

    def __init__(self, vector_engine: VectorSearchEngine):
        self._vector = vector_engine

    async def search(
        self,
        query: str,
        top_k: int = 5,
        estrategia: Optional[str] = None,
        filtros: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        # Delegar en el motor vectorial con los filtros adicionales
        return await self._vector.search(
            query=query,
            top_k=top_k,
            estrategia=estrategia,
            filtros=filtros,
        )

    async def search_by_category(
        self,
        query: str,
        categoria: str,
        top_k: int = 5,
        estrategia: Optional[str] = None,
    ) -> List[SearchResult]:
        return await self.search(
            query=query,
            top_k=top_k,
            estrategia=estrategia,
            filtros={"metadata.categoria": categoria},
        )

    async def search_by_date_range(
        self,
        query: str,
        desde: str,
        hasta: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Busca en un rango de fechas (ISO format: "2024-01-01").
        Combina filtro temporal con similitud vectorial.
        """
        from datetime import datetime

        filtros = {
            "fecha_ingesta": {
                "$gte": datetime.fromisoformat(desde),
                "$lte": datetime.fromisoformat(hasta),
            }
        }
        return await self.search(query=query, top_k=top_k, filtros=filtros)


# =============================================================================
# BÚSQUEDA POR ESTRATEGIA (para el experimento de chunking)
# =============================================================================


class ChunkingExperimentSearch:
    """
    S — Ejecuta la misma consulta en las 3 estrategias y compara resultados.
    Útil para el experimento de comparación del proyecto (sección 5.4).
    """

    def __init__(self, engine: SearchEngine):
        self._engine = engine

    async def compare_strategies(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, List[SearchResult]]:
        """
        Ejecuta la misma query en fixed, sentence-aware y semantic.
        Retorna dict con resultados por estrategia.
        """
        estrategias = ["fixed", "sentence-aware", "semantic"]
        resultados: Dict[str, List[SearchResult]] = {}

        for est in estrategias:
            resultados[est] = await self._engine.search(
                query=query,
                top_k=top_k,
                estrategia=est,
            )

        return resultados

    async def generate_comparison_report(
        self,
        queries: List[str],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Genera el reporte comparativo para las 10 consultas del proyecto (5.4).
        """
        reporte = []
        for query in queries:
            resultados = await self.compare_strategies(query, top_k)
            reporte.append(
                {
                    "query": query,
                    "resultados": {
                        est: [r.to_dict() for r in res]
                        for est, res in resultados.items()
                    },
                    "resumen": {
                        est: {
                            "total": len(res),
                            "score_promedio": (
                                round(sum(r.score for r in res) / len(res), 4)
                                if res
                                else 0
                            ),
                        }
                        for est, res in resultados.items()
                    },
                }
            )

        return reporte

"""
ingestion/pipeline.py
=====================
S — Orquesta el pipeline de ingesta: documento → chunks → embeddings → MongoDB.
D — Depende de abstracciones: ChunkingStrategy, EmbedderBase, MongoDBClient.
O — Para añadir un paso al pipeline (ej. OCR): extender sin modificar ingest().
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from bson import ObjectId
from tqdm import tqdm

from chunking.strategies import ChunkingStrategy, ChunkingStrategyFactory, ChunkDoc
from embeddings.embedder import EmbedderBase, get_embedder
from database.mongodb import mongo
from config.settings import get_settings


# =============================================================================
# MODELOS DE DOCUMENTO PADRE
# =============================================================================


class DocumentoPadre:
    """
    S — Representa un documento completo antes de chunkear.
    Estructura de la colección 'documentos'.
    """

    def __init__(
        self,
        titulo: str,
        contenido_texto: str,
        categoria: str,
        idioma: str = "es",
        url_imagen: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.titulo = titulo
        self.contenido_texto = contenido_texto
        self.categoria = categoria
        self.idioma = idioma
        self.url_imagen = url_imagen
        self.metadata = metadata or {}
        self.fecha_ingesta = datetime.now(timezone.utc)

    def to_mongo(self) -> Dict[str, Any]:
        return {
            "titulo": self.titulo,
            "contenido_texto": self.contenido_texto,
            "categoria": self.categoria,
            "idioma": self.idioma,
            "url_imagen": self.url_imagen,
            "metadata": self.metadata,
            "fecha_ingesta": self.fecha_ingesta,
        }


# =============================================================================
# PIPELINE DE INGESTA
# =============================================================================


class IngestionPipeline:
    """
    S — Orquesta la ingesta completa de un documento.
    D — Recibe estrategia y embedder como dependencias inyectadas.

    Flujo:
      documento → guardar en 'documentos' → chunkear → embeddings → guardar chunks
    """

    def __init__(
        self,
        estrategia: ChunkingStrategy,
        embedder: EmbedderBase,
    ):
        self._estrategia = estrategia
        self._embedder = embedder

    async def ingest_document(self, doc: DocumentoPadre) -> str:
        """
        Ingesta un documento completo.
        Retorna el _id del documento insertado en MongoDB.
        """
        # 1. Guardar documento padre
        result = await mongo.documentos.insert_one(doc.to_mongo())
        doc_id = str(result.inserted_id)

        # 2. Generar chunks
        chunks = self._estrategia.build_chunks(
            text=doc.contenido_texto,
            doc_id=doc_id,
            modelo=self._embedder.model_name,
        )

        if not chunks:
            return doc_id

        # 3. Generar embeddings en batch (eficiente)
        textos = [c.chunk_texto for c in chunks]
        embeddings = self._embedder.embed(textos)

        # 4. Asignar embeddings a cada chunk
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # 5. Insertar chunks en MongoDB
        docs_mongo = [c.to_mongo() for c in chunks]
        # Convertir doc_id string a ObjectId para referencia
        for d in docs_mongo:
            d["doc_id"] = ObjectId(d["doc_id"])

        await mongo.chunks.insert_many(docs_mongo)

        return doc_id

    async def ingest_batch(self, documentos: List[DocumentoPadre]) -> List[str]:
        """Ingesta múltiples documentos con barra de progreso."""
        ids: List[str] = []
        for doc in tqdm(
            documentos, desc=f"Ingesta [{self._estrategia.strategy_name()}]"
        ):
            doc_id = await self.ingest_document(doc)
            ids.append(doc_id)
        return ids


# =============================================================================
# LOADER DE JSON  (para el dataset)
# =============================================================================


class JSONDatasetLoader:
    """
    S — Solo carga documentos desde un archivo JSON.
    Formato esperado: lista de objetos con campos del DocumentoPadre.
    """

    @staticmethod
    def load(filepath: str | Path) -> List[DocumentoPadre]:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {filepath}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        documentos: List[DocumentoPadre] = []
        for item in data:
            documentos.append(
                DocumentoPadre(
                    titulo=item["titulo"],
                    contenido_texto=item["contenido_texto"],
                    categoria=item.get("categoria", "general"),
                    idioma=item.get("idioma", "es"),
                    url_imagen=item.get("url_imagen"),
                    metadata=item.get("metadata", {}),
                )
            )

        return documentos


# =============================================================================
# SCRIPT DE INGESTA COMPLETA (las 3 estrategias sobre el mismo dataset)
# =============================================================================


async def ingest_all_strategies(dataset_path: str = "data/dataset.json") -> None:
    """
    Ingesta el dataset con las 3 estrategias de chunking.
    Así se puede comparar estrategias en el experimento del proyecto.
    """
    cfg = get_settings()
    embedder = get_embedder(cfg.embedding_model)
    documentos = JSONDatasetLoader.load(dataset_path)

    print(f"\nDataset cargado: {len(documentos)} documentos")
    print(f"Modelo de embedding: {embedder.model_name}\n")

    estrategias = [
        ChunkingStrategyFactory.create(
            "fixed",
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        ),
        ChunkingStrategyFactory.create(
            "sentence-aware",
            sentence_max=cfg.sentence_max,
            sentence_overlap=cfg.sentence_overlap,
        ),
        ChunkingStrategyFactory.create(
            "semantic",
            semantic_threshold=cfg.semantic_threshold,
        ),
    ]

    for estrategia in estrategias:
        print(f"── Estrategia: {estrategia.strategy_name()} ──")
        pipeline = IngestionPipeline(estrategia, embedder)
        ids = await pipeline.ingest_batch(documentos)
        print(f"   {len(ids)} documentos procesados\n")

    # Resumen por estrategia
    print("\n── Resumen en MongoDB ──")
    for nombre in ["fixed", "sentence-aware", "semantic"]:
        count = await mongo.chunks.count_documents({"estrategia_chunking": nombre})
        print(f"  {nombre:<20} → {count:>5} chunks")


if __name__ == "__main__":
    import asyncio
    from database.mongodb import connect_db, disconnect_db

    async def main():
        await connect_db()
        await ingest_all_strategies()
        await disconnect_db()

    asyncio.run(main())

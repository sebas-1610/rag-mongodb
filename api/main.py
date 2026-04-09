"""
api/main.py
===========
FastAPI con los endpoints requeridos por el proyecto:
    POST /search → búsqueda híbrida o vectorial
    POST /rag    → genera respuesta usando contexto de MongoDB + LLM
    GET  /health → estado del sistema
    GET  /stats  → estadísticas de chunks por estrategia
    POST /experiment → ejecuta el experimento de chunking (10 consultas × 3 estrategias)

S — Cada endpoint tiene una sola responsabilidad.
D — Depende de abstracciones inyectadas via FastAPI Depends.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from config.settings import get_settings
from database.mongodb import connect_db, disconnect_db, mongo
from embeddings.embedder import get_embedder
from retrieval.search import (
    VectorSearchEngine,
    HybridSearchEngine,
    ChunkingExperimentSearch,
    SearchResult,
)
from rag.pipeline import RAGPipeline, build_rag_pipeline


# =============================================================================
# LIFESPAN — conectar/desconectar MongoDB al iniciar/apagar la app
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    cfg = get_settings()
    await connect_db()
    embedder = get_embedder(cfg.embedding_model)

    # Construir engines y pipeline una sola vez (Singleton)
    vector_engine = VectorSearchEngine(embedder)
    hybrid_engine = HybridSearchEngine(vector_engine)
    rag = build_rag_pipeline(hybrid_engine)

    # Guardar en app.state para acceso en endpoints
    app.state.vector_engine = vector_engine
    app.state.hybrid_engine = hybrid_engine
    app.state.rag_pipeline = rag
    app.state.experiment = ChunkingExperimentSearch(hybrid_engine)

    print("[API] Lista en http://localhost:8000")
    yield

    # Shutdown
    await disconnect_db()


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="Sistema RAG con MongoDB Atlas",
    description="Pipeline RAG completo: MongoDB + sentence-transformers + Groq Llama 3.1",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# SCHEMAS DE REQUEST / RESPONSE (Pydantic)
# =============================================================================


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Consulta de búsqueda")
    top_k: int = Field(5, ge=1, le=20)
    estrategia: Optional[str] = Field(
        None,
        description="Filtrar por estrategia: 'fixed' | 'sentence-aware' | 'semantic'",
    )
    filtros: Optional[Dict[str, Any]] = Field(
        None, description="Filtros adicionales de metadata (ej. {'categoria': 'tech'})"
    )


class SearchResponse(BaseModel):
    query: str
    total: int
    resultados: List[Dict[str, Any]]


class RAGRequest(BaseModel):
    pregunta: str = Field(..., min_length=5, description="Pregunta en lenguaje natural")
    estrategia: Optional[str] = Field(
        None, description="Estrategia de chunking a usar en el retrieval"
    )
    filtros: Optional[Dict[str, Any]] = None
    top_k: int = Field(5, ge=1, le=10)


class RAGResponse(BaseModel):
    pregunta: str
    respuesta: str
    chunks_usados: int
    estrategia_usada: Optional[str]
    modelo_llm: str
    contexto: List[Dict[str, Any]]


class ExperimentRequest(BaseModel):
    queries: List[str] = Field(
        ...,
        min_length=1,
        description="Lista de consultas para el experimento (recomendado: 10)",
    )
    top_k: int = Field(5, ge=1, le=10)


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.get("/health", tags=["Sistema"])
async def health_check():
    """Verifica el estado del sistema y la conexión a MongoDB."""
    try:
        await mongo.db.command("ping")
        return {"status": "ok", "mongodb": "conectado"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MongoDB no disponible: {e}")


@app.get("/stats", tags=["Sistema"])
async def get_stats():
    """Estadísticas de chunks almacenados por estrategia."""
    pipeline = [
        {
            "$group": {
                "_id": "$estrategia_chunking",
                "total_chunks": {"$sum": 1},
                "tokens_promedio": {"$avg": "$tokens"},
                "tokens_min": {"$min": "$tokens"},
                "tokens_max": {"$max": "$tokens"},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    stats = []
    async for doc in mongo.chunks.aggregate(pipeline):
        stats.append(
            {
                "estrategia": doc["_id"],
                "total_chunks": doc["total_chunks"],
                "tokens_promedio": round(doc.get("tokens_promedio", 0), 1),
                "tokens_min": doc.get("tokens_min", 0),
                "tokens_max": doc.get("tokens_max", 0),
            }
        )

    total_docs = await mongo.documentos.count_documents({})
    return {
        "total_documentos": total_docs,
        "chunks_por_estrategia": stats,
    }


@app.post("/search", response_model=SearchResponse, tags=["Búsqueda"])
async def search(request: SearchRequest):
    """
    Búsqueda vectorial o híbrida en MongoDB Atlas.
    Combina similitud semántica con filtros opcionales de metadata.
    """
    engine: HybridSearchEngine = app.state.hybrid_engine

    try:
        resultados: List[SearchResult] = await engine.search(
            query=request.query,
            top_k=request.top_k,
            estrategia=request.estrategia,
            filtros=request.filtros,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {e}")

    return SearchResponse(
        query=request.query,
        total=len(resultados),
        resultados=[r.to_dict() for r in resultados],
    )


@app.post("/rag", response_model=RAGResponse, tags=["RAG"])
async def rag_query(request: RAGRequest):
    """
    Genera una respuesta usando RAG:
    recupera contexto de MongoDB y lo pasa al LLM (Groq Llama 3.1).
    """
    pipeline: RAGPipeline = app.state.rag_pipeline

    try:
        resultado = await pipeline.query(
            pregunta=request.pregunta,
            estrategia=request.estrategia,
            filtros=request.filtros,
            top_k=request.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en pipeline RAG: {e}")

    return RAGResponse(
        pregunta=resultado.query,
        respuesta=resultado.answer,
        chunks_usados=len(resultado.chunks_usados),
        estrategia_usada=resultado.estrategia,
        modelo_llm=resultado.modelo_llm,
        contexto=[c.to_dict() for c in resultado.chunks_usados],
    )


@app.post("/experiment", tags=["Experimento Chunking"])
async def run_experiment(request: ExperimentRequest):
    """
    Ejecuta las consultas de prueba sobre las 3 estrategias de chunking.
    Genera el reporte comparativo requerido en la sección 5.4 del proyecto.
    """
    experiment: ChunkingExperimentSearch = app.state.experiment

    try:
        reporte = await experiment.generate_comparison_report(
            queries=request.queries,
            top_k=request.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en experimento: {e}")

    # Calcular resumen global
    resumen_global = {}
    for est in ["fixed", "sentence-aware", "semantic"]:
        scores = [
            entrada["resumen"][est]["score_promedio"]
            for entrada in reporte
            if est in entrada["resumen"]
        ]
        resumen_global[est] = {
            "score_promedio_global": (
                round(sum(scores) / len(scores), 4) if scores else 0
            ),
        }

    return {
        "total_queries": len(request.queries),
        "resumen_global": resumen_global,
        "detalle": reporte,
    }


# =============================================================================
# CONSULTAS DE PRUEBA OBLIGATORIAS (sección 7 del proyecto)
# =============================================================================

CONSULTAS_PRUEBA = [
    "¿Qué documentos hablan sobre sostenibilidad ambiental?",
    "Artículos en español sobre tecnología publicados en 2024",
    "Explica las principales tendencias en energías renovables",
    "¿Cómo funcionan los transformers en inteligencia artificial?",
    "¿Qué es RAG y para qué sirve?",
    "Diferencia entre machine learning y deep learning",
    "¿Cómo se implementa búsqueda vectorial en bases de datos?",
    "¿Qué son los embeddings y cómo se generan?",
    "Ventajas y desventajas de MongoDB frente a SQL",
    "¿Qué estrategia de chunking es mejor para documentos técnicos?",
]


@app.get("/experiment/default", tags=["Experimento Chunking"])
async def run_default_experiment():
    """
    Ejecuta el experimento con las 10 consultas de prueba predefinidas.
    Listo para la entrega del proyecto (sección 5.4 y caso 5 de sección 7).
    """
    experiment: ChunkingExperimentSearch = app.state.experiment

    try:
        reporte = await experiment.generate_comparison_report(
            queries=CONSULTAS_PRUEBA,
            top_k=5,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"queries": CONSULTAS_PRUEBA, "reporte": reporte}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    cfg = get_settings()
    uvicorn.run(
        "api.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=True,
    )

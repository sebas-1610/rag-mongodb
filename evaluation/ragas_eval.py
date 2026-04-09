"""
evaluation/ragas_eval.py
========================
Evaluación automática del pipeline RAG con RAGAS (nota extra).
Métricas: faithfulness, answer_relevancy, context_recall.

S — Solo evalúa, no hace retrieval ni genera respuestas.
D — Recibe el pipeline RAG como dependencia, no lo instancia.

Instalación: pip install ragas datasets
Docs: https://docs.ragas.io
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from database.mongodb import mongo
from rag.pipeline import RAGPipeline


# =============================================================================
# DATASET DE EVALUACIÓN
# =============================================================================


@dataclass
class EvalSample:
    """
    S — Representa un par (pregunta, ground_truth) para evaluación.
    El proyecto pide mínimo 20 pares.
    """

    pregunta: str
    ground_truth: str
    estrategia: Optional[str] = None


# 20 pares de evaluación sobre dominio IA/Tecnología
EVAL_DATASET: List[EvalSample] = [
    EvalSample(
        "¿Qué es RAG en inteligencia artificial?",
        "RAG (Retrieval-Augmented Generation) es una técnica que combina "
        "recuperación de información con generación de texto mediante LLMs.",
    ),
    EvalSample(
        "¿Para qué sirven los embeddings?",
        "Los embeddings son representaciones vectoriales densas que capturan "
        "el significado semántico de texto, permitiendo búsqueda por similitud.",
    ),
    EvalSample(
        "¿Qué es el chunking en sistemas RAG?",
        "El chunking es el proceso de dividir documentos largos en fragmentos "
        "más pequeños antes de vectorizarlos para mejorar la especificidad del retrieval.",
    ),
    EvalSample(
        "¿Qué ventaja tiene semantic chunking sobre fixed-size?",
        "Semantic chunking respeta las fronteras temáticas del texto, produciendo "
        "fragmentos cohesivos, mientras fixed-size puede cortar ideas a la mitad.",
    ),
    EvalSample(
        "¿Qué es pgvector?",
        "pgvector es una extensión de PostgreSQL que añade soporte nativo para "
        "vectores y búsqueda por similitud coseno mediante índices HNSW.",
    ),
    EvalSample(
        "¿Cómo funciona la arquitectura transformer?",
        "Los transformers usan mecanismos de atención multi-head para capturar "
        "relaciones entre tokens en toda la secuencia de entrada simultáneamente.",
    ),
    EvalSample(
        "¿Qué es MongoDB Atlas Vector Search?",
        "Atlas Vector Search es una funcionalidad de MongoDB Atlas que permite "
        "búsqueda por similitud vectorial usando índices knnVector sobre embeddings.",
    ),
    EvalSample(
        "¿Cuál es la diferencia entre BERT y GPT?",
        "BERT usa encoders bidireccionales para entender contexto, mientras GPT "
        "usa decoders autorregresivos para generación de texto secuencial.",
    ),
    EvalSample(
        "¿Qué es sentence-aware chunking?",
        "Sentence-aware chunking divide el texto respetando los límites de oraciones, "
        "evitando cortar frases a la mitad y manteniendo cohesión textual.",
    ),
    EvalSample(
        "¿Qué modelo de embedding usa el sistema?",
        "El sistema usa all-MiniLM-L6-v2, un modelo de sentence-transformers con "
        "dimensión 384 que genera embeddings semánticos eficientes.",
    ),
    EvalSample(
        "¿Qué es el overlap en chunking?",
        "El overlap es la cantidad de tokens compartidos entre chunks consecutivos "
        "para evitar la pérdida de contexto en las fronteras de fragmentación.",
    ),
    EvalSample(
        "¿Qué LLM se usa en el pipeline RAG?",
        "El sistema usa Llama 3.1 a través de la API de Groq, que ofrece una "
        "cuota gratuita generosa y respuestas muy rápidas.",
    ),
    EvalSample(
        "¿Qué es un índice HNSW?",
        "HNSW (Hierarchical Navigable Small World) es un algoritmo de indexación "
        "para búsqueda aproximada de vecinos más cercanos en espacios vectoriales.",
    ),
    EvalSample(
        "¿Cuándo usar fixed-size chunking?",
        "Fixed-size chunking es ideal para textos homogéneos sin estructura clara, "
        "logs y datos técnicos donde la velocidad de indexación es prioritaria.",
    ),
    EvalSample(
        "¿Qué mide faithfulness en RAGAS?",
        "Faithfulness mide si la respuesta del LLM es factualmente consistente con "
        "el contexto recuperado, penalizando información inventada (alucinaciones).",
    ),
    EvalSample(
        "¿Qué mide answer relevancy en RAGAS?",
        "Answer relevancy evalúa si la respuesta generada es pertinente a la pregunta "
        "original, penalizando respuestas incompletas o fuera de tema.",
    ),
    EvalSample(
        "¿Qué es context recall en RAGAS?",
        "Context recall mide si el contexto recuperado cubre la respuesta esperada, "
        "requiere un ground truth manual para calcular la cobertura.",
    ),
    EvalSample(
        "¿Qué es NoSQL y cómo difiere de SQL?",
        "NoSQL son bases de datos que no usan esquemas relacionales fijos, "
        "permitiendo documentos flexibles y escalabilidad horizontal más sencilla.",
    ),
    EvalSample(
        "¿Qué es el Aggregation Framework de MongoDB?",
        "Es un sistema de procesamiento de datos en etapas ($match, $group, $project, "
        "$lookup) que permite transformar y analizar documentos en colecciones.",
    ),
    EvalSample(
        "¿Qué ventaja tiene embedding vs referencing en MongoDB?",
        "Embedding incrusta documentos relacionados para lecturas rápidas sin joins, "
        "mientras referencing usa ObjectId para datos grandes o compartidos.",
    ),
]


# =============================================================================
# EVALUADOR RAGAS
# =============================================================================


class RAGASEvaluator:
    """
    S — Solo ejecuta la evaluación RAGAS y almacena resultados en MongoDB.
    D — Recibe el pipeline RAG como dependencia inyectada.
    """

    def __init__(self, rag_pipeline: RAGPipeline):
        self._pipeline = rag_pipeline

    async def evaluate(
        self,
        samples: List[EvalSample],
        estrategia: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta el pipeline RAG sobre cada muestra y evalúa con RAGAS.
        Almacena los scores en la colección 'evaluaciones' de MongoDB.
        """
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall

        print(f"[RAGAS] Evaluando {len(samples)} muestras...")

        # 1. Generar respuestas con el pipeline RAG
        preguntas, respuestas, contextos, ground_truths = [], [], [], []

        for sample in samples:
            resultado = await self._pipeline.query(
                pregunta=sample.pregunta,
                estrategia=estrategia or sample.estrategia,
            )
            preguntas.append(sample.pregunta)
            respuestas.append(resultado.answer)
            contextos.append([c.chunk_texto for c in resultado.chunks_usados])
            ground_truths.append(sample.ground_truth)

        # 2. Construir dataset RAGAS
        eval_dict = {
            "question": preguntas,
            "answer": respuestas,
            "contexts": contextos,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(eval_dict)

        # 3. Evaluar con RAGAS
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_recall],
        )

        scores = result.to_pandas()

        # 4. Guardar resultados en MongoDB (colección 'evaluaciones')
        docs_mongo = []
        for i, sample in enumerate(samples):
            row = scores.iloc[i]
            doc = {
                "id_consulta": f"eval_{i:03d}",
                "pregunta": sample.pregunta,
                "respuesta": respuestas[i],
                "faithfulness": float(row.get("faithfulness", 0)),
                "answer_relevancy": float(row.get("answer_relevancy", 0)),
                "context_recall": float(row.get("context_recall", 0)),
                "estrategia": estrategia or "todas",
                "modelo_eval": "ragas-v0.1",
                "fecha": datetime.now(timezone.utc),
            }
            docs_mongo.append(doc)

        await mongo.evaluaciones.insert_many(docs_mongo)
        print(f"[RAGAS] {len(docs_mongo)} evaluaciones guardadas en MongoDB")

        # 5. Calcular promedios globales
        promedios = {
            "faithfulness": float(scores["faithfulness"].mean()),
            "answer_relevancy": float(scores["answer_relevancy"].mean()),
            "context_recall": float(scores["context_recall"].mean()),
        }

        return {
            "estrategia": estrategia or "todas",
            "total_samples": len(samples),
            "promedios": promedios,
            "detalle": docs_mongo,
        }

    async def evaluate_all_strategies(self) -> Dict[str, Any]:
        """
        Evalúa las 3 estrategias con el mismo dataset.
        Genera el reporte comparativo completo.
        """
        resultados: Dict[str, Any] = {}

        for estrategia in ["fixed", "sentence-aware", "semantic"]:
            print(f"\n[RAGAS] Evaluando estrategia: {estrategia}")
            resultado = await self.evaluate(EVAL_DATASET, estrategia=estrategia)
            resultados[estrategia] = resultado["promedios"]

        # Determinar mejor estrategia por promedio general
        def score_total(est: str) -> float:
            p = resultados[est]
            return (p["faithfulness"] + p["answer_relevancy"] + p["context_recall"]) / 3

        mejor = max(resultados.keys(), key=score_total)

        return {
            "resultados_por_estrategia": resultados,
            "mejor_estrategia": mejor,
            "score_mejor": round(score_total(mejor), 4),
        }


# =============================================================================
# ENDPOINT ADICIONAL para FastAPI (se registra en api/main.py)
# =============================================================================


async def get_ragas_report() -> Dict[str, Any]:
    """
    Retorna el último reporte de evaluación almacenado en MongoDB.
    Para agregar al endpoint GET /evaluation en api/main.py.
    """
    pipeline = [
        {
            "$group": {
                "_id": "$estrategia",
                "faithfulness_avg": {"$avg": "$faithfulness"},
                "answer_relevancy_avg": {"$avg": "$answer_relevancy"},
                "context_recall_avg": {"$avg": "$context_recall"},
                "total": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    reporte = []
    async for doc in mongo.evaluaciones.aggregate(pipeline):
        reporte.append(
            {
                "estrategia": doc["_id"],
                "faithfulness": round(doc["faithfulness_avg"], 4),
                "answer_relevancy": round(doc["answer_relevancy_avg"], 4),
                "context_recall": round(doc["context_recall_avg"], 4),
                "total_evaluaciones": doc["total"],
            }
        )

    return {"reporte_ragas": reporte}


# =============================================================================
# SCRIPT STANDALONE
# =============================================================================

if __name__ == "__main__":
    import asyncio
    from database.mongodb import connect_db, disconnect_db
    from embeddings.embedder import get_embedder
    from retrieval.search import VectorSearchEngine, HybridSearchEngine
    from rag.pipeline import build_rag_pipeline
    from config.settings import get_settings

    async def main():
        await connect_db()
        cfg = get_settings()
        embedder = get_embedder(cfg.embedding_model)
        engine = HybridSearchEngine(VectorSearchEngine(embedder))
        pipeline = build_rag_pipeline(engine)

        evaluator = RAGASEvaluator(pipeline)
        resultado = await evaluator.evaluate_all_strategies()

        print("\n=== REPORTE RAGAS ===")
        for est, scores in resultado["resultados_por_estrategia"].items():
            print(f"\n  {est}:")
            for metrica, valor in scores.items():
                print(f"    {metrica:<25} {valor:.4f}")
        print(f"\n  Mejor estrategia: {resultado['mejor_estrategia']}")
        print(f"  Score total:      {resultado['score_mejor']}")

        await disconnect_db()

    asyncio.run(main())

"""
rag/pipeline.py
===============
S — Orquesta el pipeline RAG completo: query → retrieval → prompt → LLM → respuesta.
D — Depende de SearchEngine y LLMClient como abstracciones.
O — Para cambiar LLM: crear nueva clase que herede LLMClientBase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from retrieval.search import SearchEngine, SearchResult
from config.settings import get_settings


# =============================================================================
# INTERFAZ LLM  (Principios I y D)
# =============================================================================


class LLMClientBase(ABC):
    """I — Interfaz mínima para cualquier LLM."""

    @abstractmethod
    async def complete(self, prompt: str, system: str = "") -> str: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...


# =============================================================================
# CLIENTE GROQ — Llama 3.1 (gratis)
# =============================================================================


class GroqClient(LLMClientBase):
    """
    S — Solo envuelve la API de Groq.
    Modelo: llama-3.1-8b-instant (rápido, cuota generosa gratuita).
    """

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        from groq import AsyncGroq

        self._client = AsyncGroq(api_key=api_key)
        self._model = model

    async def complete(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,  # bajo para respuestas más fieles al contexto
        )
        return response.choices[0].message.content

    @property
    def model_name(self) -> str:
        return self._model


# =============================================================================
# PROMPT BUILDER  (S — solo construye prompts)
# =============================================================================


class PromptBuilder:
    """
    S — Única responsabilidad: construir el prompt con contexto recuperado.
    O — Extensible para diferentes tipos de prompt sin modificar el pipeline.
    """

    SYSTEM_PROMPT = """Eres un asistente experto en tecnología e inteligencia artificial.
Responde siempre en español, de forma precisa y basándote ÚNICAMENTE en el contexto proporcionado.
Si la información no está en el contexto, indica claramente que no dispones de esa información.
No inventes ni especules más allá del contexto."""

    @staticmethod
    def build(query: str, resultados: List[SearchResult]) -> str:
        """
        Construye el prompt RAG con los chunks recuperados como contexto.
        Cada chunk incluye su índice y estrategia para trazabilidad.
        """
        if not resultados:
            return (
                f"No se encontró contexto relevante para responder la siguiente pregunta:\n\n"
                f"Pregunta: {query}\n\n"
                f"Por favor indica que no tienes información suficiente."
            )

        contexto_bloques = []
        for i, r in enumerate(resultados, 1):
            contexto_bloques.append(
                f"[Fragmento {i} | estrategia: {r.estrategia_chunking} | score: {r.score:.3f}]\n"
                f"{r.chunk_texto}"
            )

        contexto = "\n\n".join(contexto_bloques)

        return (
            f"Usa el siguiente contexto para responder la pregunta.\n\n"
            f"=== CONTEXTO ===\n{contexto}\n\n"
            f"=== PREGUNTA ===\n{query}\n\n"
            f"=== RESPUESTA ==="
        )


# =============================================================================
# RESPUESTA RAG
# =============================================================================


@dataclass
class RAGResponse:
    """S — Solo almacena la respuesta completa del pipeline RAG."""

    query: str
    answer: str
    chunks_usados: List[SearchResult]
    estrategia: Optional[str]
    modelo_llm: str
    modelo_embedding: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "chunks_usados": [c.to_dict() for c in self.chunks_usados],
            "estrategia": self.estrategia,
            "modelo_llm": self.modelo_llm,
            "modelo_embedding": self.modelo_embedding,
        }


# =============================================================================
# PIPELINE RAG PRINCIPAL
# =============================================================================


class RAGPipeline:
    """
    S — Orquesta: retrieval → prompt → LLM → respuesta.
    D — Depende de SearchEngine y LLMClientBase, no de implementaciones.
    O — Para añadir reranking: extender sin modificar esta clase.
    """

    def __init__(
        self,
        search_engine: SearchEngine,
        llm_client: LLMClientBase,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
    ):
        self._search = search_engine
        self._llm = llm_client
        self._embedding_model = embedding_model
        self._top_k = top_k
        self._prompt_builder = PromptBuilder()

    async def query(
        self,
        pregunta: str,
        estrategia: Optional[str] = None,
        filtros: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """
        Pipeline completo:
            1. Buscar chunks relevantes
            2. Construir prompt con contexto
            3. Llamar al LLM
            4. Retornar respuesta estructurada
        """
        k = top_k or self._top_k

        # Paso 1: Retrieval
        chunks = await self._search.search(
            query=pregunta,
            top_k=k,
            estrategia=estrategia,
            filtros=filtros,
        )

        # Paso 2: Prompt Engineering
        prompt = self._prompt_builder.build(pregunta, chunks)

        # Paso 3: LLM
        respuesta = await self._llm.complete(
            prompt=prompt,
            system=PromptBuilder.SYSTEM_PROMPT,
        )

        return RAGResponse(
            query=pregunta,
            answer=respuesta,
            chunks_usados=chunks,
            estrategia=estrategia,
            modelo_llm=self._llm.model_name,
            modelo_embedding=self._embedding_model,
        )


# =============================================================================
# FACTORY  (para construir el pipeline desde settings)
# =============================================================================


def build_rag_pipeline(search_engine: SearchEngine) -> RAGPipeline:
    """
    D — Construye el pipeline completo con todas las dependencias inyectadas.
    Se llama una sola vez en el startup de la API.
    """
    cfg = get_settings()
    llm = GroqClient(api_key=cfg.groq_api_key, model=cfg.groq_model)
    return RAGPipeline(
        search_engine=search_engine,
        llm_client=llm,
        embedding_model=cfg.embedding_model,
        top_k=5,
    )

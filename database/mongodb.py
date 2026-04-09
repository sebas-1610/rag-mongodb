"""
database/mongodb.py
===================
S — Única responsabilidad: gestionar la conexión y exponer colecciones.
D — El resto del proyecto depende de esta abstracción, no de pymongo directo.

Colecciones:
    - documentos  : documento padre (texto completo + metadata)
    - chunks      : fragmentos con embedding y estrategia_chunking
    - evaluaciones: scores RAGAS por consulta
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, TEXT
from pymongo.operations import SearchIndexModel

from config.settings import get_settings

# ── Nombres de colecciones (constantes para evitar typos) ────────────────────
COL_DOCUMENTOS = "documentos"
COL_CHUNKS = "chunks"
COL_EVALUACIONES = "evaluaciones"


class MongoDBClient:
    """
    S — Solo gestiona el ciclo de vida de la conexión y expone colecciones.
    O — Se puede extender para añadir nuevas colecciones sin modificar connect().
    """

    _instance: MongoDBClient | None = None
    _client: AsyncIOMotorClient | None = None
    _db: AsyncIOMotorDatabase | None = None

    # ── Singleton ────────────────────────────────────────────────────────────
    def __new__(cls) -> MongoDBClient:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ── Conexión ─────────────────────────────────────────────────────────────
    async def connect(self) -> None:
        cfg = get_settings()
        self._client = AsyncIOMotorClient(cfg.mongodb_uri)
        self._db = self._client[cfg.mongodb_db]
        # Verificar conexión
        await self._client.admin.command("ping")
        print(f"[MongoDB] Conectado a '{cfg.mongodb_db}'")

    async def disconnect(self) -> None:
        if self._client:
            self._client.close()
            print("[MongoDB] Conexión cerrada")

    # ── Propiedades de colecciones ───────────────────────────────────────────
    @property
    def db(self) -> AsyncIOMotorDatabase:
        if self._db is None:
            raise RuntimeError("Llamar connect() primero")
        return self._db

    @property
    def documentos(self):
        return self.db[COL_DOCUMENTOS]

    @property
    def chunks(self):
        return self.db[COL_CHUNKS]

    @property
    def evaluaciones(self):
        return self.db[COL_EVALUACIONES]


# ── Singleton global ─────────────────────────────────────────────────────────
mongo = MongoDBClient()


# ── FastAPI lifespan helpers ─────────────────────────────────────────────────
async def connect_db() -> None:
    await mongo.connect()


async def disconnect_db() -> None:
    await mongo.disconnect()

"""
scripts/init_db.py
==================
Crea índices y schema validation en MongoDB Atlas.
Ejecutar UNA sola vez después de configurar el cluster:

    python scripts/init_db.py

S — Solo inicializa la base de datos, no hace nada más.
"""

import asyncio
from pymongo import ASCENDING, TEXT
from motor.motor_asyncio import AsyncIOMotorClient

from config.settings import get_settings


# ── Schema Validation ────────────────────────────────────────────────────────

SCHEMA_DOCUMENTOS = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["titulo", "contenido_texto", "categoria", "fecha_ingesta"],
        "properties": {
            "titulo": {"bsonType": "string"},
            "contenido_texto": {"bsonType": "string"},
            "categoria": {"bsonType": "string"},
            "idioma": {"bsonType": "string"},
            "fecha_ingesta": {"bsonType": "date"},
            "url_imagen": {"bsonType": ["string", "null"]},
            "metadata": {"bsonType": "object"},
        },
    }
}

SCHEMA_CHUNKS = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": [
            "doc_id",
            "chunk_index",
            "estrategia_chunking",
            "chunk_texto",
            "embedding",
            "modelo",
            "fecha_ingesta",
        ],
        "properties": {
            "doc_id": {"bsonType": "objectId"},
            "chunk_index": {"bsonType": "int"},
            "estrategia_chunking": {
                "bsonType": "string",
                "enum": ["fixed", "sentence-aware", "semantic"],
            },
            "chunk_texto": {"bsonType": "string"},
            "embedding": {"bsonType": "array"},
            "modelo": {"bsonType": "string"},
            "fecha_ingesta": {"bsonType": "date"},
            "tokens": {"bsonType": "int"},
            "metadata": {"bsonType": "object"},
        },
    }
}

SCHEMA_EVALUACIONES = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["id_consulta", "faithfulness", "answer_relevancy", "fecha"],
        "properties": {
            "id_consulta": {"bsonType": "string"},
            "faithfulness": {"bsonType": "double"},
            "answer_relevancy": {"bsonType": "double"},
            "context_recall": {"bsonType": ["double", "null"]},
            "modelo_eval": {"bsonType": "string"},
            "fecha": {"bsonType": "date"},
        },
    }
}


async def init_database() -> None:
    cfg = get_settings()
    client = AsyncIOMotorClient(cfg.mongodb_uri)
    db = client[cfg.mongodb_db]

    print(f"Inicializando base de datos '{cfg.mongodb_db}'...\n")

    # ── 1. Crear colecciones con schema validation ────────────────────────────
    colecciones_existentes = await db.list_collection_names()

    for nombre, schema in [
        ("documentos", SCHEMA_DOCUMENTOS),
        ("chunks", SCHEMA_CHUNKS),
        ("evaluaciones", SCHEMA_EVALUACIONES),
    ]:
        if nombre not in colecciones_existentes:
            await db.create_collection(
                nombre,
                validator=schema,
                validationAction="warn",  # warn = no bloquea, solo avisa
            )
            print(f"  ✓ Colección '{nombre}' creada con schema validation")
        else:
            # Actualizar validación si ya existe
            await db.command(
                {"collMod": nombre, "validator": schema, "validationAction": "warn"}
            )
            print(f"  ✓ Colección '{nombre}' ya existe — schema actualizado")

    # ── 2. Índices en documentos ─────────────────────────────────────────────
    print("\nCreando índices en 'documentos'...")

    await db["documentos"].create_index(
        [("fecha_ingesta", ASCENDING), ("idioma", ASCENDING)], name="idx_fecha_idioma"
    )
    await db["documentos"].create_index(
        [("contenido_texto", TEXT)],
        name="idx_texto_completo",
        default_language="spanish",
    )
    await db["documentos"].create_index(
        [("categoria", ASCENDING)], name="idx_categoria"
    )
    print("  ✓ idx_fecha_idioma (compuesto)")
    print("  ✓ idx_texto_completo (texto completo)")
    print("  ✓ idx_categoria")

    # ── 3. Índices en chunks ─────────────────────────────────────────────────
    print("\nCreando índices en 'chunks'...")

    await db["chunks"].create_index(
        [("doc_id", ASCENDING), ("estrategia_chunking", ASCENDING)],
        name="idx_doc_estrategia",
    )
    await db["chunks"].create_index(
        [("estrategia_chunking", ASCENDING), ("chunk_index", ASCENDING)],
        name="idx_estrategia_index",
    )
    print("  ✓ idx_doc_estrategia (compuesto)")
    print("  ✓ idx_estrategia_index (compuesto)")

    # ── 4. Índice vectorial en Atlas (knnVector) ─────────────────────────────
    print("\nNOTA: El índice vectorial (knnVector) debe crearse manualmente")
    print("en MongoDB Atlas UI o con la API de Atlas Search.\n")
    print("Configuración del índice vectorial para 'chunks':")
    print(
        """
    {
        "mappings": {
            "dynamic": true,
            "fields": {
            "embedding": {
                "dimensions": 384,
                "similarity": "cosine",
                "type": "knnVector"
                }
            }
        }
    }
    """
    )
    print("  Pasos:")
    print("  1. Ir a Atlas → tu cluster → Search → Create Search Index")
    print("  2. Seleccionar 'JSON Editor'")
    print("  3. Colección: chunks")
    print("  4. Pegar el JSON de arriba")
    print("  5. Nombre del índice: 'vector_index'")

    # ── 5. Índice en evaluaciones ────────────────────────────────────────────
    print("\nCreando índices en 'evaluaciones'...")
    await db["evaluaciones"].create_index([("fecha", ASCENDING)], name="idx_fecha_eval")
    print("  ✓ idx_fecha_eval")

    client.close()
    print("\n✅ Inicialización completada.")


if __name__ == "__main__":
    asyncio.run(init_database())

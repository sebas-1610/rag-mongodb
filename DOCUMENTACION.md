# Documentación del Proyecto RAG con MongoDB

## Resumen del Proyecto

Este es un sistema **RAG (Retrieval-Augmented Generation)** completo que usa:
- **MongoDB Atlas** con búsqueda vectorial (Vector Search)
- ** sentence-transformers** (modelo `all-MiniLM-L6-v2`, 384 dimensiones)
- **Groq API** (Llama 3.1 8B) como LLM para generar respuestas
- **FastAPI** como APIREST

El sistema permite almacenar documentos, dividirlos en chunks (fragmentos) usando 3 estrategias diferentes, buscar por similitud vectorial, y generar respuestas usando un LLM con contexto recuperado.

---

## Estructura del Proyecto

```
rag-mongodb/
├── config/              # Configuración centralizada
├── database/            # Conexión a MongoDB
├── chunking/           # Estrategias para dividir documentos
├── embeddings/         # Modelos para generar vectores
├── ingestion/          # Pipeline de ingesta de documentos
├── retrieval/          # Motores de búsqueda
├── rag/                # Pipeline RAG completo
├── api/                # Endpoints FastAPI
├── evaluation/         # Evaluación con RAGAS
├── scripts/            # Utilidades (init_db)
├── data/               # Dataset JSON de ejemplo
└── .env                # Variables de entorno
```

---

## Descripción de Archivos

### 1. `config/settings.py`

**¿Para qué sirve?**  
Centraliza todas las configuraciones del proyecto usando `pydantic-settings`. Lee variables de entorno desde `.env`.

**Variables que define:**
- `mongodb_uri` — URI de conexión a MongoDB Atlas
- `mongodb_db` — Nombre de la base de datos (por defecto: `rag_ia`)
- `groq_api_key` — Clave de API de Groq
- `groq_model` — Modelo LLM a usar (`llama-3.1-8b-instant`)
- `embedding_model` — Modelo de embedding (`all-MiniLM-L6-v2`)
- `embedding_dim` — Dimensiones del embedding (384)
- `chunk_size`, `chunk_overlap` — Parámetros de chunking fijo
- `sentence_max`, `sentence_overlap` — Parámetros de chunking por oraciones
- `semantic_threshold` — Umbral para chunking semántico (0.75)

**Métodos:**
- `get_settings()` — Retorna singleton de configuración (se cachea)

**¿Dónde van los datos?**  
Este archivo solo lee, no guarda nada. Los valores vienen de `.env`.

---

### 2. `database/mongodb.py`

**¿Para qué sirve?**  
Gestiona la conexión a MongoDB y expone las colecciones.

**Colecciones que usa:**
| Colección | Propósito |
|-----------|-----------|
| `documentos` | Documentos originales (título, contenido, categoría, etc.) |
| `chunks` | Fragmentos de documentos con sus embeddings |
| `evaluaciones` | Scores de evaluación RAGAS |

**Clases y métodos:**
- `MongoDBClient` — Clase singleton para gestionar conexión
  - `connect()` — Conecta a MongoDB Atlas
  - `disconnect()` — Cierra la conexión
  - `documentos` — Property: retorna colección de documentos
  - `chunks` — Property: retorna colección de chunks
  - `evaluaciones` — Property: retorna colección de evaluaciones

**¿Dónde van los datos?**
- Los documentos se guardan en la colección `documentos`
- Los chunks se guardan en la colección `chunks`
- Las evaluaciones se guardan en la colección `evaluaciones`

---

### 3. `chunking/strategies.py`

**¿Para qué sirve?**  
Implementa las 3 estrategias para dividir documentos en chunks.

**Estrategias disponibles:**

| Estrategia | Descripción | Parámetros |
|------------|-------------|-------------|
| `fixed` | Divide en chunks de tamaño fijo con overlap | `chunk_size=256`, `overlap=32` |
| `sentence-aware` | Agrupa oraciones (max 5 por chunk) | `max_sentences=5`, `overlap=1` |
| `semantic` | Detecta cambios de tema usando similitud TF-IDF | `threshold=0.75` |

**Clases principales:**
- `ChunkingStrategy` (abstracta) — Interfaz base para estrategias
- `ChunkingStrategyFactory` — Fabrica estrategias según el nombre
- `FixedSizeStrategy` — Chunking de tamaño fijo
- `SentenceAwareStrategy` — Chunking por oraciones
- `SemanticStrategy` — Chunking semántico usando TF-IDF
- `ChunkDoc` — Dataclass que representa un chunk para MongoDB

**Métodos:**
- `split(text)` — Divide el texto según la estrategia
- `build_chunks(text, doc_id, modelo)` — Genera lista de ChunkDoc
- `to_mongo()` — Serializa el chunk para guardar en MongoDB

**Estructura de un Chunk en MongoDB:**
```json
{
  "doc_id": "ObjectId",
  "chunk_index": 0,
  "estrategia_chunking": "fixed",
  "chunk_texto": "Contenido del fragmento...",
  "embedding": [0.12, -0.34, ...],  // 384 valores
  "modelo": "all-MiniLM-L6-v2",
  "tokens": 45,
  "fecha_ingesta": "ISODate",
  "metadata": {}
}
```

---

### 4. `embeddings/embedder.py`

**¿Para qué sirve?**  
Genera embeddings (vectores) para textos usando sentence-transformers.

**Modelos disponibles:**
- `all-MiniLM-L6-v2` — Rápido, 384 dimensiones (inglés/español básico)
- `paraphrase-multilingual-MiniLM-L12-v2` — Multilingüe (50+ idiomas)

**Clases:**
- `EmbedderBase` — Interfaz abstracta
- `MiniLMEmbedder` — Implementación con modelo estándar
- `MultilingualEmbedder` — Implementación multilingüe
- `EmbedderFactory` — Fabrica embedders
- `get_embedder()` — Retorna instancia singleton

**Métodos:**
- `embed(texts: List[str])` — Genera embeddings para varios textos
- `embed_single(text)` — Genera embedding para un solo texto
- `dimension` — Property: retorna dimensiones del embedding
- `model_name` — Property: retorna nombre del modelo

**¿Dónde van los datos?**  
Los embeddings se guardan en el campo `embedding` de cada documento en la colección `chunks`.

---

### 5. `ingestion/pipeline.py`

**¿Para qué sirve?**  
Orquesta el pipeline completo de ingesta: documentos → chunks → embeddings → MongoDB.

**Clases:**
- `DocumentoPadre` — Representa un documento antes de dividirlo
- `IngestionPipeline` — Pipeline de ingesta de un documento
- `JSONDatasetLoader` — Carga documentos desde JSON

**Métodos:**
- `ingest_document(doc)` — Procesa un documento completo:
  1. Guarda documento en colección `documentos`
  2. Genera chunks según estrategia
  3. Genera embeddings para cada chunk
  4. Guarda chunks en colección `chunks`
  5. Retorna `_id` del documento

- `ingest_batch(documentos)` — Procesa varios documentos con progress bar
- `load(filepath)` — Carga documentos desde JSON
- `ingest_all_strategies()` — Ejecuta ingesta con las 3 estrategias

**¿Dónde van los datos?**
- Documentos originales → colección `documentos`
- Chunks con embeddings → colección `chunks`

**Formato del documento en JSON (`data/dataset.json`):**
```json
[
  {
    "titulo": "Título del documento",
    "contenido_texto": "Contenido completo...",
    "categoria": "tecnologia",
    "idioma": "es",
    "url_imagen": null,
    "metadata": {}
  }
]
```

---

### 6. `retrieval/search.py`

**¿Para qué sirve?**  
Busca chunks relevantes en MongoDB usando búsqueda vectorial.

**Clases:**
- `SearchEngine` — Interfaz abstracta para motores de búsqueda
- `VectorSearchEngine` — Búsqueda vectorial con `$vectorSearch` de Atlas
- `HybridSearchEngine` — Búsqueda vectorial con filtros
- `ChunkingExperimentSearch` — Experimento comparativo de estrategias

**Métodos:**
- `search(query, top_k, estrategia, filtros)` — Busca chunks relevantes
  - Convierte query a embedding
  - Ejecuta `$vectorSearch` en MongoDB
  - Aplica filtros opcionales (estrategia, categoría, idioma, etc.)
  - Retorna lista de `SearchResult`

- `compare_strategies(query, top_k)` — Ejecuta query en las 3 estrategias
- `generate_comparison_report(queries, top_k)` — Genera reporte comparativo

**¿Dónde obtiene los datos?**  
Lee de la colección `chunks` en MongoDB.

**Estructura de SearchResult:**
```json
{
  "chunk_id": "ObjectId",
  "doc_id": "ObjectId",
  "chunk_texto": "Fragmento recuperado...",
  "estrategia_chunking": "fixed",
  "score": 0.85,
  "metadata": {}
}
```

---

### 7. `rag/pipeline.py`

**¿Para qué sirve?**  
Orquesta el pipeline RAG completo: query → retrieval → prompt → LLM → respuesta.

**Clases:**
- `LLMClientBase` — Interfaz para clientes LLM
- `GroqClient` — Cliente para Groq API (Llama 3.1)
- `PromptBuilder` — Construye prompts con contexto
- `RAGPipeline` — Pipeline completo
- `build_rag_pipeline()` — Fabrica el pipeline completo

**Métodos:**
- `complete(prompt, system)` — Llama al LLM y retorna texto
- `build(query, resultados)` — Construye prompt con contexto recuperado
- `query(pregunta, estrategia, filtros, top_k)` — Ejecuta pipeline RAG completo:
  1. Busca chunks relevantes (retrieval)
  2. Construye prompt con contexto
  3. Llama al LLM (Groq)
  4. Retorna respuesta + chunks usados

**¿Dónde obtiene los datos?**  
- Contextos de `retrieval/search.py`
- LLM de Groq API

**¿Qué retorna?**
```json
{
  "pregunta": "¿Qué es RAG?",
  "respuesta": "RAG es una técnica que...",
  "chunks_usados": 5,
  "estrategia_usada": "semantic",
  "modelo_llm": "llama-3.1-8b-instant",
  "contexto": [chunks retrieved]
}
```

---

### 8. `api/main.py`

**¿Para qué sirve?**  
Expone los endpoints FastAPI para interactuar con el sistema.

**Endpoints disponibles:**

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/health` | Verifica conexión a MongoDB |
| GET | `/stats` | Estadísticas de chunks por estrategia |
| POST | `/search` | Búsqueda vectorial/híbrida |
| POST | `/rag` | Query RAG completo |
| GET | `/experiment/default` | Experimento con 10 queries predefinidas |
| POST | `/experiment` | Experimento con queries personalizadas |

**Request de `/search`:**
```json
{
  "query": "¿Qué es RAG?",
  "top_k": 5,
  "estrategia": "semantic",
  "filtros": {"categoria": "tecnologia"}
}
```

**Request de `/rag`:**
```json
{
  "pregunta": "¿Cómo funciona la búsqueda vectorial?",
  "estrategia": "semantic",
  "top_k": 5
}
```

---

### 9. `scripts/init_db.py`

**¿Para qué sirve?**  
Inicializa la base de datos: crea colecciones, índices y schema validation.

**¿Qué hace?**
1. Crea colección `documentos` con schema validation
2. Crea colección `chunks` con schema validation
3. Crea colección `evaluaciones` con schema validation
4. Crea índices en `documentos`: fecha_idioma, texto_completo, categoría
5. Crea índices en `chunks`: doc_estrategia, estrategia_index
6. Muestra instrucciones para crear índice vectorial manualmente (no se puede automatizar)

**¿Dónde se ejecuta?**  
Se ejecuta una sola vez antes de ingestar documentos.

---

### 10. `evaluation/ragas_eval.py`

**¿Para qué sirve?**  
Evalúa automáticamente el pipeline RAG usando la librería RAGAS.

**Métricas de evaluación:**
| Métrica | Descripción |
|---------|-------------|
| Faithfulness | ¿La respuesta es fiel al contexto? (sin alucinaciones) |
| Answer Relevancy | ¿La respuesta responde la pregunta? |
| Context Recall | ¿El contexto recuperado cubre la respuesta esperada? |

**Clases:**
- `EvalSample` — Par pregunta + ground truth
- `RAGASEvaluator` — Ejecuta evaluación

**Métodos:**
- `evaluate(samples, estrategia)` — Evalúa con RAGAS y guarda en MongoDB
- `evaluate_all_strategies()` — Evalúa las 3 estrategias y determina la mejor
- `get_ragas_report()` — Retorna reporte desde MongoDB

**¿Dónde guarda los resultados?**  
En la colección `evaluaciones` de MongoDB.

---

## Flujo Completo del Sistema

```
1. CONFIGURACIÓN
   .env → config/settings.py

2. INICIALIZACIÓN
   scripts/init_db.py → MongoDB (colecciones, índices)

3. INGESTA (ingestion/pipeline.py)
   data/dataset.json
   ↓
   DocumentoPadre (titulo, contenido, categoria...)
   ↓
   ChunkingStrategy (fixed/sentence-aware/semantic)
   ↓
   ChunkDoc (texto dividido en fragmentos)
   ↓
   Embedder (all-MiniLM-L6-v2 → vector de 384 dims)
   ↓
   MongoDB (documentos + chunks con embeddings)

4. BÚSQUEDA (retrieval/search.py)
   Query del usuario
   ↓
   Embedder (query → vector)
   ↓
   VectorSearch (MongoDB $vectorSearch)
   ↓
   Top-K chunks relevantes

5. RAG (rag/pipeline.py)
  Chunks relevantes + query
   ↓
   PromptBuilder (construye prompt)
   ↓
   Groq API (Llama 3.1)
   ↓
   Respuesta generada

6. EVALUACIÓN (evaluation/ragas_eval.py)
   20 pares pregunta/ground_truth
   ↓
   Pipeline RAG (genera respuestas)
   ↓
   RAGAS (calcula métricas)
   ↓
   MongoDB (evaluaciones)
```

---

## Cómo Ejecutar el Proyecto

```bash
# 1. Configurar entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Copiar y configurar .env
copy .env.example .env
# Editar MONGODB_URI y GROQ_API_KEY

# 3. Inicializar MongoDB
python scripts/init_db.py

# 4. Crear índice vectorial manualmente en Atlas UI
# (no se puede automatizar)

# 5. Ingestar documentos
python ingestion/pipeline.py

# 6. Levantar API
uvicorn api.main:app --reload --port 8000

# 7. Acceder a documentación
# http://localhost:8000/docs
```

---

## Colecciones MongoDB

### `documentos`
```json
{
  "_id": "ObjectId",
  "titulo": "string",
  "contenido_texto": "string",
  "categoria": "string",
  "idioma": "string",
  "url_imagen": "string | null",
  "metadata": {},
  "fecha_ingesta": "ISODate"
}
```

### `chunks`
```json
{
  "_id": "ObjectId",
  "doc_id": "ObjectId (ref documentos)",
  "chunk_index": 0,
  "estrategia_chunking": "fixed | sentence-aware | semantic",
  "chunk_texto": "string",
  "embedding": [float, ...],  // 384 dimensiones
  "modelo": "all-MiniLM-L6-v2",
  "tokens": 45,
  "fecha_ingesta": "ISODate",
  "metadata": {}
}
```

### `evaluaciones`
```json
{
  "_id": "ObjectId",
  "id_consulta": "eval_001",
  "pregunta": "string",
  "respuesta": "string",
  "faithfulness": 0.91,
  "answer_relevancy": 0.87,
  "context_recall": 0.78,
  "estrategia": "fixed",
  "modelo_eval": "ragas-v0.1",
  "fecha": "ISODate"
}
```
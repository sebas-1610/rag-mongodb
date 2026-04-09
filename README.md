# Sistema RAG NoSQL con MongoDB Atlas

Pipeline RAG completo sobre dominio de **Tecnología e Inteligencia Artificial**,
construido con MongoDB Atlas Vector Search, sentence-transformers y Groq (Llama 3.1).

## Stack Tecnológico

| Capa | Tecnología |
|---|---|
| Base de datos | MongoDB Atlas (Vector Search) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| LLM | Groq API — Llama 3.1 8B Instant |
| API | FastAPI + uvicorn |
| Evaluación | RAGAS (nota extra) |

## Estructura del Proyecto

```
rag-mongodb/
├── config/
│   └── settings.py          # Variables de entorno centralizadas
├── database/
│   └── mongodb.py           # Conexión Atlas, colecciones
├── chunking/
│   └── strategies.py        # Fixed-size, Sentence-aware, Semantic
├── embeddings/
│   └── embedder.py          # MiniLM-L6-v2 / Multilingüe
├── ingestion/
│   └── pipeline.py          # Carga docs → chunks → embeddings → MongoDB
├── retrieval/
│   └── search.py            # Búsqueda vectorial + híbrida + experimento
├── rag/
│   └── pipeline.py          # RAG: retrieval + prompt + Groq
├── api/
│   └── main.py              # FastAPI endpoints
├── evaluation/
│   └── ragas_eval.py        # Nota extra: evaluación automática
├── scripts/
│   └── init_db.py           # Inicializa índices y schema validation
├── data/
│   └── dataset.json         # Dataset de ejemplo (escalar a 100+ docs)
├── .env.example
└── requirements.txt
```

## Instalación

### 1. Clonar y configurar entorno

```bash
git clone <repo>
cd rag-mongodb

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con tus credenciales reales
```

Necesitás:
- **MongoDB Atlas URI** → [mongodb.com/atlas](https://mongodb.com/atlas) (cluster M0 gratis)
- **Groq API Key** → [console.groq.com](https://console.groq.com) (gratis)

### 3. Crear índice vectorial en Atlas

Después de ejecutar `init_db.py`, crear el índice knnVector manualmente:

1. Atlas UI → tu cluster → **Search** → **Create Search Index**
2. Seleccionar **JSON Editor**
3. Colección: `chunks`
4. Pegar:o

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}
```

5. Nombre del índice: `vector_index`

### 4. Inicializar base de datos

```bash
python scripts/init_db.py
# si no funciona probar con:
python -m scripts.init_db
```

### 5. Ingestar dataset

```bash
# Primero agregar tus documentos a data/dataset.json
# Luego ejecutar la ingesta con las 3 estrategias:
python ingestion/pipeline.py
# si no funciona probar con:
python -m ingestion.pipeline
```

### 6. Levantar la API

```bash
uvicorn api.main:app --reload 
```

Documentación interactiva: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Endpoints

### `GET /health`
Verifica conexión a MongoDB.

### `GET /stats`
Estadísticas de chunks por estrategia.

### `POST /search`
Búsqueda vectorial o híbrida.

```json
{
  "query": "¿Qué es RAG?",
  "top_k": 5
}
```

### `POST /rag`
Genera respuesta usando contexto de MongoDB + Groq.

```json
{
  "pregunta": "¿Cómo funciona la búsqueda vectorial?",
  "top_k": 5
}
```

### `GET /experiment/default`
Ejecuta el experimento de comparación de estrategias con las 10 consultas
de prueba (requerido en sección 5.4 del proyecto).

### `POST /experiment`
Experimento con consultas personalizadas.

---

## Experimento de Chunking (Sección 5.4)

El sistema ejecuta automáticamente las 10 consultas sobre las 3 estrategias:

```bash
curl http://localhost:8000/experiment/default
```

El reporte incluye:
- Score de similitud por estrategia
- Número de chunks recuperados
- Comparativa tabla por consulta

---

## Evaluación RAGAS (Nota Extra)

```bash
python evaluation/ragas_eval.py
```

Evalúa las 3 estrategias con 20 pares pregunta/ground_truth.
Guarda resultados en colección `evaluaciones` de MongoDB.

Métricas:
- **Faithfulness** — ¿La respuesta es fiel al contexto?
- **Answer Relevancy** — ¿La respuesta responde la pregunta?
- **Context Recall** — ¿El contexto recuperado cubre la respuesta esperada?

---

## Diseño SOLID

| Principio | Aplicación |
|---|---|
| **S** — Single Responsibility | `ChunkingStrategy`, `EmbedderBase`, `SearchEngine`, `LLMClientBase`: una resp. cada uno |
| **O** — Open/Closed | Nuevas estrategias/modelos: heredar sin modificar código base |
| **L** — Liskov | `FixedSizeStrategy`, `SentenceAwareStrategy`, `SemanticStrategy` son intercambiables |
| **I** — Interface Segregation | Interfaces mínimas: `ChunkingStrategy.split()`, `EmbedderBase.embed()` |
| **D** — Dependency Inversion | Pipeline depende de abstracciones; dependencias inyectadas en constructor |

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
  "chunk_index": 3,
  "estrategia_chunking": "sentence-aware",
  "chunk_texto": "El contenido del fragmento...",
  "embedding": [0.023, -0.117, "...384 dims"],
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
  "faithfulness": 0.91,
  "answer_relevancy": 0.87,
  "context_recall": 0.78,
  "estrategia": "semantic",
  "modelo_eval": "ragas-v0.1",
  "fecha": "ISODate"
}
```
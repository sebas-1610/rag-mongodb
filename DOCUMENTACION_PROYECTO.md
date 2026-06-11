# RAG MongoDB - Documentación del Proyecto

## ¿Qué es RAG?

**RAG (Retrieval-Augmented Generation)** es una técnica de inteligencia artificial que combina recuperación de información con generación de texto. En lugar de depender únicamente del conocimiento interno del modelo, RAG:

1. **Indexa** documentos en una base de datos vectorial
2. **Busca** fragmentos relevantes para cada pregunta
3. **Genera** respuestas basadas en el contexto recuperado

**Ventajas:** Respuestas más precisas, actualizable sin reentrenar, reduce alucinaciones.

---

## Arquitectura del Sistema

```
Documento → Chunking → Embeddings → MongoDB Atlas → Vector Search → LLM → Respuesta
```

| Componente | Tecnología |
|------------|------------|
| Base de datos | MongoDB Atlas (con Vector Search) |
| Embeddings | all-MiniLM-L6-v2 (384 dimensiones) |
| LLM | Groq Llama 3.1 8B Instant |
| API | FastAPI (async) |
| Frontend | HTML/CSS/JS vanilla |

---

## Estrategias de Chunking

El chunking divide documentos en fragmentos para indexar. Se implementaron 3 estrategias:

### 1. Fixed-size Chunking
- Divide en ventanas de N tokens con overlap
- **Ventaja:** Simple y rápido
- **Desventaja:** Puede cortar ideas a la mitad

### 2. Sentence-aware Chunking
- Respeta límites de oraciones
- **Ventaja:** Fragmentos más naturales y legibles
- **Desventaja:** Tamaños variables

### 3. Semantic Chunking
- Analiza similitud entre oraciones adyacentes
- Crea fragmentos cuando detecta cambio de tema
- **Ventaja:** Fragmentos coherentes temáticamente
- **Desventaja:** Más costoso computacionalmente

---

## Búsqueda Vectorial

### ¿Cómo funciona?
1. Convierte la pregunta en un vector (embedding)
2. Busca los vectores más cercanos usando similitud coseno
3. Retorna los fragmentos con mayor score

### Búsqueda Híbrida
Combina búsqueda vectorial con filtros de metadata:
- Por categoría
- Por rango de fechas
- Por estrategia de chunking

### Expansión de Consultas
El sistema expande automáticamente las preguntas:
- Acrónimos: "RAG" → "Retrieval-Augmented Generation"
- Typos: "argument" → "augmented"
- Sinónimos y términos relacionados

---

## Evaluación con RAGAS

**RAGAS (Retrieval Augmented Generation Assessment)** evalúa automáticamente la calidad del sistema RAG.

### Métricas principales

| Métrica | ¿Qué mide? |
|---------|------------|
| **Faithfulness** | ¿La respuesta es fiel al contexto? |
| **Answer Relevancy** | ¿La respuesta es pertinente a la pregunta? |
| **Context Recall** | ¿El contexto contiene toda la info necesaria? |
| **Context Precision** | ¿Los chunks recuperados son relevantes? |

Los scores van de 0 a 1 (1 = mejor).

---

## MongoDB Atlas Vector Search

### Configuración del índice
```json
{
  "fields": {
    "embedding": {
      "type": "knnVector",
      "dimensions": 384,
      "similarity": "cosine"
    }
  }
}
```

### Pipeline de búsqueda
```python
$vectorSearch → $match (filtros) → $project
```

- `numCandidates`: Candidatos antes del reranking (top_k × 10)
- `limit`: Resultados finales

---

## API Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/health` | GET | Verificar conexión MongoDB |
| `/stats` | GET | Estadísticas de chunks |
| `/search` | POST | Búsqueda vectorial |
| `/rag` | POST | Pipeline RAG completo |
| `/upload` | POST | Subir documentos |
| `/documents` | GET | Listar documentos |
| `/documents/{id}` | DELETE | Eliminar documento |

---

## Estructura de Chunks

Cada chunk en MongoDB contiene:

```json
{
  "chunk_texto": "Texto del fragmento",
  "embedding": [0.1, 0.2, ...],  // Vector de 384 dims
  "doc_id": "ID del documento",
  "estrategia_chunking": "semantic",
  "metadata": {
    "categoria": "inteligencia-artificial",
    "autor": "usuario",
    "año": 2026
  }
}
```

---

## Configuración

### Variables de entorno (.env)
```
MONGODB_URI=mongodb+srv://...
GROQ_API_KEY=gsk_...
```

### Modelos utilizados
- **Embedding:** all-MiniLM-L6-v2 (sentence-transformers)
- **LLM:** Llama 3.1 8B Instant (Groq)

---

## Resumen del Flujo

1. Usuario pregunta → "¿Qué es un embedding?"
2. Expansión de consulta → Agrega sinónimos
3. Búsqueda vectorial → Embedding de la pregunta vs chunks
4. Recuperación top-k → Los 5 chunks más relevantes
5. Construcción de prompt → Contexto + pregunta
6. Generación con LLM → Groq responde basándose en el contexto
7. Respuesta al usuario → Con fuentes citadas

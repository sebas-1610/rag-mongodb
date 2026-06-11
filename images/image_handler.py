"""
images/image_handler.py
=======================
S — Maneja upload, almacenamiento, descripción y búsqueda de imágenes.
D — Depende de CLIPEmbedder y MongoDB.
O — Extensible para nuevos modelos de descripción.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from io import BytesIO

from PIL import Image as PILImage

from embeddings.clip_embedder import get_clip_embedder
from database.mongodb import mongo


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Formatos soportados
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Carpeta de imágenes
IMAGES_DIR = Path(__file__).parent.parent / "static" / "images"

# Etiquetas para zero-shot classification (descripción de imágenes)
DEFAULT_LABELS = [
    "clouds", "sky", "sun", "moon", "stars",
    "dog", "cat", "bird", "fish", "horse",
    "person", "people", "woman", "man", "child",
    "tree", "flower", "grass", "mountain", "river",
    "ocean", "beach", "desert", "forest", "snow",
    "car", "truck", "bus", "bicycle", "motorcycle",
    "house", "building", "city", "street", "bridge",
    "food", "fruit", "vegetable", "table", "chair",
    "computer", "phone", "book", "paper", "pen",
    "ball", "toy", "guitar", "piano", "drum",
    "art", "painting", "drawing", "photo", "landscape",
    "abstract", "pattern", "texture", "color", "shape",
]


# =============================================================================
# IMAGE HANDLER
# =============================================================================


class ImageHandler:
    """
    Maneja el ciclo de vida completo de imágenes:
    - Upload y almacenamiento
    - Descripción automática (CLIP zero-shot)
    - Generación de embeddings
    - Búsqueda por similitud
    """

    def __init__(self):
        self._clip = get_clip_embedder()
        self._ensure_images_dir()

    def _ensure_images_dir(self):
        """Crea la carpeta de imágenes si no existe."""
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    def validate_image(self, filename: str) -> bool:
        """Valida que el formato de imagen sea soportado."""
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Formato no soportado: {suffix}. "
                f"Usa: {', '.join(SUPPORTED_FORMATS)}"
            )
        return True

    async def save_image(
        self,
        file_content: bytes,
        filename: str,
    ) -> str:
        """
        Guarda la imagen en static/images/.
        
        Args:
            file_content: Bytes del archivo
            filename: Nombre original del archivo
        
        Returns:
            Nombre del archivo guardado (con UUID)
        """
        self.validate_image(filename)

        # Generar nombre único
        suffix = Path(filename).suffix.lower()
        unique_name = f"{uuid.uuid4().hex}{suffix}"
        save_path = IMAGES_DIR / unique_name

        # Guardar archivo
        with open(save_path, "wb") as f:
            f.write(file_content)

        return unique_name

    def describe_image(
        self,
        image_path: str,
        labels: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Describe una imagen usando CLIP zero-shot.
        
        Args:
            image_path: Ruta completa de la imagen
            labels: Etiquetas candidatas (usa DEFAULT_LABELS si no se especifica)
            top_k: Número de etiquetas a retornar
        
        Returns:
            {
                "descripcion": str,
                "etiquetas": [{"label": str, "score": float}, ...]
            }
        """
        if labels is None:
            labels = DEFAULT_LABELS

        image = PILImage.open(image_path)

        # Obtener clasificación
        results = self._clip.describe_image(image, labels)

        # Construir descripción
        top_labels = [r["label"] for r in results[:top_k]]
        description = f"a photo of {', '.join(top_labels)}"

        return {
            "descripcion": description,
            "etiquetas": results[:top_k],
        }

    def embed_image(self, image_path: str) -> List[float]:
        """
        Genera el embedding CLIP de una imagen.
        
        Args:
            image_path: Ruta de la imagen
        
        Returns:
            Vector de 512 floats
        """
        image = PILImage.open(image_path)
        return self._clip.embed_image(image)

    async def process_image(
        self,
        file_content: bytes,
        filename: str,
        category: str = "imagen",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Procesa una imagen completa:
        1. Guarda la imagen
        2. Genera descripción con CLIP
        3. Genera embedding con CLIP
        4. Inserta en MongoDB
        
        Returns:
            {
                "doc_id": str,
                "filename": str,
                "descripcion": str,
                "etiquetas": list,
                "url_imagen": str,
            }
        """
        # 1. Guardar imagen
        saved_filename = await self.save_image(file_content, filename)
        image_path = str(IMAGES_DIR / saved_filename)

        # 2. Describir imagen
        description_result = self.describe_image(image_path)

        # 3. Generar embedding
        embedding = self.embed_image(image_path)

        # 4. Insertar en MongoDB
        doc_data = {
            "titulo": filename,
            "descripcion": description_result["descripcion"],
            "etiquetas": description_result["etiquetas"],
            "embedding": embedding,
            "url_imagen": f"/static/images/{saved_filename}",
            "categoria": category,
            "modelo": self._clip.model_name,
            "dimension": self._clip.dimension,
            "fecha_ingesta": datetime.utcnow(),
            "metadata": metadata or {},
        }

        result = await mongo.imagenes.insert_one(doc_data)
        doc_id = str(result.inserted_id)

        return {
            "doc_id": doc_id,
            "filename": filename,
            "saved_filename": saved_filename,
            "descripcion": description_result["descripcion"],
            "etiquetas": description_result["etiquetas"],
            "url_imagen": f"/static/images/{saved_filename}",
        }

    async def search_by_text(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Busca imágenes similares a una consulta de texto.
        
        Args:
            query: Texto de búsqueda (ej: "nubes bonitas")
            top_k: Número de resultados
        
        Returns:
            Lista de imágenes similares con scores
        """
        # Codificar texto con CLIP
        query_embedding = self._clip.embed_single(query)

        # Buscar en MongoDB
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index_imagenes",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "titulo": 1,
                    "descripcion": 1,
                    "etiquetas": 1,
                    "url_imagen": 1,
                    "categoria": 1,
                    "fecha_ingesta": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        results = []
        async for doc in mongo.imagenes.aggregate(pipeline):
            results.append({
                "doc_id": str(doc["_id"]),
                "titulo": doc.get("titulo", ""),
                "descripcion": doc.get("descripcion", ""),
                "etiquetas": doc.get("etiquetas", []),
                "url_imagen": doc.get("url_imagen", ""),
                "categoria": doc.get("categoria", ""),
                "fecha_ingesta": doc.get("fecha_ingesta"),
                "score": doc.get("score", 0.0),
            })

        return results

    async def search_by_image(
        self,
        image_path: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Busca imágenes similares a una imagen de consulta.
        
        Args:
            image_path: Ruta de la imagen de consulta
            top_k: Número de resultados
        
        Returns:
            Lista de imágenes similares con scores
        """
        # Codificar imagen con CLIP
        query_embedding = self._clip.embed_image(image_path)

        # Buscar en MongoDB
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index_imagenes",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "titulo": 1,
                    "descripcion": 1,
                    "etiquetas": 1,
                    "url_imagen": 1,
                    "categoria": 1,
                    "fecha_ingesta": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        results = []
        async for doc in mongo.imagenes.aggregate(pipeline):
            results.append({
                "doc_id": str(doc["_id"]),
                "titulo": doc.get("titulo", ""),
                "descripcion": doc.get("descripcion", ""),
                "etiquetas": doc.get("etiquetas", []),
                "url_imagen": doc.get("url_imagen", ""),
                "categoria": doc.get("categoria", ""),
                "fecha_ingesta": doc.get("fecha_ingesta"),
                "score": doc.get("score", 0.0),
            })

        return results

    async def get_image(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene una imagen por su ID.
        """
        from bson import ObjectId
        doc = await mongo.imagenes.find_one({"_id": ObjectId(doc_id)})
        if doc:
            return {
                "doc_id": str(doc["_id"]),
                "titulo": doc.get("titulo", ""),
                "descripcion": doc.get("descripcion", ""),
                "etiquetas": doc.get("etiquetas", []),
                "url_imagen": doc.get("url_imagen", ""),
                "categoria": doc.get("categoria", ""),
                "fecha_ingesta": doc.get("fecha_ingesta"),
            }
        return None

    async def delete_image(self, doc_id: str) -> bool:
        """
        Elimina una imagen y su archivo físico.
        """
        from bson import ObjectId

        doc = await mongo.imagenes.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return False

        # Eliminar archivo físico
        url_imagen = doc.get("url_imagen", "")
        if url_imagen:
            # Extraer nombre del archivo de la URL
            filename = url_imagen.split("/")[-1]
            file_path = IMAGES_DIR / filename
            if file_path.exists():
                os.remove(file_path)

        # Eliminar de MongoDB
        await mongo.imagenes.delete_one({"_id": ObjectId(doc_id)})
        return True

    async def list_images(
        self,
        limit: int = 50,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Lista todas las imágenes.
        """
        results = []
        async for doc in mongo.imagenes.find().skip(skip).limit(limit):
            results.append({
                "doc_id": str(doc["_id"]),
                "titulo": doc.get("titulo", ""),
                "descripcion": doc.get("descripcion", ""),
                "url_imagen": doc.get("url_imagen", ""),
                "categoria": doc.get("categoria", ""),
                "fecha_ingesta": doc.get("fecha_ingesta"),
            })
        return results

    async def count_images(self) -> int:
        """Cuenta el total de imágenes."""
        return await mongo.imagenes.count_documents({})


# =============================================================================
# SINGLETON
# =============================================================================

_image_handler: Optional[ImageHandler] = None


def get_image_handler() -> ImageHandler:
    """Retorna instancia singleton del ImageHandler."""
    global _image_handler
    if _image_handler is None:
        _image_handler = ImageHandler()
    return _image_handler

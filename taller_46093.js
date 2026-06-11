// TALLER FINAL - 46093 - Jhoan Sebastian Perilla Delgado
// Universidad de Caldas - Ingeniería de Sistemas y Computación
// Bases de datos no relacionales 

`-------------------------------------------------------------------------------------------------------------------`

// Actividad A:
// A1: Aplicar validador estricto a la colección 'chunks'
    db.runCommand({
    collMod: "chunks",
    validator: {
        $jsonSchema: {
        bsonType: "object",
        required: [
            "doc_id",
            "chunk_index",
            "estrategia_chunking",
            "chunk_texto",
            "embedding",
            "modelo",
            "fecha_ingesta"
        ],
        properties: {
            doc_id: {
            bsonType: "objectId",
            description: "Referencia obligatoria al documento maestro"
            },
            chunk_index: {
            bsonType: "int",
            minimum: 0
            },
            estrategia_chunking: {
            bsonType: "string",
            enum: ["fixed", "sentence-aware", "semantic"],
            description: "Solo se admiten las 3 estrategias del motor RAG"
            },
            chunk_texto: {
            bsonType: "string",
            minLength: 10
            },
            embedding: {
            bsonType: "array",
            minItems: 384,
            maxItems: 384,
            items: { bsonType: "double" },
            description: "Vector denso generado por all-MiniLM-L6-v2 (384 dimensiones)"
            },
            modelo: {
            bsonType: "string"
            },
            fecha_ingesta: {
            bsonType: "date"
            },
            metadata: {
            bsonType: "object",
            required: ["posicion", "total_chunks"],
            properties: {
                posicion: { bsonType: "int" },
                total_chunks: { bsonType: "int" }
            }
            }
        }
        }
    },
    validationLevel: "strict",
    validationAction: "error"
    });

    // RESPUESTA:
    // la respuesta del comando es positiva
    `{
    ok: 1,
    '$clusterTime': {
        clusterTime: Timestamp({ t: 1780588494, i: 5 }),
        signature: {
        hash: Binary.createFromBase64('E4XY4kj1wR6ffpw6Z9JNPna74TE=', 0),
        keyId: Long('7606450882955182151')
        }
    },
    operationTime: Timestamp({ t: 1780588494, i: 5 })
    }`

    // A2: Insertar un documento que cumpla con el esquema
    db.chunks.insertOne({
    doc_id: new ObjectId(),
    chunk_index: 0,
    estrategia_chunking: "semantic",
    chunk_texto: "Los sistemas RAG combinan recuperación de información con generación de texto.",
    embedding: Array(384).fill(0.015), 
    modelo: "all-MiniLM-L6-v2",
    fecha_ingesta: new Date(),
    metadata: { posicion: 0, total_chunks: 1 }
    });
    // RESPUESTA:
    // La respuesta del comando es positiva y se inserta el documento correctamente
    `{
    acknowledged: true,
    insertedId: ObjectId('6a21a1bbccde27f80dce35f4')
    }`

    // A3: Intentar insertar un documento que no cumpla con el esquema
    try {
    db.chunks.insertOne({
        doc_id: new ObjectId(),
        chunk_index: 1,
        estrategia_chunking: "estrategia-inventada", 
        chunk_texto: "Texto corto",
        embedding: [0.1, -0.2], 
        modelo: "all-MiniLM-L6-v2",
        fecha_ingesta: new Date()
    });
    } catch (error) {
    print("=== DETALLES DE LA VIOLACIÓN DE ESQUEMA ===");
    printjson(error.errInfo); 
    }

    // RESPUESTA:
    // Aqui se evidencia que el esquema funciona correctamente al no dejar insertar un doc que no cumple
    // Las validaciones que fallan son:
    // embedding y estrategia_chunking, embedding porque no tiene las 384 dimensiones y 
    // estrategia_chunking porque no es una de las 3 opciones permitidas
    `{
        failingDocumentId: ObjectId('6a21a733ccde27f80dce35f8'),
            details: {
                operatorName: '$jsonSchema',
                schemaRulesNotSatisfied: [ [Object] ]
            }   
    }`

`-------------------------------------------------------------------------------------------------------------------`

//Actividad B:
    // CREACION DE INDICES 
        //  Indice Simple Ascendente
            db.chunks.createIndex({ estado: 1});
        // Crear indice compuesto bajo ESR
            db.chunks.createIndex(
            { estrategia_chunking: 1, chunk_index: 1, fecha_ingesta: -1 },
            { name: "idx_estrategia_index_fecha_ESR" }
            );
        // RESPUESTA:
        // Esta respuesta indica que se creo correctamente el indice
            `idx_estrategia_index_fecha_ESR`

        // Indice unico sobre codigo del proyecto
            db.chunks.createIndex(
            { doc_id: 1, estrategia_chunking: 1, chunk_index: 1 },
            { unique: true, name: "idx_chunk_identificador_unico" }
            );
        //RESPUESTA:
        // Esta respuesta indica que se creo correctamente el indice unico
            `idx_chunk_identificador_unico`

        //Prueba de Rechazo
            // insertamos un doc 
            const miDocId = new ObjectId();
            db.chunks.insertOne({
            doc_id: miDocId,
            chunk_index: 0,
            estrategia_chunking: "semantic",
            chunk_texto: "Fragmento original de inteligencia artificial...",
            embedding: Array(384).fill(0.1),
            modelo: "all-MiniLM-L6-v2",
            fecha_ingesta: new Date()
            });

            // intentamos insertar el mismo documento
            try {
                db.chunks.insertOne({
                    doc_id: miDocId, // Usamos el  mismo id del padre
                    chunk_index: 0,  // Mismo indice de chunk
                    estrategia_chunking: "semantic", // se usa la misma estrategia
                    chunk_texto: "Un intento malicioso de duplicar el fragmento 0...",
                    embedding: Array(384).fill(0.1),
                    modelo: "all-MiniLM-L6-v2",
                    fecha_ingesta: new Date()
                });
            } catch (error) {
                print("\n--- EVIDENCIA: RECHAZO DE DUPLICADO EN CHUNKS ---");
                print(error.message); 
            }
        // RESPUESTA:
            `--- EVIDENCIA : RECHAZO DE DUPLICADO EN CHUNKS ---
            E11000 duplicate key error collection: rag_ia.chunks index: idx_chunk_identificador_unico dup key: { doc_id: ObjectId('6a21abf9ccde27f80dce35f9'), estrategia_chunking: "semantic", chunk_index: 0 }`
        // Indice de texto completo sobre nombre y descripcion
            try {
            db.chunks.createIndex(
                { chunk_texto: "text", modelo: "text" },
                { name: "idx_texto_chunks_modelo", default_language: "spanish" }
            );
            } catch (error) {
            print("\n--- EVIDENCIA: RESTRICCIÓN DE ÍNDICE DE TEXTO ÚNICO ---");
            print(error.message); 
            }
        // RESPUESTA:
            `--- EVIDENCIA : RESTRICCIÓN DE ÍNDICE DE TEXTO ÚNICO ---
            An equivalent index already exists with a different name and options. Requested index: { v: 2, key: { _fts: "text", _ftsx: 1 }, name: "idx_texto_chunks_modelo", default_language: "spanish", weights: { chunk_texto: 1, modelo: 1 }, language_override: "language", textIndexVersion: 3 }, existing index: { v: 2, key: { _fts: "text", _ftsx: 1 }, name: "idx_texto_completo", default_language: "spanish", weights: { chunk_texto: 1 }, language_override: "language", textIndexVersion: 3 }`
        
        // Busqueda completa :
            print("\n--- EJECUTANDO BÚSQUEDA FULL-TEXT EN CHUNKS ---");
            const resultadosTexto = db.chunks.find(
            { $text: { $search: "atención transformers" } }
            ).limit(2).toArray();

            printjson(resultadosTexto);
            // RESPUESTA:
            // Como la respuesta es tan larga la resumi eliminandole varios embeddings, pero consrervando la estructura de esta
            `[
                {
                    _id: ObjectId('6a219a088f2133c17e7eb786'),
                    doc_id: ObjectId('6a219a078f2133c17e7eb785'),
                    chunk_index: 0,
                    estrategia_chunking: 'sentence-aware',
                    chunk_texto: "La arquitectura transformer, presentada en el paper 'Attention is All You Need' (2017), revolucionó el procesamiento de lenguaje natural. El mecanismo de atención multi-head permite al modelo enfocarse en diferentes partes de la secuencia de entrada simultáneamente. Cada cabeza de atención aprende relaciones distintas entre tokens. BERT usa encoders bidireccionales para entender el contexto completo de una oración, lo que lo hace ideal para clasificación y extracción de información. GPT usa decoders autorregresivos para predecir el siguiente token, siendo la base de los modelos generativos actuales.",
                    embedding: [
                            -0.1555614173412323,   -0.026170359924435616,   -0.02364956960082054,
                            ................. MAS EMBEDDINGS OMITIDOS POR LONGITUD .................
                    ],
                    modelo: 'all-MiniLM-L6-v2',
                    tokens: 86,
                    fecha_ingesta: 2026-06-04T15:30:15.974Z,
                    metadata: { posicion: 0, total_chunks: 2 }
                },
                {
                    _id: ObjectId('6a219a058f2133c17e7eb76f'),
                    doc_id: ObjectId('6a219a048f2133c17e7eb76e'),
                    chunk_index: 0,
                    estrategia_chunking: 'fixed',
                    chunk_texto: "La arquitectura transformer, presentada en el paper 'Attention is All You Need' (2017), revolucionó el procesamiento de lenguaje natural. El mecanismo de atención multi-head permite al modelo enfocarse en diferentes partes de la secuencia de entrada simultáneamente. Cada cabeza de atención aprende relaciones distintas entre tokens. BERT usa encoders bidireccionales para entender el contexto completo de una oración, lo que lo hace ideal para clasificación y extracción de información. GPT usa decoders autorregresivos para predecir el siguiente token, siendo la base de los modelos generativos actuales.",
                    embedding: [
                        -0.15556149184703827,   -0.026170389726758003,  -0.023649565875530243,
                        ................. MAS EMBEDDINGS OMITIDOS POR LONGITUD .................
                    ],
                    modelo: 'all-MiniLM-L6-v2',
                    tokens: 86,
                    fecha_ingesta: 2026-06-04T15:30:13.109Z,
                    metadata: { posicion: 0, total_chunks: 1 }
                }
                ]`
        // Indice TTL, borrar proyectos cancelados tras 30 dias

        // Inspeccionar indeces existentes 
        
        // EXPLAIN: Analizar el uso del indice
            const planSinIndice = db.chunks
            .find({ modelo: "all-MiniLM-L6-v2" })
            .explain("executionStats");
            print("--- MÉTRICAS DE ESCANEO COMPLETO (COLLSCAN) ---");
            print("Stage:             ", planSinIndice.queryPlanner.winningPlan.stage);
            print("Docs examinados:   ", planSinIndice.executionStats.totalDocsExamined);
            print("Docs retornados:   ", planSinIndice.executionStats.nReturned);
            print("Tiempo (ms):       ", planSinIndice.executionStats.executionTimeMillis);
        // RESPUESTA:
            `--- MÉTRICAS DE ESCANEO COMPLETO (COLLSCAN) ---
                Stage:             
                COLLSCAN
                Docs examinados:   
                166
                Docs retornados:   
                166
                Tiempo (ms):       
                0`
        
        // Creacion de indice full text para buscar por palabras clave
        db.chunks.createIndex(
        { chunk_texto: "text" },
        { default_language: "spanish", name: "idx_texto_completo" }
        );
        // RESPUESTA:
        // Esta respuesta indica que se creo correctamente el indice
        `idx_texto_completo`
        // Borrar un indice
            db.chunks.dropIndex("idx_texto_completo");
            // RESPUESTA:
            // Esta respuesta indica que se borro correctamente el indice
            `{
                nIndexesWas: 6,
                ok: 1,
                '$clusterTime': {
                    clusterTime: Timestamp({ t: 1780592566, i: 2 }),
                    signature: {
                    hash: Binary.createFromBase64('RmBmUteyv8j3qdo4ob7qo4LOTbg=', 0),
                    keyId: Long('7606450882955182151')
                    }
                },
                operationTime: Timestamp({ t: 1780592566, i: 2 })
            }`

`-------------------------------------------------------------------------------------------------------------------`

// Actividad C:
    // C1 Actualizacion masiva y Atomica de metadatos
        const resultadoActualizacion = db.chunks.updateMany(
        { estrategia_chunking: "fixed" },
        { 
            $set: { 
            modelo: "all-MiniLM-L6-v2-upgrade",
            fecha_ingesta: new Date() 
            } 
        }
        );

        printjson(resultadoActualizacion);
    // RESPUESTA:
        // La respuesta indica que se actualizaron 20 documentos que cumplian la condicion de estrategia_chunking: "fixed"
        `{
            acknowledged: true,
            insertedId: null,
            matchedCount: 20,
            modifiedCount: 20,
            upsertedCount: 0
        }`
    
    // C2 Integridad en contadores y campos numericos
        const resultadoIncremento = db.chunks.updateMany(
        { estrategia_chunking: "semantic" },
        { $inc: { tokens: 10 } }
        );

        printjson(resultadoIncremento);
    // RESPUESTA:
        // La respuesta indica que se actualizaron 108 documentos que cumplian la condicion de estrategia_chunking: "semantic"
        `{
        acknowledged: true,
        insertedId: null,
        matchedCount: 108,
        modifiedCount: 108,
        upsertedCount: 0
        }`

    // C3 Borrado en Cascada y Prueba de Integridad Referencial
        const documentoAEliminar = db.chunks.findOne({}, { doc_id: 1 }).doc_id;
        if (documentoAEliminar) {
        const resultadoLimpieza = db.chunks.deleteMany({ doc_id: documentoAEliminar });
        print("Limpieza de chunks huérfanos para el documento: " + documentoAEliminar);
        printjson(resultadoLimpieza);
        } else {
        print("No se encontraron chunks para realizar la prueba de integridad referencial.");
        }
    // RESPUESTA:
        // La respuesta indica que se eliminaron 2 documentos que tenian el mismo doc_id, lo que evidencia la integridad referencial entre chunks y su documento padre
        `Limpieza de chunks huérfanos para el documento: 69d7cee882fb9ea2f8c839cb
        {
        acknowledged: true,
        deletedCount: 1
        }`
    
    // C4 Conteo de consistencia
    db.chunks.aggregate([
    {
        $group: {
        _id: "$estrategia_chunking",
        total_chunks: { $sum: 1 },
        promedio_tokens: { $avg: "$tokens" }
        }
    }
    ]).forEach(printjson);
    // RESPUESTA:
        // La respuesta muestra el conteo total de chunks por cada estrategia de chunking, asi como el promedio de tokens por chunk para cada estrategia, lo que evidencia la consistencia de los datos en la colección
    `   { _id: 'fixed', total_chunks: 19, promedio_tokens: 88.57894736842105 },
        {
            _id: 'sentence-aware',
            total_chunks: 38,
            promedio_tokens: 54.63157894736842
        },
        {
            _id: 'semantic',
            total_chunks: 108,
            promedio_tokens: 26.40740740740741
        }`

`--------------------------------------------------------------------------------------------------------------------`

// Actividad D MQL + Aggregation + Busqueda Vectorial:
    // Vector de 384 dimensiones generado por all-MiniLM-L6-v2
        const miQueryVector = Array(384).fill(0.015);
    // Respuesta:
        `const miQueryVector = Array(384).fill(0.015);`
        // es decir se ejecuto correctamente
    
    // D1 y D2:
        db.chunks.aggregate([
        // STAGE 1: Búsqueda semántica 
        {
            $vectorSearch: {
            index: "vector_index",
            path: "embedding",
            queryVector: miQueryVector,
            numCandidates: 100, 
            limit: 20           
            }
        },
        
        // STAGE 2: Filtros relacionales exactos
        {
            $match: {
            estrategia_chunking: "semantic",
            fecha_ingesta: { $gte: new Date("2026-01-01") }
            }
        },
        
        // STAGE 3: Proyección 
        {
            $project: {
            _id: 0, 
            doc_id: 1,
            chunk_texto: 1,
            estrategia_chunking: 1,
            score_semantico: { $meta: "vectorSearchScore" }
            }
        },

        // STAGE 4: Limite de resultados finales
        { $limit: 5 }
        ]).forEach(printjson);
        // RESPUESTA:
            // No esta respondiendo nada

    // D3 Consulta con Join
        db.chunks.aggregate([
            {
                $vectorSearch: {
                index: "vector_index",
                path: "embedding",
                queryVector: miQueryVector,
                numCandidates: 100,
                limit: 20
                }
            },
            // 1. Unir con la colección de documentos
            {
                $lookup: {
                from: "documentos", // Nombre de tu colección de documentos
                localField: "doc_id",
                foreignField: "_id",
                as: "doc_padre"
                }
            },
            // 2. Convertir el array de documentos a un solo objeto
            { $unwind: "$doc_padre" },
            // 3. Proyección final (Aquí puedes ver campos de ambos lados)
            {
                $project: {
                _id: 0,
                titulo_documento: "$doc_padre.titulo",
                fragmento: "$chunk_texto",
                estrategia: "$estrategia_chunking",
                score_semantico: { $meta: "vectorSearchScore" }
                // Puedes agregar aquí más campos de doc_padre si los necesitas
                }
            },
            { $limit: 5 }
            ]).forEach(printjson);

        // RESPUESTA:
            // La respuesta muestra los resultados de la consulta con join entre chunks y documentos, evidenciando que se obtuvieron fragmentos relevantes junto con el título del documento padre.
            `{
                titulo_documento: 'RAGAS: Evaluación Automática de Sistemas RAG',
                fragmento: 'Para calcular Context Recall se requiere un ground truth manual.',
                estrategia: 'semantic',
                score_semantico: 0.5117796659469604
            },
            {
                titulo_documento: 'RAGAS: Evaluación Automática de Sistemas RAG',
                fragmento: 'Para calcular Context Recall se requiere un ground truth manual.',
                estrategia: 'semantic',
                score_semantico: 0.5117796659469604
            },
            {
                titulo_documento: 'FastAPI: Framework Moderno para APIs Python',
                fragmento: 'Genera documentación interactiva automáticamente via Swagger UI.',
                estrategia: 'semantic',
                score_semantico: 0.5061275362968445
            },
            {
                titulo_documento: 'FastAPI: Framework Moderno para APIs Python',
                fragmento: 'Genera documentación interactiva automáticamente via Swagger UI.',
                estrategia: 'semantic',
                score_semantico: 0.5061275362968445
            },
            {
                titulo_documento: 'RAGAS: Evaluación Automática de Sistemas RAG',
                fragmento: 'Answer Relevancy mide si la respuesta es pertinente a la pregunta original.',
                estrategia: 'semantic',
                score_semantico: 0.5050328373908997
            }`
        

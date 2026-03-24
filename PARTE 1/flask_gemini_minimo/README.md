# Flask + Gemini API + RAG con PDFs en Cloud Storage

Esta versión añade la **parte 3** de la sesión:

- PDFs subidos manualmente a un bucket de **Cloud Storage**
- carga de PDFs desde GCS
- extracción de texto
- chunking
- embeddings con **Gemini API**
- recuperación semántica en memoria
- generación final con contexto recuperado (RAG)

## Archivos nuevos o modificados

- `app.py` → ahora soporta modo RAG
- `rag.py` → lógica de carga desde bucket, chunking, embeddings y retrieval
- `templates/index.html` → botón para recargar PDFs desde GCS y panel de estado
- `static/styles.css` → estilos para la nueva interfaz
- `requirements.txt` → dependencias extra para Storage, PDF y vectores
- `.env.example` → variables de entorno para GCS y RAG
- `docs_to_collect.md` → sugerencias de documentos para la demo

## Variables nuevas

- `GCS_BUCKET_NAME`
- `GCS_PREFIX`
- `RAG_EMBEDDING_MODEL`
- `RAG_EMBEDDING_DIM`
- `RAG_CHUNK_SIZE`
- `RAG_CHUNK_OVERLAP`
- `RAG_TOP_K`
- `RAG_MAX_PDFS`
- `RAG_EMBED_BATCH_SIZE`

## Flujo recomendado

1. Crear bucket en la consola de GCP.
2. Subir manualmente los PDFs.
3. Dar permiso de lectura al service account de Cloud Run sobre ese bucket.
4. Añadir `GCS_BUCKET_NAME` y `GCS_PREFIX` en Cloud Run.
5. Hacer push al repo para redeploy.
6. Abrir la app y pulsar **Recargar PDFs desde GCS**.
7. Hacer preguntas con la casilla **Usar RAG** activada.

## Ejecución local

La app base sigue funcionando igual con `GEMINI_API_KEY`.

Para probar la parte de GCS en local sin `gcloud`, necesitarías credenciales de Google Cloud disponibles para el cliente de Storage. Para la sesión, el camino más simple es validar la parte RAG **ya desplegada en Cloud Run**.

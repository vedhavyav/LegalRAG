Full legal judgment
Stored in legal_documents.full_text 
        |
        |
        |
    Chunking algorithm
chunk_0     chunk_1     chunk_2

A. Count documents and chunks
SELECT COUNT(*) AS document_count
FROM legal_documents;

SELECT COUNT(*) AS chunk_count
FROM document_chunks;

B. Inspect chunks from your sample judgment
SELECT
    source_id,
    chunk_index,
    char_start,
    char_end,
    LENGTH(chunk_text) AS chunk_length,
    LEFT(chunk_text, 200) AS preview
FROM document_chunks
WHERE source_id = '117859307'
ORDER BY chunk_index
LIMIT 10;

C. Check for missing or empty chunks
SELECT COUNT(*) AS invalid_chunks
FROM document_chunks
WHERE chunk_text IS NULL
   OR LENGTH(TRIM(chunk_text)) = 0;

D. Check that every chunk references a valid judgment
SELECT COUNT(*) AS orphan_chunks
FROM document_chunks c
LEFT JOIN legal_documents d
    ON d.source_id = c.source_id
WHERE d.source_id IS NULL;

E. Inspect the chunk boundaries
SELECT
    chunk_index,
    char_start,
    char_end,
    LENGTH(chunk_text) AS chunk_length
FROM document_chunks
WHERE source_id = '117859307'
ORDER BY chunk_index;
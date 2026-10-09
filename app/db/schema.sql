
CREATE TABLE IF NOT EXISTS legal_documents (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    source_id TEXT NOT NULL UNIQUE,

    title TEXT NOT NULL,
    court TEXT,
    judgment_date DATE,

    source_url TEXT,

    full_text TEXT NOT NULL,

    num_cites INTEGER DEFAULT 0,
    num_cited_by INTEGER DEFAULT 0,

    categories JSONB NOT NULL DEFAULT '[]'::jsonb,
    related_queries JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_legal_documents_court
    ON legal_documents (court);

CREATE INDEX IF NOT EXISTS idx_legal_documents_date
    ON legal_documents (judgment_date);

CREATE INDEX IF NOT EXISTS idx_legal_documents_title
    ON legal_documents (title);

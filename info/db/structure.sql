CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE FUNCTION modified_trigger() RETURNS trigger
    LANGUAGE plpgsql
    AS $$BEGIN
            NEW.modified_at := NOW();
            RETURN NEW;
        END;$$;

-- search_request

CREATE TABLE search_request (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query TEXT NOT NULL DEFAULT '',
    search_start_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    full_execution_time INTEGER NOT NULL DEFAULT 0,
    search_execution_time INTEGER NOT NULL DEFAULT 0,
    dense_search_time INTEGER NOT NULL DEFAULT 0,
    lex_search_time INTEGER NOT NULL DEFAULT 0,
    query_norm_time INTEGER NOT NULL DEFAULT 0,
    reranker_time INTEGER NOT NULL DEFAULT 0,
    model_name VARCHAR(500) NOT NULL DEFAULT '',
    reranker_name VARCHAR(500) NOT NULL DEFAULT '',
    reranker_enable BOOLEAN NOT NULL DEFAULT FALSE,
    lex_enable BOOLEAN NOT NULL DEFAULT FALSE,
    from_cache BOOLEAN NOT NULL DEFAULT FALSE,
    lex_candidate VARCHAR(500) DEFAULT 'OpenSearch',
    dense_top_k INTEGER NOT NULL DEFAULT 0,
    lex_top_k INTEGER NOT NULL DEFAULT 0,
    top_k INTEGER NOT NULL DEFAULT 0,
    weight_ce REAL NOT NULL DEFAULT 0.0,
    weight_dense REAL NOT NULL DEFAULT 0.0,
    weight_lex REAL NOT NULL DEFAULT 0.0,
    results JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_search_request_query_gin_trgm ON search_request USING GIN (query gin_trgm_ops);
CREATE INDEX idx_search_request_model_name_gin_trgm ON search_request USING GIN (model_name gin_trgm_ops);
CREATE INDEX idx_search_request_reranker_name_gin_trgm ON search_request USING GIN (reranker_name gin_trgm_ops);
CREATE INDEX idx_search_request_lex_candidate_gin_trgm ON search_request USING GIN (lex_candidate gin_trgm_ops);

CREATE INDEX idx_search_request_results_jsonb_path ON search_request USING GIN (results jsonb_path_ops);

CREATE INDEX idx_search_request_search_start_time_asc ON search_request USING btree (search_start_time ASC);
CREATE INDEX idx_search_request_search_start_time_desc ON search_request USING btree (search_start_time DESC);
CREATE INDEX idx_search_request_created_at_asc ON search_request USING btree (created_at ASC);
CREATE INDEX idx_search_request_created_at_desc ON search_request USING btree (created_at DESC);
CREATE INDEX idx_search_request_modified_at_asc ON search_request USING btree (modified_at ASC);
CREATE INDEX idx_search_request_modified_at_desc ON search_request USING btree (modified_at DESC);

CREATE INDEX idx_search_request_full_execution_time ON search_request USING btree (full_execution_time);
CREATE INDEX idx_search_request_dense_search_time ON search_request USING btree (dense_search_time);
CREATE INDEX idx_search_request_lex_search_time ON search_request USING btree (lex_search_time);
CREATE INDEX idx_search_request_query_norm_time ON search_request USING btree (query_norm_time);
CREATE INDEX idx_search_request_reranker_time ON search_request USING btree (reranker_time);
CREATE INDEX idx_search_request_search_execution_time ON search_request USING btree (search_execution_time);
CREATE INDEX idx_search_request_dense_top_k ON search_request USING btree (dense_top_k);
CREATE INDEX idx_search_request_lex_top_k ON search_request USING btree (lex_top_k);
CREATE INDEX idx_search_request_top_k ON search_request USING btree (top_k);
CREATE INDEX idx_search_request_weight_ce ON search_request USING btree (weight_ce);
CREATE INDEX idx_search_request_weight_dense ON search_request USING btree (weight_dense);
CREATE INDEX idx_search_request_weight_lex ON search_request USING btree (weight_lex);

CREATE INDEX idx_search_request_reranker_enable ON search_request USING btree (reranker_enable);
CREATE INDEX idx_search_request_lex_enable ON search_request USING btree (lex_enable);

CREATE TRIGGER modified_trigger BEFORE UPDATE ON search_request FOR EACH ROW EXECUTE FUNCTION modified_trigger();

-- search_feedback

CREATE TYPE rating_enum AS ENUM (
    'like',
    'dislike'
);

CREATE TABLE search_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    search_request_id UUID NOT NULL,
    question_id VARCHAR(500) NOT NULL DEFAULT '',
    rating rating_enum NOT NULL,
    source VARCHAR(500) NOT NULL DEFAULT '',
    comment TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_search_feedback_request_question UNIQUE (search_request_id, question_id)
);

CREATE INDEX idx_search_feedback_created_at_asc ON search_feedback USING btree (created_at ASC);
CREATE INDEX idx_search_feedback_created_at_desc ON search_feedback USING btree (created_at DESC);
CREATE INDEX idx_search_feedback_modified_at_asc ON search_feedback USING btree (modified_at ASC);
CREATE INDEX idx_search_feedback_modified_at_desc ON search_feedback USING btree (modified_at DESC);

CREATE INDEX idx_search_feedback_search_request_id ON search_feedback USING btree (search_request_id);
CREATE INDEX idx_search_feedback_question_id ON search_feedback USING btree (question_id);
CREATE INDEX idx_search_feedback_source ON search_feedback USING btree (source);
CREATE INDEX idx_search_feedback_comment_gin ON search_feedback USING GIN (comment gin_trgm_ops);

CREATE INDEX idx_search_feedback_id_search_request_id ON search_feedback USING btree (id, search_request_id);
CREATE INDEX idx_search_feedback_search_request_id_question_id ON search_feedback USING btree (search_request_id, question_id);

ALTER TABLE search_feedback
    ADD CONSTRAINT fk_search_feedback_search_request
        FOREIGN KEY (search_request_id)
            REFERENCES search_request(id)
                ON DELETE CASCADE;

CREATE TRIGGER modified_trigger BEFORE UPDATE ON search_feedback FOR EACH ROW EXECUTE FUNCTION modified_trigger();

-- knowledge_feedback

CREATE TABLE knowledge_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    search_request_id UUID NOT NULL,
    question_id VARCHAR(500) NOT NULL DEFAULT '',
    rating rating_enum NOT NULL,
    source VARCHAR(500) NOT NULL DEFAULT '',
    comment TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_knowledge_feedback_request_question UNIQUE (search_request_id, question_id),
);

CREATE INDEX idx_knowledge_feedback_created_at_asc ON knowledge_feedback USING btree (created_at ASC);
CREATE INDEX idx_knowledge_feedback_created_at_desc ON knowledge_feedback USING btree (created_at DESC);
CREATE INDEX idx_knowledge_feedback_modified_at_asc ON knowledge_feedback USING btree (modified_at ASC);
CREATE INDEX idx_knowledge_feedback_modified_at_desc ON knowledge_feedback USING btree (modified_at DESC);

CREATE INDEX idx_knowledge_feedback_search_request_id ON knowledge_feedback USING btree (search_request_id);
CREATE INDEX idx_knowledge_feedback_question_id ON knowledge_feedback USING btree (question_id);
CREATE INDEX idx_knowledge_feedback_source ON knowledge_feedback USING btree (source);
CREATE INDEX idx_knowledge_feedback_comment_gin ON knowledge_feedback USING GIN (comment gin_trgm_ops);

CREATE INDEX idx_knowledge_feedback_id_search_request_id ON knowledge_feedback USING btree (id, search_request_id);
CREATE INDEX idx_knowledge_feedback_search_request_id_question_id ON knowledge_feedback USING btree (search_request_id, question_id);

ALTER TABLE knowledge_feedback
    ADD CONSTRAINT fk_knowledge_feedback_search_request
        FOREIGN KEY (search_request_id)
            REFERENCES search_request(id)
                ON DELETE CASCADE;

CREATE TRIGGER modified_trigger BEFORE UPDATE ON knowledge_feedback FOR EACH ROW EXECUTE FUNCTION modified_trigger();

--

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

    CONSTRAINT unique_knowledge_feedback_request_question UNIQUE (search_request_id, question_id)
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

-- source

CREATE TABLE source (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_source_name UNIQUE (name)
);

CREATE INDEX idx_source_name ON source USING btree (name);
CREATE INDEX idx_source_name_gin_trgm ON source USING GIN (name gin_trgm_ops);

CREATE INDEX idx_source_created_at_asc ON source USING btree (created_at ASC);
CREATE INDEX idx_source_created_at_desc ON source USING btree (created_at DESC);
CREATE INDEX idx_source_modified_at_asc ON source USING btree (modified_at ASC);
CREATE INDEX idx_source_modified_at_desc ON source USING btree (modified_at DESC);

CREATE TRIGGER modified_trigger
    BEFORE UPDATE ON source
    FOR EACH ROW
    EXECUTE FUNCTION modified_trigger();

-- product

CREATE TABLE product (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_product_name UNIQUE (name)
);

CREATE INDEX idx_product_name ON product USING btree (name);
CREATE INDEX idx_product_name_gin_trgm ON product USING GIN (name gin_trgm_ops);

CREATE INDEX idx_product_created_at_asc ON product USING btree (created_at ASC);
CREATE INDEX idx_product_created_at_desc ON product USING btree (created_at DESC);
CREATE INDEX idx_product_modified_at_asc ON product USING btree (modified_at ASC);
CREATE INDEX idx_product_modified_at_desc ON product USING btree (modified_at DESC);

CREATE TRIGGER modified_trigger
    BEFORE UPDATE ON product
    FOR EACH ROW
    EXECUTE FUNCTION modified_trigger();

-- role

CREATE TABLE role (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_role_name UNIQUE (name)
);

CREATE INDEX idx_role_name ON role USING btree (name);
CREATE INDEX idx_role_name_gin_trgm ON role USING GIN (name gin_trgm_ops);

CREATE INDEX idx_role_created_at_asc ON role USING btree (created_at ASC);
CREATE INDEX idx_role_created_at_desc ON role USING btree (created_at DESC);
CREATE INDEX idx_role_modified_at_asc ON role USING btree (modified_at ASC);
CREATE INDEX idx_role_modified_at_desc ON role USING btree (modified_at DESC);

CREATE TRIGGER modified_trigger
    BEFORE UPDATE ON role
    FOR EACH ROW
    EXECUTE FUNCTION modified_trigger();

-- component

CREATE TABLE component (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_component_name UNIQUE (name)
);

CREATE INDEX idx_component_name ON component USING btree (name);
CREATE INDEX idx_component_name_gin_trgm ON component USING GIN (name gin_trgm_ops);

CREATE INDEX idx_component_created_at_asc ON component USING btree (created_at ASC);
CREATE INDEX idx_component_created_at_desc ON component USING btree (created_at DESC);
CREATE INDEX idx_component_modified_at_asc ON component USING btree (modified_at ASC);
CREATE INDEX idx_component_modified_at_desc ON component USING btree (modified_at DESC);

CREATE TRIGGER modified_trigger
    BEFORE UPDATE ON component
    FOR EACH ROW
    EXECUTE FUNCTION modified_trigger();

-- search_entry

CREATE TABLE search_entry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    source_id UUID NOT NULL,
    knowledge_number VARCHAR(500) NOT NULL DEFAULT '',
    page_id VARCHAR(500) NOT NULL DEFAULT '',

    is_actual BOOLEAN,
    is_second_line BOOLEAN,

    role_id UUID NOT NULL,
    product_id UUID NOT NULL,
    component_id UUID NOT NULL,

    question_markdown TEXT NOT NULL DEFAULT '',
    question_clean TEXT NOT NULL DEFAULT '',

    error_analysis_markdown TEXT NOT NULL DEFAULT '',
    error_analysis_clean TEXT NOT NULL DEFAULT '',

    answer_markdown TEXT NOT NULL DEFAULT '',
    answer_clean TEXT NOT NULL DEFAULT '',

    is_for_user BOOLEAN NOT NULL DEFAULT FALSE,
    jira VARCHAR(2000) NOT NULL DEFAULT '',

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_search_entry_source_knowledge_page
        UNIQUE (source_id, knowledge_number, page_id)
);

ALTER TABLE search_entry
    ADD CONSTRAINT fk_search_entry_source
        FOREIGN KEY (source_id)
            REFERENCES source(id)
                ON DELETE RESTRICT;

ALTER TABLE search_entry
    ADD CONSTRAINT fk_search_entry_role
        FOREIGN KEY (role_id)
            REFERENCES role(id)
                ON DELETE RESTRICT;

ALTER TABLE search_entry
    ADD CONSTRAINT fk_search_entry_product
        FOREIGN KEY (product_id)
            REFERENCES product(id)
                ON DELETE RESTRICT;

ALTER TABLE search_entry
    ADD CONSTRAINT fk_search_entry_component
        FOREIGN KEY (component_id)
            REFERENCES component(id)
                ON DELETE RESTRICT;

CREATE INDEX idx_search_entry_created_at_asc ON search_entry USING btree (created_at ASC);
CREATE INDEX idx_search_entry_created_at_desc ON search_entry USING btree (created_at DESC);
CREATE INDEX idx_search_entry_modified_at_asc ON search_entry USING btree (modified_at ASC);
CREATE INDEX idx_search_entry_modified_at_desc ON search_entry USING btree (modified_at DESC);

CREATE INDEX idx_search_entry_source_id ON search_entry USING btree (source_id);
CREATE INDEX idx_search_entry_role_id ON search_entry USING btree (role_id);
CREATE INDEX idx_search_entry_product_id ON search_entry USING btree (product_id);
CREATE INDEX idx_search_entry_component_id ON search_entry USING btree (component_id);

CREATE INDEX idx_search_entry_knowledge_number ON search_entry USING btree (knowledge_number);
CREATE INDEX idx_search_entry_page_id ON search_entry USING btree (page_id);
CREATE INDEX idx_search_entry_jira ON search_entry USING btree (jira);

CREATE INDEX idx_search_entry_knowledge_number_gin_trgm ON search_entry USING GIN (knowledge_number gin_trgm_ops);
CREATE INDEX idx_search_entry_page_id_gin_trgm ON search_entry USING GIN (page_id gin_trgm_ops);
CREATE INDEX idx_search_entry_jira_gin_trgm ON search_entry USING GIN (jira gin_trgm_ops);

CREATE INDEX idx_search_entry_question_markdown_gin_trgm ON search_entry USING GIN (question_markdown gin_trgm_ops);
CREATE INDEX idx_search_entry_question_clean_gin_trgm ON search_entry USING GIN (question_clean gin_trgm_ops);
CREATE INDEX idx_search_entry_error_analysis_markdown_gin_trgm ON search_entry USING GIN (error_analysis_markdown gin_trgm_ops);
CREATE INDEX idx_search_entry_error_analysis_clean_gin_trgm ON search_entry USING GIN (error_analysis_clean gin_trgm_ops);
CREATE INDEX idx_search_entry_answer_markdown_gin_trgm ON search_entry USING GIN (answer_markdown gin_trgm_ops);
CREATE INDEX idx_search_entry_answer_clean_gin_trgm ON search_entry USING GIN (answer_clean gin_trgm_ops);


CREATE INDEX idx_search_entry_is_actual ON search_entry USING btree (is_actual);
CREATE INDEX idx_search_entry_is_second_line ON search_entry USING btree (is_second_line);
CREATE INDEX idx_search_entry_is_for_user ON search_entry USING btree (is_for_user);

CREATE INDEX idx_search_entry_id_source_id ON search_entry USING btree (id, source_id);
CREATE INDEX idx_search_entry_id_role_id ON search_entry USING btree (id, role_id);
CREATE INDEX idx_search_entry_id_product_id ON search_entry USING btree (id, product_id);
CREATE INDEX idx_search_entry_id_component_id ON search_entry USING btree (id, component_id);

CREATE INDEX idx_search_entry_source_id_knowledge_number ON search_entry USING btree (source_id, knowledge_number);
CREATE INDEX idx_search_entry_source_id_knowledge_number_page_id ON search_entry USING btree (source_id, knowledge_number, page_id);
CREATE INDEX idx_search_entry_role_id_product_id_component_id ON search_entry USING btree (role_id, product_id, component_id);

CREATE TRIGGER modified_trigger
    BEFORE UPDATE ON search_entry
    FOR EACH ROW
    EXECUTE FUNCTION modified_trigger();

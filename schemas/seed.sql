-- PostgreSQL schema for seeds data
-- Stores seed repository scores for each run
-- Source files: seed/*.csv

CREATE TABLE IF NOT EXISTS devrank.seeds (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES devrank.runs(run_id) ON DELETE CASCADE,
    user_id VARCHAR(512) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- Ensure unique combination of run_id and user_id
    CONSTRAINT unique_seeds_run_user UNIQUE (run_id, user_id)
);

-- Index for filtering by run_id
CREATE INDEX IF NOT EXISTS idx_seeds_run_id ON devrank.seeds(run_id);

-- Index for user_id lookups
CREATE INDEX IF NOT EXISTS idx_seeds_user_id ON devrank.seeds(user_id);

-- Index for value-based sorting/filtering
CREATE INDEX IF NOT EXISTS idx_seeds_value ON devrank.seeds(value DESC);

-- Composite index for run_id and user_id lookups
CREATE INDEX IF NOT EXISTS idx_seeds_run_user_id ON devrank.seeds(run_id, user_id);

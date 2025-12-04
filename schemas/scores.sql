-- PostgreSQL schema for scores data
-- Source files: scores/*.csv
-- Stores repository scores from devrank calculations

CREATE TABLE IF NOT EXISTS devrank.scores (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES devrank.runs(run_id) ON DELETE CASCADE,
    user_id VARCHAR(512) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- Ensure unique combination of run_id and user_id
    CONSTRAINT unique_scores_run_user UNIQUE (run_id, user_id)
);

-- Index for faster lookups by run_id
CREATE INDEX IF NOT EXISTS idx_scores_run_id ON devrank.scores(run_id);

-- Index for faster lookups by user_id
CREATE INDEX IF NOT EXISTS idx_scores_user_id ON devrank.scores(user_id);

-- Index for value ordering/filtering
CREATE INDEX IF NOT EXISTS idx_scores_value ON devrank.scores(value DESC);

-- Composite index for common query pattern
CREATE INDEX IF NOT EXISTS idx_scores_run_user ON devrank.scores(run_id, user_id);

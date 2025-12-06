-- PostgreSQL schema for scores data
-- Source files: scores/*.csv
-- Stores repository scores from devrank calculations

CREATE TABLE IF NOT EXISTS devrank.scores (
    id SERIAL PRIMARY KEY,
    community_id TEXT NOT NULL,
    run_id INTEGER NOT NULL,
    user_id VARCHAR(512) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- Foreign key to runs table (composite key)
    CONSTRAINT fk_scores_run FOREIGN KEY (community_id, run_id)
        REFERENCES devrank.runs(community_id, run_id) ON DELETE CASCADE,
    -- Ensure unique combination of community_id, run_id and user_id
    CONSTRAINT unique_scores_community_run_user UNIQUE (community_id, run_id, user_id)
);

-- Index for faster lookups by community_id
CREATE INDEX IF NOT EXISTS idx_scores_community_id ON devrank.scores(community_id);

-- Index for faster lookups by run_id
CREATE INDEX IF NOT EXISTS idx_scores_run_id ON devrank.scores(run_id);

-- Index for faster lookups by user_id
CREATE INDEX IF NOT EXISTS idx_scores_user_id ON devrank.scores(user_id);

-- Index for value ordering/filtering
CREATE INDEX IF NOT EXISTS idx_scores_value ON devrank.scores(value DESC);

-- Composite index for common query pattern
CREATE INDEX IF NOT EXISTS idx_scores_community_run ON devrank.scores(community_id, run_id);

-- PostgreSQL schema for runs data
-- Tracks execution runs of the devrank system

CREATE TABLE IF NOT EXISTS devrank.runs (
    run_id SERIAL PRIMARY KEY,
    community_id TEXT NOT NULL,
    ecosystems TEXT,
    days_back INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for filtering by community_id
CREATE INDEX IF NOT EXISTS idx_runs_community_id ON devrank.runs(community_id);

-- Index for filtering by creation time
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON devrank.runs(created_at);

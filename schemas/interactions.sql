-- PostgreSQL schema for cache/interactions data
-- Source files: cache/interactions_MM_YYYY.csv

CREATE TABLE IF NOT EXISTS devrank.interactions (
    id SERIAL PRIMARY KEY,
    user_login VARCHAR(255) NOT NULL,
    repo VARCHAR(512) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_count INTEGER NOT NULL DEFAULT 0,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_interactions_user_login ON devrank.interactions(user_login);
CREATE INDEX IF NOT EXISTS idx_interactions_repo ON devrank.interactions(repo);
CREATE INDEX IF NOT EXISTS idx_interactions_event_type ON devrank.interactions(event_type);
CREATE INDEX IF NOT EXISTS idx_interactions_year_month ON devrank.interactions(year, month);
CREATE INDEX IF NOT EXISTS idx_interactions_user_repo ON devrank.interactions(user_login, repo);

-- Unique constraint to prevent duplicate entries
CREATE UNIQUE INDEX IF NOT EXISTS idx_interactions_unique
    ON devrank.interactions(user_login, repo, event_type, year, month);

-- Event type enum for validation (optional, can be used as CHECK constraint)
-- Known event types: ISSUE_OPENED, COMMIT_CODE, ISSUE_COMMENTED, ISSUE_CLOSED,
--                    PULL_REQUEST_MERGED, STARRED, etc.

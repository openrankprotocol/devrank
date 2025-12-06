-- PostgreSQL schema for cache/interactions data with indexes
-- Source files: cache/interactions_MM_YYYY.csv
-- Use this schema after bulk loading data for better query performance

CREATE TABLE IF NOT EXISTS devrank.interactions (
    id SERIAL PRIMARY KEY,
    user_login VARCHAR(255) NOT NULL,
    repo VARCHAR(512) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_count INTEGER NOT NULL DEFAULT 0,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL
);

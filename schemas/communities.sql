-- PostgreSQL schema for communities data
-- Stores community information with social links

CREATE TABLE IF NOT EXISTS openrank.communities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    display_picture TEXT,
    telegram BIGINT,
    discord BIGINT,
    github TEXT,
    x BIGINT,
    farcaster TEXT,
    seed_x TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster lookups by name
CREATE INDEX IF NOT EXISTS idx_communities_name ON openrank.communities(name);

-- Index for github lookups
CREATE INDEX IF NOT EXISTS idx_communities_github ON openrank.communities(github);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION openrank.update_communities_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update updated_at on row updates
DROP TRIGGER IF EXISTS trigger_communities_updated_at ON openrank.communities;
CREATE TRIGGER trigger_communities_updated_at
    BEFORE UPDATE ON openrank.communities
    FOR EACH ROW
    EXECUTE FUNCTION openrank.update_communities_updated_at();

-- PostgreSQL schema for ecosystems data
-- Stores repository URLs and their associated sub-ecosystems

CREATE TABLE IF NOT EXISTS devrank.ecosystems (
    id SERIAL PRIMARY KEY,
    ecosystem_name VARCHAR(100) NOT NULL,
    url TEXT NOT NULL,
    sub_ecosystems TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- Ensure unique combination of ecosystem and URL
    CONSTRAINT unique_ecosystem_url UNIQUE (ecosystem_name, url)
);

-- Index for faster lookups by ecosystem name
CREATE INDEX IF NOT EXISTS idx_ecosystems_name ON devrank.ecosystems(ecosystem_name);

-- Index for URL searches
CREATE INDEX IF NOT EXISTS idx_ecosystems_url ON devrank.ecosystems(url);

-- Index for sub_ecosystems searches
CREATE INDEX IF NOT EXISTS idx_ecosystems_sub_ecosystems ON devrank.ecosystems(sub_ecosystems);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION devrank.update_ecosystems_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update updated_at on row updates
DROP TRIGGER IF EXISTS trigger_ecosystems_updated_at ON devrank.ecosystems;
CREATE TRIGGER trigger_ecosystems_updated_at
    BEFORE UPDATE ON devrank.ecosystems
    FOR EACH ROW
    EXECUTE FUNCTION devrank.update_ecosystems_updated_at();

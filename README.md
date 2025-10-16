# Crypto Ecosystem Contributor Network Analyzer

A powerful tool to analyze cryptocurrency development ecosystems by mapping contributor networks across GitHub repositories using the Open Source Observer (OSO) database.

## 🎯 What It Does

This tool performs a comprehensive 4-step analysis:

1. **Seed Repository Analysis** - Starts with configurable crypto repositories (Bitcoin, Ethereum, Cosmos, etc.)
2. **Core Contributor Discovery** - Finds the most active contributors to these seed projects
3. **Extended Repository Mapping** - Discovers all other repositories these contributors work on
4. **Extended Contributor Network** - Maps the broader ecosystem of developers in related projects

## 🚀 Quick Start

### Prerequisites

1. **OSO Account & API Key**
   - Create account at [opensource.observer](https://www.opensource.observer)
   - Generate API key in Account Settings > API Keys
   - Set environment variable: `export OSO_API_KEY="your_key_here"`

2. **Install Dependencies**
   ```bash
   pip install pyoso pandas python-dotenv tomli
   ```

3. **Configure Analysis**
   - Copy and customize `config.toml` (see Configuration section below)

### Run Analysis

```bash
# Step 1: Run the main analysis
python oso_github_repositories.py

# Step 2: Generate trust network (optional)
python generate_trust.py
```

## ⚙️ Configuration

All parameters are configured via TOML files. Here's the complete configuration structure:

### Basic Configuration (`config.toml`)

```toml
[general]
# Output directory for CSV files
output_dir = "./raw"

# Date range for contributions (in days from now)
# Set to 0 to include all historical data
days_back = 0  # 0 = all time, 365 = last year, 730 = last 2 years

# List of seed organizations to analyze
seed_orgs = [
    "ethereum",
    "bitcoin",
    "cosmos",
    "paradigmxyz",
    "compound-finance",
    "smartcontractkit",
    "uniswap",
    "offchainlabs",
    "foundry-rs",
    "paritytech"
]

[analysis]
# Enable extended repository and contributor analysis (set to false to only analyze seed repos)
extended_analysis = true

# Minimum commits threshold for all analysis (repositories, core contributors, extended contributors)
min_commits = 5

# Repository filtering
max_repos_per_org = 200

# Minimum core contributors threshold for all analysis (seed repos, extended repos, all phases)
min_core_contributors = 2
```

### Advanced Filters

```toml
[filters]
# Repository filters
exclude_forks = true
exclude_archived = false

# Contributor filters
exclude_bots = false  # Set to true to exclude bot accounts
bot_keywords = ["bot", "dependabot", "mergify", "renovate", "github-actions"]

# Date filter - uses days_back from [general] section

[output]
# Output file naming
timestamp_format = "%Y%m%d_%H%M%S"
file_prefix = "crypto"
include_headers = true
include_timestamp_in_filename = true
```



## 🔗 Graph Builder (Trust Networks)

After running the main analysis, you can generate trust relationships between contributors and repositories using the graph builder:

### How It Works

The `generate_trust.py` script processes your analysis results to create weighted trust networks:

1. **Reads Analysis Data** - Uses the CSV files from your main analysis (seed repos, contributors, extended repos)
2. **Queries OSO Database** - Gets detailed GitHub activity data for user-repository pairs
3. **Calculates Trust Scores** - Weights different activities:
   - **Commits**: 5 points (user → repo), 3 points (repo → user)
   - **Pull Requests Opened**: 20 points (user → repo), 5 points (repo → user)
   - **Pull Requests Merged**: 10 points (user → repo), 1 point (repo → user)
   - **Issues Opened**: 10 points (user → repo)
   - **Stars**: 5 points (user → repo)
   - **Forks**: 1 point (user → repo)
4. **Generates Bidirectional Graph** - Creates both user-to-repository and repository-to-user trust relationships

### Configuration

The graph builder uses your existing `config.toml` settings and can be enabled in the output section:

```toml
[output]
include_headers = true
include_timestamp_in_filename = false
```

### Usage Example

```bash
# 1. Run main analysis first
python oso_github_repositories.py

# 2. Generate trust network from the results
python generate_trust.py

# 3. Results will be in trust/github.csv
head trust/github.csv
# i,j,v
# vitalik,ethereum/go-ethereum,245.0
# ethereum/solidity,chriseth,89.5
# ...
```

### Output

Trust relationships are saved as CSV files in the `trust/` directory:
- **`github.csv`** - Main trust network with columns: `i` (from), `j` (to), `v` (trust value)
- Format suitable for graph analysis tools like NetworkX, igraph, or Gephi

### Performance

- Processes relationships in batches of 200 to avoid database timeouts
- Automatically saves progress every 10 batches
- Filters out bot accounts and inactive relationships

## 📊 Output Files

The main analyzer generates 4 CSV files:

### 1. Seed Repositories (`*_seed_repos_*.csv`)
```csv
organization,repository_name,contributor_count,total_commits,status
ethereum,go-ethereum,31,18238,found
bitcoin,bitcoin,14,38508,found
```

### 2. Core Contributors (`*_core_contributors_*.csv`)
```csv
contributor_handle,total_commits,total_active_days,seed_repositories
chriseth,51499,1276,ethereum/solidity
vitalik,8234,945,ethereum/go-ethereum, ethereum/solidity
```

### 3. Extended Repositories (`*_extended_repos_*.csv`)
```csv
organization,repository_name,core_contributor_count,total_commits
ethereum,solc-js,7,3390
cosmos,gaia,4,1296
```

### 4. Extended Contributors (`*_extended_contributors_*.csv`)
```csv
contributor_handle,repos_contributed,total_commits
alexanderbez,13,3064
marbar3778,11,6758
```

## 🔧 Example Configurations

### DeFi Focused Analysis
```toml
[general]
seed_orgs = [
    "Uniswap",
    "compound-finance",
    "makerdao",
    "aave"
]

[analysis]
min_commits = 10

[filters]
exclude_bots = true
# Uses days_back from [general] section for date filtering
```

### Layer 2 Ecosystem
```toml
[general]
seed_orgs = [
    "ethereum-optimism",
    "0xPolygonMatic",
    "matter-labs",
    "starkware-libs"
]

[analysis]
min_commits = 3  # Lower threshold for newer projects
```

### Bitcoin Ecosystem
```toml
[general]
seed_orgs = [
    "bitcoin",
    "lightninglabs",
    "ElementsProject",
    "BlockstreamResearch"
]

[general]
days_back = 1095  # Last 3 years
```

## 📈 Use Cases

- **Investment Research** - Map development activity across crypto ecosystems
- **Talent Discovery** - Find top contributors in specific blockchain domains
- **Ecosystem Analysis** - Understand project relationships and cross-pollination
- **Due Diligence** - Assess developer community strength and engagement
- **Recruitment** - Identify active developers for hiring
- **Academic Research** - Study open source collaboration patterns

## 🔍 Analysis Insights

The tool reveals:
- **Developer Migration Patterns** - How contributors move between projects
- **Ecosystem Boundaries** - Which projects share common developers
- **Influence Networks** - Key individuals working across multiple important projects
- **Project Relationships** - Unexpected connections between seemingly unrelated repos
- **Community Health** - Distribution of contributions and contributor diversity

## ⚡ Performance Tips

1. **Start Small** - Begin with 3-5 seed repos and adjust limits
2. **Use Date Filters** - Limit analysis to recent periods for faster queries
3. **Enable Bot Filtering** - Exclude automated contributions for cleaner results
4. **Adjust Repository Limits** - Use `max_repos_per_org` to balance thoroughness vs. speed
5. **Custom Output Directories** - Organize results by analysis type or date

## 🚨 Limitations

- **Data Freshness** - OSO data may have some lag from live GitHub
- **Private Repos** - Only analyzes public GitHub repositories
- **Attribution** - Relies on consistent GitHub usernames/emails
- **Scope** - Currently focused on GitHub; doesn't include GitLab, Bitbucket, etc.

## 🛠️ Troubleshooting

### Common Issues

**"No valid seed repositories found"**
- Check repository names in config (must be "org/repo" format)
- Verify repos exist in OSO database

**"Query timeout or error"**
- Reduce repository limits in config (e.g., `max_repos_per_org`)
- Add date filters to limit scope
- Try smaller batches of seed repositories

**"No contributors found"**
- Lower `min_commits` threshold
- Check date range isn't too restrictive
- Verify seed repos have recent activity

### Debug Mode
Set environment variable for verbose output:
```bash
export OSO_DEBUG=1
python oso_github_repositories.py
```

## 📝 License

Apache 2.0

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional data sources beyond GitHub
- Enhanced network visualization capabilities
- Advanced statistical analysis features
- Graph metrics computation (centrality, clustering, etc.)
- Performance optimizations
- Additional export formats (JSON, Parquet, etc.)

## 📚 Related Projects

- [Open Source Observer](https://www.opensource.observer) - Data source
- [PyOSO](https://github.com/opensource-observer/oso/tree/main/packages/pyoso) - Python client library

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
# Using default config.toml
python oso_github_repositories.py

# Using custom configuration file
python oso_github_repositories.py my_custom_config.toml
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

[seed_repositories]
# List of seed repositories to analyze
repos = [
    "ethereum/go-ethereum",
    "bitcoin/bitcoin",
    "ethereum/solidity",
    "cosmos/cosmos-sdk",
    "ethereum/web3.py",
]

[contributors]
# Minimum number of commits a contributor must have to be considered "core"
min_commits = 5

# Maximum number of top contributors to fetch per seed repository
max_contributors_per_repo = 50

# Maximum number of total core contributors to use for extended analysis
max_core_contributors_for_extended_analysis = 20

[extended_repositories]
# Minimum number of core contributors a repo must have to be included
min_core_contributors = 2

# Maximum number of extended repositories to find
max_extended_repos = 100

# Maximum number of extended repos to use for finding extended contributors
max_extended_repos_for_contributors = 30

[extended_contributors]
# Minimum commits in extended repos to be considered an extended contributor
min_commits_extended = 3

# Maximum number of extended contributors to fetch
max_extended_contributors = 200
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

# Date filters (overrides days_back if set)
start_date = ""  # Format: "2020-01-01" or empty for no limit
end_date = ""    # Format: "2024-12-31" or empty for no limit

[output]
# Output file naming
timestamp_format = "%Y%m%d_%H%M%S"
file_prefix = "crypto"
include_headers = true
include_timestamp_in_filename = true
```

### Performance Tuning

```toml
[query_limits]
# SQL query limits to prevent timeouts
# Increase these for more comprehensive results (but slower queries)
core_contributors_limit = 1000
extended_repos_limit = 200
extended_contributors_limit = 500
```

## 📊 Output Files

The analyzer generates 4 CSV files:

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
[seed_repositories]
repos = [
    "Uniswap/v3-core",
    "compound-finance/compound-protocol",
    "makerdao/dss",
    "aave/protocol-v2"
]

[contributors]
min_commits = 10
max_core_contributors_for_extended_analysis = 15

[filters]
exclude_bots = true
start_date = "2020-01-01"  # DeFi boom period
```

### Layer 2 Ecosystem
```toml
[seed_repositories]
repos = [
    "ethereum-optimism/optimism",
    "0xPolygonMatic/bor",
    "matter-labs/zksync",
    "starkware-libs/cairo-lang"
]

[contributors]
min_commits = 3  # Lower threshold for newer projects
```

### Bitcoin Ecosystem
```toml
[seed_repositories]
repos = [
    "bitcoin/bitcoin",
    "lightninglabs/lnd",
    "ElementsProject/elements",
    "BlockstreamResearch/secp256k1-zkp"
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
4. **Tune Query Limits** - Increase gradually based on your needs vs. speed requirements
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
- Reduce query limits in config
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
- Network visualization capabilities
- Statistical analysis features
- Performance optimizations
- Additional export formats (JSON, Parquet, etc.)

## 📚 Related Projects

- [Open Source Observer](https://www.opensource.observer) - Data source
- [PyOSO](https://github.com/opensource-observer/oso/tree/main/packages/pyoso) - Python client library

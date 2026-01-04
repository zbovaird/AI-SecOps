# AI SecOps Workspace - Setup Instructions

## Prerequisites

Before starting, ensure you have:

1. **Python 3.9+** installed
2. **Google Cloud Platform Account** with:
   - Vertex AI API enabled
   - Service account with appropriate permissions
   - GCP project set up
3. **Git** for version control
4. **Docker** (optional, for containerized deployments)

## Initial Setup

### 1. Clone/Initialize Repository

```bash
# If starting fresh
cd "AI SecOps"
git init

# If cloning from GitHub (once pushed)
git clone <repository-url>
cd "AI SecOps"
```

### 2. Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Google Cloud SDK

```bash
# macOS (using Homebrew)
brew install --cask google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install

# Initialize and authenticate
gcloud init
gcloud auth application-default login
```

### 4. Set Up Google Cloud Project

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## Tool Installation

### Quick Installation (Recommended)

Use the provided installation script:

```bash
./install_tools.sh
```

### Manual Installation

#### Install Microsoft PyRIT

```bash
# Install PyRIT from Microsoft
pip install pyrit

# Or install from source for latest version
git clone https://github.com/Azure/PyRIT.git
cd PyRIT
pip install -e .
cd ..
```

#### Install IBM Adversarial Robustness Toolbox (ART)

```bash
# Install ART
pip install adversarial-robustness-toolbox

# For specific frameworks, install extras:
# pip install adversarial-robustness-toolbox[tensorflow]
# pip install adversarial-robustness-toolbox[pytorch]
```

### Verify Installation and Explore Capabilities

After installation, explore what's available:

```bash
# Run the detailed exploration script
python explore_tools_detailed.py

# This will:
# 1. Show all capabilities organized by testing type
# 2. Save detailed results to tools_exploration_results.json
# 3. Display summary statistics

# Or view the pre-documented capabilities
cat tools_capabilities.md
```

### Install Additional Dependencies

```bash
# Install project requirements
pip install -r requirements.txt
```

## Configuration

### 1. Google Cloud Configuration

Create `config/gcp_config.yaml`:

```yaml
project_id: YOUR_PROJECT_ID
region: us-central1
service_account: path/to/service-account-key.json
vertex_endpoint_config:
  machine_type: n1-standard-4
  min_replica_count: 1
  max_replica_count: 3
```

### 2. Environment Variables

Create `.env` file (add to `.gitignore`):

```bash
# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1

# Optional: API Keys for external services
# OPENAI_API_KEY=your-key-here
```

### 3. Test Configuration

Create `config/test_config.yaml`:

```yaml
test_settings:
  blackbox:
    enabled: true
    timeout: 300
  whitebox:
    enabled: true
    timeout: 600
  workflow:
    enabled: true
    timeout: 900

tools:
  pyrit:
    enabled: true
    config_path: config/pyrit_config.yaml
  art:
    enabled: true
    framework: tensorflow  # or pytorch
```

## Model Deployment to Vertex AI

### 1. Prepare Model for Deployment

```bash
# Example: Deploy a test model
python scripts/deploy_model.py \
  --model-name test-model \
  --model-path models/local/test_model \
  --endpoint-name test-endpoint \
  --region us-central1
```

### 2. Verify Deployment

```bash
# List endpoints
gcloud ai endpoints list --region=us-central1

# Test endpoint
python scripts/test_endpoint.py \
  --endpoint-id <endpoint-id> \
  --sample-input data/sample_input.json
```

## Running Security Tests

### Blackbox Testing

```bash
# Run PyRIT blackbox tests
python scripts/run_tests.py \
  --test-type blackbox \
  --tool pyrit \
  --endpoint-id <vertex-endpoint-id>

# Run ART blackbox tests
python scripts/run_tests.py \
  --test-type blackbox \
  --tool art \
  --endpoint-id <vertex-endpoint-id>
```

### Whitebox Testing

```bash
# Run whitebox tests (requires model artifacts)
python scripts/run_tests.py \
  --test-type whitebox \
  --tool art \
  --model-path models/local/model.pkl
```

### End-to-End Workflow Testing

```bash
# Run complete workflow test
python scripts/run_tests.py \
  --test-type workflow \
  --scenario scenarios/workflow_adversarial.json \
  --endpoint-id <vertex-endpoint-id>
```

## Usage Examples

### Example 1: Quick Security Assessment

```bash
# Run all test types on a deployed endpoint
python scripts/run_security_assessment.py \
  --endpoint-id <vertex-endpoint-id> \
  --output-dir results/assessment_$(date +%Y%m%d)
```

### Example 2: Testing Models from GitHub

```bash
# Clone and test a model from GitHub
git clone <model-repository-url> models/external/model-repo
python scripts/test_external_model.py \
  --repo-path models/external/model-repo \
  --test-type blackbox
```

### Example 3: Custom Attack Scenario

```python
# Create custom attack scenario
from tools.custom.attack_scenario import AttackScenario

scenario = AttackScenario(
    name="jailbreak_attempt",
    attack_type="prompt_injection",
    target_endpoint="<endpoint-id>"
)

results = scenario.execute()
results.save_report("results/jailbreak_report.json")
```

## Troubleshooting

### Common Issues

1. **GCP Authentication Errors**
   ```bash
   # Re-authenticate
   gcloud auth application-default login
   ```

2. **Import Errors for PyRIT/ART**
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate
   pip install --upgrade pyrit adversarial-robustness-toolbox
   ```

3. **Vertex AI Endpoint Not Found**
   ```bash
   # Verify endpoint exists and is accessible
   gcloud ai endpoints describe <endpoint-id> --region=us-central1
   ```

4. **Permission Denied Errors**
   - Verify service account has Vertex AI User role
   - Check project billing is enabled

## Project Structure Guide

- `config/`: Configuration files for tools and cloud services
- `models/`: Local model files and Vertex AI endpoint configurations
- `tools/`: PyRIT, ART, and custom tool integrations
- `tests/`: Test cases organized by type (blackbox, whitebox, workflow)
- `scripts/`: Utility and deployment scripts
- `results/`: Test results and generated reports
- `docs/`: Additional documentation

## Next Steps

1. Complete Phase 1 setup tasks from `plan.md`
2. Deploy a test model to Vertex AI
3. Run your first security test
4. Review and document findings
5. Iterate based on results

## Resources

- [Microsoft PyRIT Documentation](https://github.com/Azure/PyRIT)
- [IBM ART Documentation](https://adversarial-robustness-toolbox.org/)
- [Google Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [GCP Service Account Setup](https://cloud.google.com/iam/docs/service-accounts)

## Red Team Kit - Attack Chain Usage

For detailed instructions on using the Red Team Kit attack chain and modules against real targets, see:

**[`redteam_kit/instructions.md`](redteam_kit/instructions.md)**

The Red Team Kit provides comprehensive attack chain capabilities for:
- Real target reconnaissance (port scanning, service detection)
- Exploit testing and execution (SQL injection, RCE, command injection, XXE)
- Post-exploitation activities (credential harvesting, privilege escalation, lateral movement)
- Persistence mechanisms (scheduled tasks, startup scripts, registry modification)
- Multi-stage attack chain orchestration

**⚠️ WARNING: FOR AUTHORIZED SECURITY TESTING IN SANDBOXED ENVIRONMENTS ONLY**

## Latent Space Red Teaming Framework v1 - Analysis Reports

### Creating Analysis Reports

When asked to create an analysis report based on latent space red teaming (Framework v1) results, **always use the template** located at:

**`latent_space_framework/reports/templates/LATENT_SPACE_REDTEAM_ANALYSIS_TEMPLATE.md`**

### Template Usage

1. **Copy the template:**
   ```bash
   cp latent_space_framework/reports/templates/LATENT_SPACE_REDTEAM_ANALYSIS_TEMPLATE.md YOUR_MODEL_ANALYSIS.md
   ```

2. **Fill in placeholders** using data from the JSON/CSV reports:
   - `phase1_targets.json` - Layer classifications and compositional kappa
   - `gradient_attack_results.json` - FGSM, PGD, BIM, MIM attack results
   - `complete_analysis.json` - Full analysis data
   - `layer_summary.csv` - Per-layer κ, σ_min, σ_max values
   - `high_value_targets.json` - Priority attack targets
   - `attack_summary.json` - Attack results and exploit classifications
   - `phase5_reproducibility.json` - Reproducibility test results

3. **Reference the example report** for formatting guidance:
   - `latent_space_framework/reports/REDTEAM_ANALYSIS_REPORT.md` (if available)

### Report Structure

The template includes all required sections:
- Executive Summary with assessment table
- Methodology overview (6-phase pipeline)
- Detailed results for all 6 phases
- Technical findings (Jacobian analysis, singular values)
- Key findings and insights
- Red team and defender recommendations
- Visual scorecards (ASCII art)
- Risk matrices
- Metrics reference appendix

### Important Notes

- **Always use the template** - Do not create reports from scratch
- **Replace all placeholders** - Use `[BRACKETS]` format for placeholders
- **Maintain structure** - Keep the same sections and formatting
- **Include visuals** - Use ASCII art for scorecards and risk matrices
- **Save with timestamp** - Include date/time in filename to avoid overwriting
- **Key metrics:** Focus on κ (condition number), σ_max/σ_min (singular values), κ_comp (compositional kappa)

See `latent_space_framework/reports/templates/README.md` for detailed usage instructions.
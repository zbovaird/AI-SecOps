# AI SecOps Workspace

A comprehensive workspace for AI Security Operations, focused on red teaming AI/ML models using industry-standard tools like Microsoft PyRIT and IBM Adversarial Robustness Toolbox (ART).

## Overview

This workspace enables:
- **Blackbox Testing**: Test deployed models without internal access
- **Whitebox Testing**: Deep analysis with full model access
- **Workflow Testing**: End-to-end security assessment pipelines

## Quick Start

1. **Install Tools**:
   ```bash
   ./install_tools.sh
   ```

2. **Explore Capabilities**:
   ```bash
   python explore_tools_detailed.py
   ```

3. **Review Documentation**:
   - `plan.md` - Strategic plan and architecture
   - `instructions.md` - Setup and usage instructions
   - `tools_capabilities.md` - Detailed tool capabilities by testing type

## Project Structure

```
AI SecOps/
├── plan.md                      # Strategic plan
├── instructions.md              # Setup instructions
├── tools_capabilities.md        # Tool capabilities reference
├── requirements.txt             # Python dependencies
├── install_tools.sh             # Installation script
├── explore_tools_detailed.py    # Tool exploration script
└── README.md                    # This file
```

## Tools

### Microsoft PyRIT
Python Risk Identification Tool for red teaming generative AI systems.

**Key Strengths:**
- Prompt-based attack scenarios
- Endpoint testing (perfect for Vertex AI)
- Workflow orchestration
- Response scoring and evaluation

### IBM ART
Adversarial Robustness Toolbox for comprehensive model security evaluation.

**Key Strengths:**
- Extensive attack library (blackbox and whitebox)
- Defense mechanisms
- Robustness metrics
- Framework support (TensorFlow, PyTorch, Scikit-learn)

## Testing Types

### Blackbox Testing
- Test models via APIs/endpoints
- No internal model knowledge required
- Suitable for production deployments
- Tools: PyRIT prompt targets, ART query-based attacks

### Whitebox Testing
- Full access to model internals
- Gradient-based attacks
- Internal state analysis
- Tools: ART gradient attacks, PyRIT model access

### Workflow Testing
- End-to-end security assessment
- Multi-stage attack pipelines
- Comprehensive evaluation
- Tools: PyRIT orchestrators, ART evaluation frameworks

## Next Steps

1. Follow `instructions.md` for detailed setup
2. Review `tools_capabilities.md` to understand available features
3. Deploy a test model to Google Vertex AI
4. Run your first security assessment

## Resources

- [Microsoft PyRIT Documentation](https://azure.github.io/PyRIT/)
- [IBM ART Documentation](https://adversarial-robustness-toolbox.readthedocs.io/)
- [Google Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)

## License

[Add your license here]


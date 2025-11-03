# AI SecOps Workspace

A comprehensive workspace for AI Security Operations, focused on red teaming AI/ML models using industry-standard tools like Microsoft PyRIT and IBM Adversarial Robustness Toolbox (ART).

## Overview

This workspace enables:
- **Blackbox Testing**: Test deployed models without internal access
- **Whitebox Testing**: Deep analysis with full model access
- **Workflow Testing**: End-to-end security assessment pipelines
- **Red Team Testing**: Real target reconnaissance, exploitation, and post-exploitation capabilities

**⚠️ WARNING: FOR AUTHORIZED SECURITY TESTING IN SANDBOXED ENVIRONMENTS ONLY**

This toolkit is designed for legitimate security testing purposes only. Use only on systems you own or have explicit written authorization to test.

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
├── instructions.md              # Setup instructions (PyRIT/ART + Red Team Kit)
├── tools_capabilities.md        # Tool capabilities reference
├── requirements.txt             # Python dependencies (PyRIT/ART)
├── install_tools.sh             # Installation script
├── explore_tools_detailed.py    # Tool exploration script
├── pyrit_gradio_app.py          # PyRIT Gradio GUI
├── pyrit_gemini_api.py          # PyRIT Gemini integration
├── pyrit_vertex_deepseek.py    # PyRIT DeepSeek integration
├── prompt_injection_test_gemini.py  # Prompt injection testing
├── redteam_kit/                 # Red Team Testing Kit (real target capabilities)
│   ├── core/modules/            # Core exploitation modules
│   ├── utils/                   # Utilities (logger, config)
│   ├── examples/                # Usage examples
│   ├── instructions.md          # Red Team Kit usage instructions
│   ├── requirements.txt        # Red Team Kit dependencies
│   └── README.md                # Red Team Kit documentation
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

### Red Team Testing Kit
Custom framework for real target security testing and red team exercises.

**Key Features:**
- Real reconnaissance (port scanning, service detection, banner grabbing)
- Exploit testing (SQL injection, RCE, command injection, XXE)
- Post-exploitation (credential harvesting, privilege escalation, lateral movement)
- Persistence mechanisms (scheduled tasks, startup scripts, registry modification)
- Multi-stage attack chain orchestration
- Support for IP addresses, domains, URLs, and hostnames

See `redteam_kit/README.md` for detailed documentation.

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
   - PyRIT and IBM ART installation and usage
   - Red Team Kit attack chain usage for real targets
2. Review `tools_capabilities.md` to understand available features
3. For AI Model Testing:
   - Deploy a test model to Google Vertex AI
   - Run PyRIT/ART security assessments
4. For Real Target Testing:
   - See `redteam_kit/README.md` for Red Team Kit overview
   - See `redteam_kit/instructions.md` for detailed attack chain usage
5. Run your first security assessment

## Resources

- [Microsoft PyRIT Documentation](https://azure.github.io/PyRIT/)
- [IBM ART Documentation](https://adversarial-robustness-toolbox.readthedocs.io/)
- [Google Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)

## License

[Add your license here]


# PyRIT and IBM ART Capabilities by Testing Type

**Last Updated**: Based on actual installation exploration (PyRIT v0.9.0, IBM ART v1.20.1)

This document organizes the **actual discovered capabilities** of Microsoft PyRIT and IBM Adversarial Robustness Toolbox (ART) into three testing categories: **Blackbox**, **Whitebox**, and **Workflow Testing**.

---

## Microsoft PyRIT v0.9.0 Capabilities

### 📦 Blackbox Testing

Blackbox testing in PyRIT focuses on testing models through APIs/endpoints without internal model access.

#### Core Components:

**Models & Data Structures:**
- `pyrit.models.AzureBlobStorageIO` - Azure blob storage integration
- `pyrit.models.ChatMessage` - Chat message data structures
- `pyrit.models.ChatMessageListDictContent` - Message content handling
- `pyrit.models.ChatMessagesDataset` - Dataset for chat messages
- `pyrit.models.DataTypeSerializer` - Data serialization
- `pyrit.models.PromptRequestPiece` - Individual prompt request handling
- `pyrit.models.PromptRequestResponse` - Request/response structures
- `pyrit.models.PromptResponse` - Response handling
- `pyrit.models.SeedPrompt` - Seed prompt management
- `pyrit.models.SeedPromptDataset` - Seed prompt datasets
- `pyrit.models.SeedPromptGroup` - Grouped seed prompts
- `pyrit.models.StorageIO` - Storage I/O operations

**Scoring & Evaluation:**
- `pyrit.score.AzureContentFilterScorer` - Azure content filtering scorer
- `pyrit.score.InsecureCodeScorer` - Insecure code detection
- `pyrit.score.PromptShieldScorer` - Prompt shield evaluation
- `pyrit.score.SelfAskCategoryScorer` - Self-ask category evaluation
- `pyrit.score.SelfAskGeneralScorer` - General self-ask scoring

**Memory & Tracking:**
- `pyrit.memory.AzureSQLMemory` - Azure SQL memory storage
- `pyrit.memory.EmbeddingDataEntry` - Embedding data entries
- `pyrit.memory.MemoryInterface` - Memory interface abstraction
- `pyrit.memory.PromptMemoryEntry` - Prompt memory tracking
- `pyrit.memory.SeedPromptEntry` - Seed prompt entry tracking

#### Available Datasets (from `pyrit.datasets`):
- `fetch_adv_bench_dataset()` - AdvBench dataset
- `fetch_aya_redteaming_dataset()` - Aya red teaming dataset
- `fetch_babelscape_alert_dataset()` - Babelscape alerts
- `fetch_darkbench_dataset()` - DarkBench dataset
- `fetch_decoding_trust_stereotypes_dataset()` - Decoding trust stereotypes
- `fetch_forbidden_questions_dataset()` - Forbidden questions
- `fetch_harmbench_dataset()` - HarmBench dataset
- `fetch_many_shot_jailbreaking_dataset()` - Many-shot jailbreaking
- `fetch_red_team_social_bias_dataset()` - Social bias red teaming
- `fetch_tdc23_redteaming_dataset()` - TDC23 red teaming
- `fetch_wmdp_dataset()` - WMDP dataset
- `fetch_xstest_dataset()` - XSTest dataset
- And more...

#### Use Cases:
- Testing deployed endpoints (Vertex AI, OpenAI, Azure OpenAI)
- Prompt injection testing via APIs
- Jailbreak detection on production models
- Response quality analysis without model access
- Data extraction attempts through conversation

---

### 🔬 Whitebox Testing

Whitebox testing in PyRIT allows deeper inspection with access to local models and datasets.

#### Core Components:

**Models & Datasets:**
- `pyrit.models.ChatMessagesDataset` - Local chat message datasets
- `pyrit.models.DataTypeSerializer` - Data serialization
- `pyrit.models.DiskStorageIO` - Disk-based storage I/O
- `pyrit.models.QuestionAnsweringDataset` - QA datasets
- `pyrit.models.QuestionAnsweringEntry` - QA entries
- `pyrit.models.SeedPromptDataset` - Local seed prompt datasets
- `pyrit.models.StorageIO` - Storage operations

**Memory & Embeddings:**
- `pyrit.memory.AzureSQLMemory` - SQL memory storage
- `pyrit.memory.DuckDBMemory` - DuckDB memory storage
- `pyrit.memory.EmbeddingDataEntry` - Embedding data management
- `pyrit.memory.MemoryEmbedding` - Memory embedding operations
- `pyrit.memory.MemoryInterface` - Memory interface
- `pyrit.memory.SeedPromptEntry` - Seed prompt entries

#### Use Cases:
- Testing local models with full access
- Dataset analysis and manipulation
- Embedding-based analysis
- Internal prompt flow testing

---

### 🔄 Workflow Testing

Workflow testing combines multiple components for end-to-end security assessment pipelines.

#### Core Components:

**Scoring & Evaluation (Comprehensive):**
- `pyrit.score.AzureContentFilterScorer` - Azure content filter
- `pyrit.score.CompositeScorer` - Composite scoring system
- `pyrit.score.FloatScaleThresholdScorer` - Threshold-based scoring
- `pyrit.score.GandalfScorer` - Gandalf game scoring
- `pyrit.score.HumanInTheLoopScorer` - Human-in-the-loop evaluation
- `pyrit.score.HumanInTheLoopScorerGradio` - Gradio-based HITL scorer
- `pyrit.score.InsecureCodeScorer` - Insecure code detection
- `pyrit.score.MarkdownInjectionScorer` - Markdown injection detection
- `pyrit.score.PromptShieldScorer` - Prompt shield evaluation
- `pyrit.score.QuestionAnswerScorer` - Question-answer evaluation
- `pyrit.score.Scorer` - Base scorer interface
- `pyrit.score.SelfAskCategoryScorer` - Self-ask category evaluation
- `pyrit.score.SelfAskGeneralScorer` - General self-ask scoring
- `pyrit.score.SelfAskLikertScorer` - Likert scale self-ask
- `pyrit.score.SelfAskRefusalScorer` - Refusal detection
- `pyrit.score.SelfAskScaleScorer` - Scale-based self-ask
- `pyrit.score.SelfAskTrueFalseScorer` - True/false self-ask
- `pyrit.score.SubStringScorer` - Substring matching scorer
- `pyrit.score.TrueFalseInverterScorer` - True/false inversion

**Memory & State Management:**
- `pyrit.memory.AzureSQLMemory` - Centralized SQL memory
- `pyrit.memory.CentralMemory` - Central memory management
- `pyrit.memory.DuckDBMemory` - DuckDB memory storage
- `pyrit.memory.EmbeddingDataEntry` - Embedding tracking
- `pyrit.memory.MemoryEmbedding` - Embedding operations
- `pyrit.memory.MemoryExporter` - Memory export functionality
- `pyrit.memory.MemoryInterface` - Memory interface
- `pyrit.memory.PromptMemoryEntry` - Prompt memory tracking
- `pyrit.memory.SeedPromptEntry` - Seed prompt tracking

**Request Management:**
- `pyrit.models.PromptRequestPiece` - Request piece handling

#### Use Cases:
- Complete security audit workflows
- Automated red team exercises
- Multi-stage attack pipelines
- Compliance testing
- Continuous security monitoring
- End-to-end evaluation pipelines

---

## IBM Adversarial Robustness Toolbox (ART) v1.20.1 Capabilities

### 📦 Blackbox Testing

Blackbox attacks in ART use only model inputs and outputs, perfect for testing deployed endpoints like Vertex AI.

#### Evasion Attacks (Query-Based):

**Decision Boundary Attacks:**
- `art.attacks.evasion.BoundaryAttack` - Decision boundary-based attack
- `art.attacks.evasion.HopSkipJump` - Query-efficient boundary attack
- `art.attacks.evasion.SimBA` - Simple Black-Box Adversarial Attack
- `art.attacks.evasion.SquareAttack` - Query-efficient square attack
- `art.attacks.evasion.ZooAttack` - Zeroth order optimization attack

#### Attack Characteristics:
- **No gradient access required** - Works with API endpoints
- **Query-efficient** - Minimizes API calls
- **Decision-based** - Uses only model predictions
- **Transferable** - Can work across model types

#### Use Cases:
- Testing Vertex AI endpoints
- API-based model security testing
- Production model evaluation
- Query-efficient adversarial testing

---

### 🔬 Whitebox Testing

Whitebox attacks in ART require full access to model architecture, weights, and gradients.

#### Evasion Attacks (Gradient-Based):

**Fast Gradient Methods:**
- `art.attacks.evasion.FastGradientMethod` - FGSM attack
- `art.attacks.evasion.ProjectedGradientDescent` - PGD attack (iterative FGSM)
- `art.attacks.evasion.BasicIterativeMethod` - BIM attack
- `art.attacks.evasion.MomentumIterativeMethod` - Momentum-based PGD
- `art.attacks.evasion.AutoProjectedGradientDescent` - Auto-PGD
- `art.attacks.evasion.AutoConjugateGradient` - Auto-conjugate gradient
- `art.attacks.evasion.RescalingAutoConjugateGradient` - Rescaling variant

**Carlini-Wagner Attacks:**
- `art.attacks.evasion.CarliniL0Method` - C&W L0 attack
- `art.attacks.evasion.CarliniL2Method` - C&W L2 attack
- `art.attacks.evasion.CarliniLInfMethod` - C&W L_inf attack
- `art.attacks.evasion.CarliniWagnerASR` - C&W for ASR

**Optimization-Based Attacks:**
- `art.attacks.evasion.DeepFool` - DeepFool attack
- `art.attacks.evasion.ElasticNet` - EAD (Elastic-Net) attack
- `art.attacks.evasion.NewtonFool` - NewtonFool attack
- `art.attacks.evasion.Wasserstein` - Wasserstein distance attack

**Saliency & Feature Attacks:**
- `art.attacks.evasion.SaliencyMapMethod` - JSMA (Jacobian-based) attack
- `art.attacks.evasion.FeatureAdversariesNumpy` - Feature-level attacks (NumPy)
- `art.attacks.evasion.FeatureAdversariesPyTorch` - Feature-level attacks (PyTorch)
- `art.attacks.evasion.FeatureAdversariesTensorFlowV2` - Feature-level attacks (TF)

**Universal & Patch Attacks:**
- `art.attacks.evasion.UniversalPerturbation` - Universal adversarial perturbations
- `art.attacks.evasion.TargetedUniversalPerturbation` - Targeted universal perturbations
- `art.attacks.evasion.AdversarialPatch` - Patch-based attacks
- `art.attacks.evasion.AdversarialPatchNumpy` - Patch attacks (NumPy)
- `art.attacks.evasion.AdversarialPatchPyTorch` - Patch attacks (PyTorch)
- `art.attacks.evasion.AdversarialPatchTensorFlowV2` - Patch attacks (TensorFlow)
- `art.attacks.evasion.AdversarialTexturePyTorch` - Texture attacks
- `art.attacks.evasion.DPatch` - DPatch attack
- `art.attacks.evasion.RobustDPatch` - Robust DPatch

**Specialized Attacks:**
- `art.attacks.evasion.AutoAttack` - Ensemble of attacks
- `art.attacks.evasion.PixelAttack` - Pixel-level attack
- `art.attacks.evasion.ThresholdAttack` - Threshold attack
- `art.attacks.evasion.SpatialTransformation` - Spatial/geometric attacks
- `art.attacks.evasion.VirtualAdversarialMethod` - Virtual adversarial training
- `art.attacks.evasion.ShadowAttack` - Shadow attack
- `art.attacks.evasion.GeoDA` - Geometric Decision-based Attack
- `art.attacks.evasion.SNAL` - SNAL attack
- `art.attacks.evasion.LaserAttack` - Laser attack
- `art.attacks.evasion.LowProFool` - LowProFool
- `art.attacks.evasion.HighConfidenceLowUncertainty` - HCLU attack
- `art.attacks.evasion.SignOPTAttack` - Sign-OPT attack
- `art.attacks.evasion.FrameSaliencyAttack` - Frame saliency attack
- `art.attacks.evasion.MalwareGDTensorFlow` - Malware gradient descent
- `art.attacks.evasion.OverloadPyTorch` - Overload attack
- `art.attacks.evasion.OverTheAirFlickeringPyTorch` - OTA flickering
- `art.attacks.evasion.LaserAttack` - Laser attack
- `art.attacks.evasion.GRAPHITEBlackbox` - GRAPHITE (blackbox)
- `art.attacks.evasion.GRAPHITEWhiteboxPyTorch` - GRAPHITE (whitebox)
- `art.attacks.evasion.CompositeAdversarialAttackPyTorch` - Composite attacks
- `art.attacks.evasion.DecisionTreeAttack` - Decision tree attacks

**Audio/ASR Attacks:**
- `art.attacks.evasion.ImperceptibleASR` - Imperceptible ASR attack
- `art.attacks.evasion.ImperceptibleASRPyTorch` - Imperceptible ASR (PyTorch)

#### Poisoning Attacks (Training-Time):
- `art.attacks.poisoning.PoisoningAttackAdversarialEmbedding` - Adversarial embedding poisoning
- `art.attacks.poisoning.PoisoningAttackBackdoor` - Backdoor poisoning
- `art.attacks.poisoning.PoisoningAttackCleanLabelBackdoor` - Clean-label backdoor
- `art.attacks.poisoning.PoisoningAttackSVM` - SVM poisoning

#### Model Extraction Attacks:
- `art.attacks.extraction.CopycatCNN` - Copycat CNN extraction
- `art.attacks.extraction.FunctionallyEquivalentExtraction` - Functionally equivalent extraction
- `art.attacks.extraction.KnockoffNets` - Knockoff networks

#### Inference Attacks:
- `art.attacks.inference.attribute_inference` - Attribute inference
- `art.attacks.inference.membership_inference` - Membership inference
- `art.attacks.inference.model_inversion` - Model inversion
- `art.attacks.inference.reconstruction` - Reconstruction attacks

#### Defenses (Whitebox):

**Preprocessor Defenses:**
- `art.defences.preprocessor.FeatureSqueezing` - Feature squeezing
- `art.defences.preprocessor.GaussianAugmentation` - Gaussian augmentation
- `art.defences.preprocessor.JpegCompression` - JPEG compression defense
- `art.defences.preprocessor.Mp3Compression` - MP3 compression (audio)
- `art.defences.preprocessor.PixelDefend` - Pixel-level defense
- `art.defences.preprocessor.SpatialSmoothing` - Spatial smoothing
- `art.defences.preprocessor.ThermometerEncoding` - Thermometer encoding
- `art.defences.preprocessor.TotalVarMin` - Total variation minimization
- `art.defences.preprocessor.VideoCompression` - Video compression
- `art.defences.preprocessor.CutMix` / `Cutout` / `Mixup` - Data augmentation defenses
- `art.defences.preprocessor.LabelSmoothing` - Label smoothing
- `art.defences.preprocessor.Resample` - Resampling defense

#### Use Cases:
- Testing local models with full access
- Gradient-based vulnerability analysis
- Adversarial training
- Defense mechanism evaluation
- Model robustness certification

---

### 🔄 Workflow Testing

Workflow testing combines attacks, defenses, and metrics for comprehensive security evaluation.

#### Adversarial Training (Training-Time Defenses):

**Training Methods:**
- `art.defences.trainer.AdversarialTrainer` - Base adversarial trainer
- `art.defences.trainer.AdversarialTrainerMadryPGD` - Madry PGD training
- `art.defences.trainer.AdversarialTrainerTRADES` - TRADES training
- `art.defences.trainer.AdversarialTrainerTRADESPyTorch` - TRADES (PyTorch)
- `art.defences.trainer.AdversarialTrainerAWP` - Adversarial Weight Perturbation
- `art.defences.trainer.AdversarialTrainerAWPPyTorch` - AWP (PyTorch)
- `art.defences.trainer.AdversarialTrainerFBF` - Fast Batch Flip
- `art.defences.trainer.AdversarialTrainerFBFPyTorch` - FBF (PyTorch)
- `art.defences.trainer.AdversarialTrainerOAAT` - Once-At-A-Time training
- `art.defences.trainer.AdversarialTrainerOAATPyTorch` - OAAT (PyTorch)
- `art.defences.trainer.AdversarialTrainerCertifiedPytorch` - Certified training
- `art.defences.trainer.AdversarialTrainerCertifiedIBPPyTorch` - IBP certified training
- `art.defences.trainer.DPInstaHideTrainer` - DP InstaHide training

#### Detection Defenses:

**Evasion Detection:**
- `art.defences.detector.evasion` - Evasion attack detection

**Poisoning Detection:**
- `art.defences.detector.poison` - Poisoning attack detection

#### Metrics & Evaluation:

**Robustness Metrics:**
- `art.metrics.empirical_robustness` - Empirical robustness measurement
- `art.metrics.clever` - CLEVER score (Lipschitz constant estimation)
- `art.metrics.clever_t` - CLEVER-T (targeted)
- `art.metrics.clever_u` - CLEVER-U (untargeted)
- `art.metrics.adversarial_accuracy` - Accuracy on adversarial examples
- `art.metrics.loss_sensitivity` - Loss sensitivity analysis
- `art.metrics.loss_gradient_check` - Gradient checking

**Verification Metrics:**
- `art.metrics.RobustnessVerificationTreeModelsCliqueMethod` - Tree model verification
- `art.metrics.verification_decisions_trees` - Decision tree verification
- `art.metrics.gradient_check` - Gradient verification

**Privacy Metrics:**
- `art.metrics.privacy` - Privacy evaluation metrics

**Distance Metrics:**
- `art.metrics.wasserstein_distance` - Wasserstein distance calculation

**Advanced Metrics:**
- `art.metrics.SHAPr` - SHAP-based robustness
- `art.metrics.PDTP` - PDTP metric
- `art.metrics.ComparisonType` - Comparison types for evaluation

#### Use Cases:
- Complete security assessment workflows
- Adversarial training pipelines
- Robustness benchmarking
- Defense comparison studies
- Model certification processes
- End-to-end evaluation frameworks
- Production deployment validation

---

## Summary Statistics

### PyRIT v0.9.0
- **Blackbox Features**: 22 components
- **Whitebox Features**: 13 components  
- **Workflow Features**: 29 components

### IBM ART v1.20.1
- **Blackbox Features**: 5 evasion attacks
- **Whitebox Features**: 139+ attacks, defenses, and utilities
- **Workflow Features**: 46 training methods, detectors, and metrics

## Recommended Usage Patterns

### Blackbox Endpoint Testing
- **Primary**: PyRIT (prompt-based attacks, scoring, datasets)
- **Secondary**: ART blackbox evasion attacks (BoundaryAttack, HopSkipJump, etc.)
- **Best For**: Testing Vertex AI endpoints, production models

### Whitebox Model Analysis
- **Primary**: ART gradient-based attacks (PGD, C&W, DeepFool, etc.)
- **Secondary**: PyRIT local dataset management
- **Best For**: Local model security evaluation, adversarial training

### Complete Workflow Assessment
- **Primary**: PyRIT orchestrators + ART comprehensive evaluation
- **Components**: 
  - PyRIT scoring and memory management
  - ART adversarial training and metrics
- **Best For**: Full security audits, compliance testing, research

---

## Framework Support

### ART Framework Compatibility:
- **TensorFlow/Keras**: Full support
- **PyTorch**: Full support  
- **Scikit-learn**: Full support
- **NumPy**: Base support
- **ONNX**: Import/export support

### PyRIT Integration:
- **OpenAI**: Native support
- **Azure OpenAI**: Native support
- **Custom Endpoints**: Extensible support
- **Vertex AI**: Via custom chat targets

---

## Next Steps

1. **For Vertex AI Testing**: 
   - Use PyRIT prompt targets with custom endpoints
   - Combine with ART blackbox attacks via API wrappers

2. **For Local Model Testing**:
   - Use ART whitebox attacks directly
   - Integrate PyRIT datasets for prompt-based testing

3. **For Workflow Development**:
   - Combine PyRIT orchestrators with ART evaluation
   - Use PyRIT scorers with ART metrics for comprehensive reports

---

*This document is based on actual exploration of installed packages. Run `python explore_comprehensive.py` to regenerate from your installation.*

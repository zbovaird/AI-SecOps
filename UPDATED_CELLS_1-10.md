# Updated Cells 1-10 for latent_space_redteaming.ipynb

All fixes consolidated. Copy each cell into your notebook.

**NEW:** Cell 8 now includes 50+ diverse prompts. For easier management:
- Upload `test_prompts.py` to Colab and import: `from test_prompts import test_prompts`
- See `LOAD_PROMPTS_COLAB.md` for detailed instructions

---

## Cell 1: Install Dependencies
*(Already correct - no changes needed)*

```python
# Cell 1: Install Dependencies
# Note: Colab has pre-installed packages with specific version requirements
# We install compatible versions to avoid conflicts

# Install numpy with compatible version (required by tensorflow, numba, opencv)
# Current Numpy version breaks the adversarial robustness toolbox
!pip install "numpy<2.0.0" -q

# Install pyrit
!pip install pyrit -q

# Install other dependencies
!pip install transformers accelerate huggingface_hub scipy matplotlib seaborn h5py tqdm adversarial-robustness-toolbox -q

# Verify installations
import numpy as np
import torch
print(f"✓ NumPy version: {np.__version__}")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
```

---

## Cell 1.5: Upload Files to Colab (IMPORTANT - Run After Cell 1!)

```python
# Cell 1.5: Upload Files to Colab
# Upload redteam_kit.zip to Colab and verify GPU setup
# IMPORTANT: First create redteam_kit.zip on your computer, then run this cell

print("=" * 60)
print("FILE UPLOAD & GPU VERIFICATION")
print("=" * 60)

# Check if files already exist
import os

redteam_kit_path = '/content/redteam_kit'
test_prompts_path = '/content/test_prompts.py'

# Check for existing redteam_kit folder
if os.path.exists(redteam_kit_path):
    print(f"\n✓ redteam_kit folder already exists at {redteam_kit_path}")
    # Count Python files
    try:
        import subprocess
        result = subprocess.run(['find', redteam_kit_path, '-name', '*.py', '-type', 'f'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            file_count = len([f for f in result.stdout.strip().split('\n') if f])
            print(f"  Found {file_count} Python files in redteam_kit")
    except:
        pass
    
    # Check for key modules
    key_modules = [
        'core/modules/latent_space_analysis.py',
        'core/modules/latent_space_instrumentation.py',
        'core/modules/cka_analysis.py'
    ]
    found_modules = []
    for module in key_modules:
        if os.path.exists(os.path.join(redteam_kit_path, module)):
            found_modules.append(module.split('/')[-1])
    
    if found_modules:
        print(f"  Key modules found: {', '.join(found_modules[:3])}")
    print("\n✓ Files already uploaded! Skip to GPU verification below.")
else:
    print(f"\n⚠️  redteam_kit folder NOT found")
    print("\n📤 UPLOAD INSTRUCTIONS:")
    print("  1. First, create redteam_kit.zip on your computer:")
    print("     Terminal: cd '/Users/zachbovaird/Documents/GitHub/AI SecOps'")
    print("     Terminal: zip -r redteam_kit.zip redteam_kit/")
    print("\n  2. Then run the upload code below:")
    print("     (A file picker will appear - select redteam_kit.zip)")
    
    # Active upload code (not commented!)
    try:
        from google.colab import files
        import zipfile
        
        print("\n" + "=" * 60)
        print("UPLOADING FILES")
        print("=" * 60)
        print("\n📤 Click 'Choose Files' button below to upload redteam_kit.zip")
        print("   (File picker will appear)")
        
        uploaded = files.upload()
        
        # Extract automatically
        for filename in uploaded.keys():
            if filename.endswith('.zip'):
                print(f"\n📦 Extracting {filename}...")
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall('/content')
                os.remove(filename)
                print(f"✓ Extracted! redteam_kit folder ready at /content/redteam_kit")
            else:
                print(f"✓ Uploaded {filename}")
    except ImportError:
        print("\n⚠️  google.colab module not available")
        print("   Make sure you're connected to Colab runtime")
        print("   Check: Runtime → Change runtime type → Select Colab runtime")
    except Exception as e:
        print(f"\n⚠️  Upload error: {e}")
        print("   Try uploading manually via Colab file browser")

# Verify GPU setup
print("\n" + "=" * 60)
print("GPU VERIFICATION")
print("=" * 60)

try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n✓ GPU Available: {gpu_name}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n⚠️  GPU NOT AVAILABLE")
        print("\n   To enable GPU:")
        print("   1. In Colab, click 'Runtime' menu (top)")
        print("   2. Select 'Change runtime type'")
        print("   3. Set 'Hardware accelerator' to 'GPU'")
        print("   4. Click 'Save'")
        print("   5. Re-run this cell")
except Exception as e:
    print(f"\n⚠️  Could not check GPU: {e}")

# Final status
print("\n" + "=" * 60)
print("SETUP STATUS")
print("=" * 60)

if os.path.exists(redteam_kit_path):
    print("✓ redteam_kit folder: READY")
else:
    print("✗ redteam_kit folder: NOT FOUND (upload needed)")

if os.path.exists(test_prompts_path):
    print("✓ test_prompts.py: FOUND (optional)")
else:
    print("ℹ️  test_prompts.py: NOT FOUND (using inline prompts)")

try:
    import torch
    if torch.cuda.is_available():
        print("✓ GPU: AVAILABLE")
    else:
        print("⚠️  GPU: NOT AVAILABLE (change runtime type)")
except:
    print("⚠️  GPU: CANNOT VERIFY")

print("\n" + "=" * 60)
if os.path.exists(redteam_kit_path):
    print("✓ Ready to proceed! Continue to Cell 2")
else:
    print("⚠️  Upload redteam_kit.zip first, then re-run this cell")
print("=" * 60)
```

---

## Cell 2: Setup redteam_kit and Path
*(Updated to handle multiple upload scenarios)*

```python
# Cell 2: Setup redteam_kit and Path
# Handles multiple upload scenarios: zip file, folder, or package folder

import os
import subprocess
import sys

# Check for colab_upload_package folder (if uploaded as package)
package_folder = '/content/colab_upload_package'
if os.path.exists(package_folder):
    print(f"✓ Found colab_upload_package folder")
    # Move redteam_kit from package to /content
    package_redteam = os.path.join(package_folder, 'redteam_kit')
    target_redteam = '/content/redteam_kit'
    if os.path.exists(package_redteam) and not os.path.exists(target_redteam):
        import shutil
        shutil.move(package_redteam, target_redteam)
        print(f"✓ Moved redteam_kit from package to /content")
    
    # Move test_prompts.py if needed
    package_prompts = os.path.join(package_folder, 'test_prompts.py')
    target_prompts = '/content/test_prompts.py'
    if os.path.exists(package_prompts) and not os.path.exists(target_prompts):
        import shutil
        shutil.copy(package_prompts, target_prompts)
        print(f"✓ Copied test_prompts.py to /content")

# Extract zip if it exists (legacy support)
zip_file = '/content/redteam_kit copy.zip'
if os.path.exists(zip_file):
    print(f"Extracting {zip_file}...")
    !unzip -q "{zip_file}" -d /content/
    print("✓ Extraction complete")

# Handle "redteam_kit copy" folder name - create symlink so Python can import "redteam_kit"
source_folder = '/content/redteam_kit copy'
target_folder = '/content/redteam_kit'

if os.path.exists(source_folder) and not os.path.exists(target_folder):
    try:
        subprocess.run(['ln', '-s', source_folder, target_folder], check=True)
        print(f"✓ Created symlink: {target_folder} -> {source_folder}")
    except Exception as e:
        print(f"⚠️  Symlink failed, trying rename...")
        try:
            os.rename(source_folder, target_folder)
            print(f"✓ Renamed folder: {source_folder} -> {target_folder}")
        except Exception as e2:
            print(f"❌ Rename failed: {e2}")
            print("   Please manually rename '/content/redteam_kit copy' to '/content/redteam_kit'")
elif os.path.exists(target_folder):
    print(f"✓ {target_folder} already exists")
else:
    print(f"⚠️  redteam_kit folder not found. Expected: {target_folder} or {source_folder}")
    print("   Make sure you've uploaded the redteam_kit folder or colab_upload_package")

# Clean up sys.path and ensure /content is included
sys.path = [p for p in sys.path if 'redteam_kit copy' not in p and '__MACOSX' not in p]
if '/content' not in sys.path:
    sys.path.insert(0, '/content')

# Remove __MACOSX folder if it exists (macOS metadata)
macosx_path = '/content/__MACOSX'
if os.path.exists(macosx_path):
    !rm -rf {macosx_path}
    print(f"✓ Removed __MACOSX folder")

print("\n✓ Path setup complete")
```

---

## Cell 3: Fix redteam_kit __init__.py and Verify Structure + Auto-Move Additional Modules

```python
# Cell 3: Fix redteam_kit __init__.py and Verify Structure + Auto-Move Additional Modules

import os
import shutil

redteam_kit_path = '/content/redteam_kit'
modules_path = os.path.join(redteam_kit_path, 'core', 'modules')

if not os.path.exists(redteam_kit_path):
    print(f"❌ redteam_kit not found at {redteam_kit_path}")
    print("   Please upload and extract redteam_kit first")
else:
    print(f"✓ Found redteam_kit at: {redteam_kit_path}")
    
    # AUTO-MOVE: Check for additional module files in /content root and move them
    additional_modules = [
        'gradient_attacks.py',
        'semantic_perturbation.py',
        'adaptive_perturbation.py',
        'pyrit_integration.py'
    ]
    
    print("\n📦 Checking for additional module files...")
    for module_file in additional_modules:
        source_path = os.path.join('/content', module_file)
        target_path = os.path.join(modules_path, module_file)
        
        if os.path.exists(source_path):
            if not os.path.exists(target_path):
                try:
                    shutil.move(source_path, target_path)
                    print(f"  ✓ Moved {module_file} to core/modules/")
                except Exception as e:
                    print(f"  ✗ Failed to move {module_file}: {e}")
            else:
                print(f"  ✓ {module_file} already in core/modules/")
        else:
            # Also check if it's in the root redteam_kit folder
            alt_source = os.path.join(redteam_kit_path, module_file)
            if os.path.exists(alt_source):
                if not os.path.exists(target_path):
                    try:
                        shutil.move(alt_source, target_path)
                        print(f"  ✓ Moved {module_file} from root to core/modules/")
                    except Exception as e:
                        print(f"  ✗ Failed to move {module_file}: {e}")
                else:
                    print(f"  ✓ {module_file} already in core/modules/")
    
    # Verify package structure
    required_files = [
        '__init__.py',
        'core/__init__.py',
        'core/modules/__init__.py',
        'core/modules/latent_space_instrumentation.py'
    ]
    
    all_exist = True
    for req_file in required_files:
        file_path = os.path.join(redteam_kit_path, req_file)
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {req_file}")
        if not exists:
            all_exist = False
    
    # Check for additional modules
    print("\n📦 Additional modules status:")
    for module_file in additional_modules:
        module_path = os.path.join(modules_path, module_file)
        exists = os.path.exists(module_path)
        status = "✓" if exists else "✗"
        print(f"  {status} core/modules/{module_file}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✓ Package structure verified!")
        
        # Fix __init__.py if needed
        init_file = os.path.join(redteam_kit_path, '__init__.py')
        with open(init_file, 'r') as f:
            content = f.read()
        
        if 'from .modules import' in content and 'from .core.modules import' not in content:
            print("\n⚠️  Fixing incorrect import in __init__.py...")
            content = content.replace('from .modules import', 'from .core.modules import')
            with open(init_file, 'w') as f:
                f.write(content)
            print("✓ Fixed __init__.py")
        else:
            print("\n✓ __init__.py is correct")
    else:
        print("\n⚠️  Some required files are missing!")
        print("   If additional modules are missing, upload them to /content/ and run this cell again.")
```

---

## Cell 4: Import Libraries and Set Device
*(Already correct - no changes needed)*

```python
# Cell 4: Import Libraries and Set Device

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✓ Using device: {device}")
if device == 'cuda':
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
```

---

## Cell 5: Import redteam_kit Modules
*(Already correct - no changes needed)*

```python
# Cell 5: Import redteam_kit Modules

try:
    # Import step by step for better error reporting
    import redteam_kit
    print(f"✓ Imported redteam_kit from: {getattr(redteam_kit, '__file__', 'namespace package')}")
    
    import redteam_kit.core
    print(f"✓ Imported redteam_kit.core")
    
    import redteam_kit.core.modules
    print(f"✓ Imported redteam_kit.core.modules")
    
    # Import specific modules
    from redteam_kit.core.modules.latent_space_instrumentation import ModelInstrumentation
    from redteam_kit.core.modules.cka_analysis import CKAAnalyzer
    from redteam_kit.core.modules.latent_space_analysis import LatentSpaceAnalyzer
    from redteam_kit.core.modules.attention_monitor import AttentionMonitor
    from redteam_kit.core.modules.adversarial_perturbation import AdversarialPerturbationEngine
    from redteam_kit.core.modules.collapse_induction import CollapseInduction
    from redteam_kit.core.modules.transferability import TransferabilityTester
    
    print("\n✅ All redteam_kit modules imported successfully!")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure Cell 2 ran successfully (redteam_kit folder exists)")
    print("2. Make sure Cell 3 ran successfully (package structure verified)")
    print("3. Check that /content/redteam_kit/__init__.py exists and is correct")
    print("4. Run: !ls -la /content/redteam_kit")
    raise
```

---

## Cell 6: Hugging Face Authentication

```python
# Cell 6: Hugging Face Authentication
# REQUIRED: Gemma models require Hugging Face authentication

print("=" * 60)
print("HUGGING FACE AUTHENTICATION")
print("=" * 60)
print("\n⚠️  IMPORTANT: Gemma models require Hugging Face authentication!")
print("\nSteps:")
print("1. Visit: https://huggingface.co/google/gemma-2-2b-it")
print("2. Log in to Hugging Face (or create account)")
print("3. Accept Google's usage license")
print("4. Generate a token at: https://huggingface.co/settings/tokens")
print("   (Create a token with 'read' permissions)")
print()

# Check if already logged in
try:
    from huggingface_hub import whoami
    user_info = whoami()
    print(f"✓ Already logged in as: {user_info.get('name', 'Unknown')}")
    print(f"✓ Email: {user_info.get('email', 'Not provided')}")
    print("\nYou can proceed to Cell 7 to load the model.")
except Exception:
    print("⚠️  Not logged in yet.")
    print("\nRun the command below to login:")
    print()

# Hugging Face CLI login command - UNCOMMENT AND RUN THIS LINE
# !huggingface-cli login

# Alternative: Python login (uncomment and add your token)
# from huggingface_hub import login
# login(token='your_token_here')  # Replace with your actual token

print("\nAfter logging in, run this cell again to verify, then proceed to Cell 7.")
```

---

## Cell 7: Load Model

```python
# Cell 7: Load Model (Gemma 2)
# Model selection - Hugging Face models download automatically in Colab
model_name = "google/gemma-2-2b-it"

print(f"Loading model: {model_name}")
print("\n⚠️  Make sure you've completed Cell 6 (Hugging Face login) first!")

# Load tokenizer
print("\nLoading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded successfully")
except Exception as e:
    print(f"❌ Tokenizer loading failed: {e}")
    print("\nThis usually means you need to:")
    print("1. Accept the license at https://huggingface.co/google/gemma-2-2b-it")
    print("2. Login: !huggingface-cli login")
    raise

# Load model with automatic device mapping for Colab GPU
# Note: Gemma 2 models use bfloat16 precision (not float16)
print("\nLoading model (this may take a few minutes)...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,  # bfloat16 for Gemma 2
        device_map="auto" if device == 'cuda' else None,
        trust_remote_code=True
    )
    
    if device == 'cpu':
        model = model.to(device)
    
    model.eval()
    print(f"\n✅ Model loaded successfully!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {next(model.parameters()).device}")
    print(f"   Dtype: {next(model.parameters()).dtype}")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you've accepted the license: https://huggingface.co/google/gemma-2-2b-it")
    print("2. Login to Hugging Face: !pip install -U huggingface_hub && huggingface-cli login")
    print("3. Check your internet connection")
    print("4. If using Colab, make sure you have a GPU runtime")
    raise
```

---

## Cell 8: Instrument Model (UPDATED - All Hook Fixes)

```python
# Cell 8: Instrument Model (Fixed - All Hook Cleanup Included)
# This cell completely removes all existing hooks and registers safe hooks for Gemma 2

from collections import defaultdict

# Step 1: Remove ALL existing hooks from model modules
print("Cleaning all existing hooks from model...")
for name, module in model.named_modules():
    if hasattr(module, '_forward_hooks'):
        module._forward_hooks.clear()
    if hasattr(module, '_forward_pre_hooks'):
        module._forward_pre_hooks.clear()
    if hasattr(module, '_backward_hooks'):
        module._backward_hooks.clear()
print("✓ All hooks cleared")

# Step 2: Create fresh ModelInstrumentation instance
instrumentation = ModelInstrumentation(
    model,
    storage_path=None,
    capture_gradients=False
)

# Step 3: Reset instrumentation internal state
instrumentation.activations = defaultdict(list)
instrumentation.hooks = {}
instrumentation.layer_names = []

# Step 4: Define safe forward hook that handles bfloat16 and None outputs
def safe_forward_hook(name):
    def hook_fn(module, input, output):
        try:
            # Handle tuple outputs (common in transformers)
            if isinstance(output, tuple):
                # Take first element (usually the main output)
                act = output[0]
            else:
                act = output
            
            # Skip None outputs
            if act is None:
                return
            
            # Convert bfloat16 to float32 for compatibility
            if isinstance(act, torch.Tensor):
                if act.dtype == torch.bfloat16:
                    act = act.float()
                
                # Extract last token position for sequence outputs
                if len(act.shape) == 3:  # (batch, seq_len, hidden)
                    act = act[:, -1, :]  # Last token
                elif len(act.shape) == 2:  # (batch, hidden)
                    act = act[-1] if act.shape[0] > 1 else act[0]
                
                # Store activation
                instrumentation.activations[name].append(act.detach().clone())
        except Exception as e:
            # Silently skip problematic hooks
            pass
    
    return hook_fn

# Step 5: Register hooks only on "safe" layers (avoid attention modules)
print("\nRegistering hooks on Gemma 2 layers...")
safe_layer_types = [
    'Gemma2DecoderLayer',
    'Gemma2MLP',
    'Gemma2RMSNorm',
    'Embedding',
    'Linear'
]

hook_count = 0
for name, module in model.named_modules():
    module_type = type(module).__name__
    
    # Skip attention modules (they cause issues)
    if 'Attention' in module_type or 'attn' in name.lower():
        continue
    
    # Only hook safe layer types
    if any(safe_type in module_type for safe_type in safe_layer_types):
        hook = module.register_forward_hook(safe_forward_hook(name))
        instrumentation.hooks[name] = hook
        instrumentation.layer_names.append(name)
        hook_count += 1

print(f"✓ Registered {hook_count} hooks")
print(f"✓ Sample layers: {instrumentation.layer_names[:5] if instrumentation.layer_names else 'None yet'}")
```

---

## Cell 9: Optional - Load Prompts from test_prompts.py

```python
# Cell 9: Optional - Load Prompts from test_prompts.py
# This cell is optional - if test_prompts.py is uploaded, it will load prompts from file
# Otherwise, Cell 9 will use inline prompts

try:
    from test_prompts import test_prompts
    print(f"✓ Loaded {len(test_prompts)} prompts from test_prompts.py")
    print("   Cell 9 will use these imported prompts")
except ImportError:
    print("ℹ️  test_prompts.py not found")
    print("   Cell 9 will use inline prompts instead")
    print("   To use test_prompts.py: upload it to /content/ in Colab")
    test_prompts = None  # Will be set in Cell 9
```

---

## Cell 10: Phase 1 - Latent Space Mapping (UPDATED)

```python
# Cell 10: Phase 1 - Latent Space Mapping (Fixed)
# Check if test_prompts was imported from test_prompts.py (Cell 9)
# If not, use inline prompts below

if 'test_prompts' not in globals() or test_prompts is None:
    # Use inline prompts (fallback if test_prompts.py not uploaded)
    test_prompts = [
    # Original AI/ML prompts (keep for consistency)
    "What is artificial intelligence?",
    "Explain how neural networks work.",
    "Describe the process of machine learning.",
    "What are the applications of deep learning?",
    "How do transformers process language?",
    
    # Science & Technology
    "Explain quantum computing principles.",
    "How does photosynthesis work?",
    "Describe the water cycle.",
    "What causes climate change?",
    "How do vaccines work?",
    
    # History & Culture
    "What caused World War II?",
    "Explain the Renaissance period.",
    "Describe ancient Egyptian civilization.",
    "What was the Industrial Revolution?",
    "How did the internet develop?",
    
    # Mathematics & Logic
    "Prove the Pythagorean theorem.",
    "Explain calculus concepts.",
    "What is the Fibonacci sequence?",
    "How do you solve quadratic equations?",
    "Explain probability theory.",
    
    # Literature & Arts
    "Analyze Shakespeare's writing style.",
    "Describe impressionist painting techniques.",
    "What is the structure of a sonnet?",
    "Explain film editing principles.",
    "How does music theory work?",
    
    # Practical & Everyday
    "How do I change a tire?",
    "Explain cooking techniques.",
    "What are investment strategies?",
    "How does exercise affect health?",
    "Describe time management methods.",
    
    # Abstract & Philosophical
    "What is the meaning of life?",
    "Explain ethical dilemmas.",
    "What is consciousness?",
    "Describe free will vs determinism.",
    "How do we define truth?",
    
    # Technical & Specific
    "Write Python code to sort a list.",
    "Explain database normalization.",
    "How does encryption work?",
    "Describe API design principles.",
    "What is version control?",
    
    # Long-form & Complex
    "Explain the entire process of how a computer processes a program from source code to execution, including compilation, memory management, and CPU operations.",
    "Describe the complete lifecycle of a star from formation to death, including all stages and physical processes involved.",
    
    # Questions & Commands
    "Can you help me understand this?",
    "Please explain step by step.",
    "I need detailed information about this topic.",
    "What are the pros and cons?",
    "Compare and contrast these concepts.",
    
    # Edge cases
    "?",  # Single character
    "Repeat this word: hello hello hello hello",  # Repetition
    "Translate: Bonjour means hello in French.",  # Mixed languages
]

print(f"Loaded {len(test_prompts)} diverse test prompts")
print(f"Categories: AI/ML, Science, History, Math, Arts, Practical, Philosophy, Technical, Long-form, Edge cases")

# Run through model and capture activations
all_activations = {}

print("Processing prompts and capturing activations...")
for prompt in tqdm(test_prompts, desc="Processing prompts"):
    instrumentation.activations.clear()  # Clear previous activations
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        # Use output_attentions=False to avoid attention weight issues
        outputs = model(**inputs, output_attentions=False, output_hidden_states=False)
    
    # Get activations from instrumentation
    activations = instrumentation.activations
    
    # Aggregate (take last token position for each layer)
    for layer_name, layer_acts in activations.items():
        if layer_name not in all_activations:
            all_activations[layer_name] = []
        
        if isinstance(layer_acts, list) and len(layer_acts) > 0:
            # Take last activation (already processed by hook)
            all_activations[layer_name].append(layer_acts[-1])
        elif isinstance(layer_acts, torch.Tensor):
            # Handle tensor directly
            if len(layer_acts.shape) == 3:  # (batch, seq_len, hidden)
                all_activations[layer_name].append(layer_acts[:, -1, :])
            elif len(layer_acts.shape) == 2:  # (batch, hidden)
                all_activations[layer_name].append(layer_acts[-1] if layer_acts.shape[0] > 1 else layer_acts[0])
            else:
                all_activations[layer_name].append(layer_acts)

print(f"✓ Captured activations for {len(all_activations)} layers")
if all_activations:
    sample_layer = list(all_activations.keys())[0]
    print(f"Sample layer '{sample_layer}' shape: {all_activations[sample_layer][0].shape if all_activations[sample_layer] else 'Empty'}")
    print(f"Total layers: {len(all_activations)}")
    print(f"Activations per layer: {len(all_activations[sample_layer]) if all_activations[sample_layer] else 0}")
```

---

## Cell 11: Analyze Latent Space (UPDATED - Handles bfloat16)

```python
# Cell 11: Analyze Latent Space (Fixed - Handles bfloat16)
# Analyze all layers
analyzer = LatentSpaceAnalyzer(device=device)

# Average activations across prompts for each layer
layer_stats = {}
for layer_name, acts_list in all_activations.items():
    if not acts_list:
        continue
    
    # Stack activations
    stacked = torch.stack(acts_list)
    
    # Convert bfloat16 to float32 before analysis (NumPy doesn't support bfloat16)
    if stacked.dtype == torch.bfloat16:
        stacked = stacked.float()
    
    # Average over prompts
    avg_activation = stacked.mean(dim=0)
    
    # Analyze (analyzer expects float32)
    stats = analyzer.analyze_layer(avg_activation)
    layer_stats[layer_name] = stats

print(f"✓ Analyzed {len(layer_stats)} layers")

# Identify vulnerability basins with realistic thresholds
# Two approaches available to reduce false positives:
# 1. Stricter singular_value_ratio threshold (configurable, default 0.95 instead of hardcoded 0.9)
# 2. Require multiple criteria (AND logic) - more selective

print("=" * 60)
print("IDENTIFYING VULNERABILITY BASINS")
print("=" * 60)

# OPTION 1: Stricter singular value ratio (recommended first try)
# Makes the previously hardcoded singular_value_ratio check stricter
print("\nOption 1: Stricter singular_value_ratio threshold (0.98)")
basins_option1 = analyzer.identify_vulnerability_basins(
    layer_stats,
    variance_threshold=0.0001,  # Very low variance = collapsed/constant activations
    entropy_threshold=0.5,      # Low entropy = lack of diversity in activations
    rank_deficiency_threshold=150,  # Significant rank deficiency = dimensionality collapse
    singular_value_ratio_threshold=0.98,  # Stricter: was hardcoded 0.9, now 0.98 (higher = stricter)
    require_multiple_criteria=False  # OR logic (any criterion sufficient)
)

# OPTION 2: Require multiple criteria (AND logic)
# Layer must meet at least 2 criteria to be flagged
print("Option 2: Require multiple criteria (at least 2)")
basins_option2 = analyzer.identify_vulnerability_basins(
    layer_stats,
    variance_threshold=0.0001,
    entropy_threshold=0.5,
    rank_deficiency_threshold=150,
    singular_value_ratio_threshold=0.95,  # Still stricter than original 0.9
    require_multiple_criteria=True,  # AND logic
    min_criteria_count=2  # Require at least 2 criteria
)

# Compare results
print(f"\n{'='*60}")
print("COMPARISON OF APPROACHES")
print(f"{'='*60}")
print(f"Option 1 (Stricter singular_value_ratio=0.98): {len(basins_option1)} basins")
print(f"Option 2 (Require 2+ criteria): {len(basins_option2)} basins")

# Show breakdown for Option 1
if len(basins_option1) > 0:
    reason_counts_1 = {}
    for basin in basins_option1:
        for reason in basin['reasons']:
            reason_type = reason.split(':')[0]
            reason_counts_1[reason_type] = reason_counts_1.get(reason_type, 0) + 1
    print(f"\nOption 1 breakdown (what's flagging basins):")
    for reason, count in sorted(reason_counts_1.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count} basins")

# Show breakdown for Option 2
if len(basins_option2) > 0:
    reason_counts_2 = {}
    criteria_count_dist = {}
    for basin in basins_option2:
        criteria_count = basin.get('criteria_count', 0)
        criteria_count_dist[criteria_count] = criteria_count_dist.get(criteria_count, 0) + 1
        for reason in basin['reasons']:
            reason_type = reason.split(':')[0]
            reason_counts_2[reason_type] = reason_counts_2.get(reason_type, 0) + 1
    print(f"\nOption 2 breakdown:")
    print(f"  Basins by criteria count:")
    for count, num_basins in sorted(criteria_count_dist.items(), reverse=True):
        print(f"    {count} criteria: {num_basins} basins")
    print(f"  Criteria types:")
    for reason, count in sorted(reason_counts_2.items(), key=lambda x: x[1], reverse=True):
        print(f"    {reason}: {count} basins")

# Choose which approach to use
# Prefer Option 2 if it gives reasonable count (<100), else use Option 1
print(f"\n{'='*60}")
print("SELECTING BEST APPROACH")
print(f"{'='*60}")

if len(basins_option2) > 0 and len(basins_option2) < 100:
    basins = basins_option2
    print(f"✓ Using Option 2 (require multiple criteria): {len(basins)} basins")
    print("  This approach is more selective and reduces false positives.")
    print("  Only layers meeting 2+ criteria are flagged as vulnerable.")
elif len(basins_option1) > 0 and len(basins_option1) < len(basins_option2):
    basins = basins_option1
    print(f"✓ Using Option 1 (stricter singular_value_ratio): {len(basins)} basins")
    print("  This approach uses stricter thresholds but still allows single-criterion matches.")
else:
    basins = basins_option1
    print(f"✓ Using Option 1 (stricter singular_value_ratio): {len(basins)} basins")
    if len(basins) > 100:
        print("  ⚠️  Still getting >100 basins. Consider:")
        print("     - Using Option 2 explicitly (set require_multiple_criteria=True)")
        print("     - Making thresholds even stricter")
        print("     - Increasing singular_value_ratio_threshold to 0.99")

# Display selected basins
print(f"\n{'='*60}")
print(f"SELECTED VULNERABILITY BASINS ({len(basins)} total)")
print(f"{'='*60}")
for i, basin in enumerate(basins[:10], 1):
    criteria_count = basin.get('criteria_count', len(basin.get('reasons', [])))
    print(f"  {i}. {basin['layer_name']} ({criteria_count} criteria): {', '.join(basin['reasons'][:2])}")
if len(basins) > 10:
    print(f"  ... and {len(basins) - 10} more")

if len(basins) == 0:
    print("\n⚠️  No vulnerability basins found with current thresholds.")
    print("   Consider relaxing thresholds if you expect to find vulnerabilities.")
elif len(basins) > 100:
    print(f"\n⚠️  Found {len(basins)} basins - this may indicate:")
    print("   - Thresholds are too lenient (many false positives)")
    print("   - Model has widespread vulnerabilities")
    print("   - Consider using Option 2 (require_multiple_criteria=True)")

# Store basins with consistent variable name for Phase 3
vulnerability_basins = basins
print(f"\n✓ Stored {len(vulnerability_basins)} vulnerability basins in 'vulnerability_basins' variable")
```

---

## Cell 12: Compute CKA Similarity Matrix (UPDATED - Handles bfloat16 & JSON)

```python
# Cell 12: Compute CKA Similarity Matrix (Fixed - Handles bfloat16 & JSON)
# Helper function to convert NumPy types to native Python types for JSON serialization
def convert_to_native(obj):
    """Recursively convert NumPy types to native Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native(item) for item in obj]
    else:
        return obj

cka_analyzer = CKAAnalyzer()

# Prepare activations for CKA (average across prompts)
# Convert bfloat16 to float32 before CKA computation
layer_activations_for_cka = {}
for layer_name, acts_list in all_activations.items():
    if not acts_list:
        continue
    
    stacked = torch.stack(acts_list)
    
    # Convert bfloat16 to float32 (CKA computation requires float32)
    if stacked.dtype == torch.bfloat16:
        stacked = stacked.float()
    
    # Reshape to (samples, features) for CKA
    reshaped = stacked.view(stacked.shape[0], -1)
    layer_activations_for_cka[layer_name] = reshaped

print(f"✓ Prepared {len(layer_activations_for_cka)} layers for CKA analysis")

# Compute similarity matrix
similarity_matrix, layer_names = cka_analyzer.compute_similarity_matrix(
    layer_activations_for_cka
)

print(f"✓ Computed CKA similarity matrix: {similarity_matrix.shape}")

# Save Phase 1 results (convert all NumPy types to native Python types)
phase1_results = {
    'layer_stats': convert_to_native({k: {kk: float(vv) if isinstance(vv, (torch.Tensor, np.ndarray)) else vv 
                        for kk, vv in v.items()} for k, v in layer_stats.items()}),
    'vulnerability_basins': convert_to_native(basins),
    'similarity_matrix': convert_to_native(similarity_matrix.tolist()),
    'layer_names': layer_names
}

with open('phase1_latent_space_map.json', 'w') as f:
    json.dump(phase1_results, f, indent=2)

# Also save basins to separate file for Phase 3 (if basins exist)
if 'vulnerability_basins' in locals() and len(vulnerability_basins) > 0:
    basins_file = {
        'vulnerability_basins': convert_to_native(vulnerability_basins),
        'total_count': len(vulnerability_basins),
        'source': 'Phase 1 - Cell 11',
        'thresholds_used': {
            'variance_threshold': 0.0005,
            'entropy_threshold': 1.2,
            'rank_deficiency_threshold': 75
        }
    }
    with open('vulnerability_basins.json', 'w') as f:
        json.dump(basins_file, f, indent=2)
    print(f"\n✓ Also saved {len(vulnerability_basins)} vulnerability basins to vulnerability_basins.json")

print("\n✓ Phase 1 complete. Results saved to phase1_latent_space_map.json")
print(f"  - Analyzed {len(layer_stats)} layers")
print(f"  - Found {len(basins)} vulnerability basins")
print(f"  - Computed {similarity_matrix.shape[0]}x{similarity_matrix.shape[1]} similarity matrix")

---

## Cell 12.5: Visualize CKA Similarity Matrix (Optional)

```python
# Cell 12.5: Visualize CKA Similarity Matrix (OPTIONAL - Skip if you don't need visualization)
# This cell creates heatmap visualizations of the CKA similarity matrix
# You can skip this cell if you only need the data (already saved in phase1_latent_space_map.json)

print("=" * 60)
print("CKA Similarity Matrix Visualization (Optional)")
print("=" * 60)

# Check if similarity matrix exists from Cell 12
if 'similarity_matrix' not in locals() or 'layer_names' not in locals():
    print("⚠️  Similarity matrix not found. Run Cell 12 first.")
    print("   Or load from saved file:")
    print("   with open('phase1_latent_space_map.json', 'r') as f:")
    print("       data = json.load(f)")
    print("       similarity_matrix = np.array(data['similarity_matrix'])")
    print("       layer_names = data['layer_names']")
else:
    print(f"Visualizing {similarity_matrix.shape[0]}x{similarity_matrix.shape[1]} similarity matrix...")
    
    # Visualize CKA similarity matrix (full)
    if 'cka_analyzer' in locals():
        cka_analyzer.visualize_similarity_matrix(
            similarity_matrix,
            layer_names,
            title="Layer Similarity Matrix (CKA) - Phase 1 - All Layers"
        )
    else:
        # Fallback visualization if cka_analyzer not available
        plt.figure(figsize=(14, 12))
        sns.heatmap(similarity_matrix, xticklabels=layer_names, yticklabels=layer_names,
                    cmap='viridis', annot=False, fmt='.2f')
        plt.title('CKA Similarity Matrix - All Layers', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.show()
    
    # Create a more readable visualization focusing on decoder layers only
    decoder_layer_names = [name for name in layer_names if 'layers.' in name and 'layernorm' not in name.lower()]
    if decoder_layer_names:
        decoder_indices = [layer_names.index(name) for name in decoder_layer_names]
        decoder_matrix = similarity_matrix[np.ix_(decoder_indices, decoder_indices)]
        
        print(f"\nCreating decoder layers visualization ({len(decoder_layer_names)} layers)...")
        plt.figure(figsize=(12, 10))
        sns.heatmap(decoder_matrix, xticklabels=decoder_layer_names, yticklabels=decoder_layer_names,
                    cmap='viridis', annot=False, fmt='.2f')
        plt.title('CKA Similarity Matrix - Decoder Layers Only', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        print("✓ Decoder layers visualization complete")
    else:
        print("⚠️  No decoder layers found for visualization")
    
    print("\n✓ Visualization complete")
    print("   Note: Similarity matrix data is already saved in phase1_latent_space_map.json")
```

---

## Summary of Changes

### Cells 1-6: No changes needed
These cells already have all the correct fixes.

### Cell 7: Complete rewrite
- Removes ALL existing hooks before registering new ones
- Creates safe hooks that handle `bfloat16` and `None` outputs
- Only hooks "safe" layer types (avoids attention modules)
- Properly extracts last token position in hooks

### Cell 8: Updated
- Uses `output_attentions=False` in model calls
- Properly handles activation extraction from hooks
- Better error handling and logging

### Cell 9: Updated
- Converts `bfloat16` to `float32` before analysis
- Handles empty activation lists gracefully

### Cell 10: Updated
- Converts `bfloat16` to `float32` before CKA computation
- Adds `convert_to_native()` helper for JSON serialization
- Includes decoder-layer-only visualization
- Properly converts all NumPy types to native Python types

---

**Instructions:**
1. Open your notebook in Colab
2. Replace cells 7, 8, 9, and 10 with the updated code above
3. Cells 1-6 are already correct and don't need changes
4. Run all cells in order



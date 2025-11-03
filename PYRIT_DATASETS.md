# PyRIT Built-in Datasets

**Source**: These datasets come with PyRIT (Microsoft), downloaded from various sources for red teaming.

**Total**: 19 datasets available via `pyrit.datasets`

---

## Complete Dataset List

1. **`fetch_adv_bench_dataset()`** - AdvBench dataset
2. **`fetch_aya_redteaming_dataset()`** - Aya red teaming dataset
3. **`fetch_babelscape_alert_dataset()`** - Babelscape alerts dataset
4. **`fetch_darkbench_dataset()`** - DarkBench dataset
5. **`fetch_decoding_trust_stereotypes_dataset()`** - Decoding Trust stereotypes
6. **`fetch_examples()`** - Example prompts
7. **`fetch_forbidden_questions_dataset()`** - Forbidden questions dataset
8. **`fetch_harmbench_dataset()`** - HarmBench (400 prompts)
9. **`fetch_librAI_do_not_answer_dataset()`** - LibrAI dataset
10. **`fetch_llm_latent_adversarial_training_harmful_dataset()`** - LLM latent adversarial training
11. **`fetch_many_shot_jailbreaking_dataset()`** - Many-shot jailbreaking attempts
12. **`fetch_mlcommons_ailuminate_demo_dataset()`** - MLCommons AILuminate demo
13. **`fetch_multilingual_vulnerability_dataset()`** - Multilingual vulnerability testing
14. **`fetch_pku_safe_rlhf_dataset()`** - PKU Safe RLHF dataset
15. **`fetch_red_team_social_bias_dataset()`** - Social bias red teaming
16. **`fetch_seclists_bias_testing_dataset()`** - SecLists bias testing
17. **`fetch_tdc23_redteaming_dataset()`** - TDC23 red teaming dataset
18. **`fetch_wmdp_dataset()`** - WMDP (World Model Dangerous Prompts) dataset
19. **`fetch_xstest_dataset()`** - XSTest dataset

---

## Usage Example

```python
from pyrit.datasets import fetch_harmbench_dataset

# Load a dataset
dataset = fetch_harmbench_dataset()

# Dataset is a SeedPromptDataset with prompts attribute
print(f"Dataset contains: {len(dataset.prompts)} prompts")

# Access prompts via .prompts attribute
for prompt in dataset.prompts:
    # Each prompt is a SeedPrompt object with 'value' attribute
    print(prompt.value)  # The prompt text is in .value
    print(f"Categories: {prompt.harm_categories}")
    # Use with PromptRequestPiece for red teaming
```

## Dataset Format

All datasets return `SeedPromptDataset` objects which contain:
- `.prompts` attribute: List of `SeedPrompt` objects
- Each `SeedPrompt` has:
  - `.value`: The prompt text
  - `.harm_categories`: List of harm categories
  - `.dataset_name`: Name of the dataset
  - `.description`: Dataset description
  - And more metadata

## Example: Exploring HarmBench

```python
from pyrit.datasets import fetch_harmbench_dataset

dataset = fetch_harmbench_dataset()
print(f"Type: {type(dataset)}")
print(f"Contains: {len(dataset)} prompts")

# Get first prompt
first_prompt = next(iter(dataset))
print(f"Sample: {first_prompt.original_prompt_text[:100]}...")
```

## Use Cases

These datasets are designed for:

- **Red Team Testing**: Pre-built attack prompts
- **Jailbreak Attempts**: Prompt injection scenarios
- **Bias Testing**: Social bias evaluation prompts
- **Security Testing**: Vulnerability assessment
- **Benchmarking**: Standardized test suites

---

**Note**: These are real datasets included with PyRIT. They are downloaded from their respective sources when first accessed.


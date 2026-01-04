# Behavioral Red Teaming Analysis Report
## [MODEL_NAME] Model

**Analysis Date:** [DATE]  
**Framework:** Behavioral Red Teaming Framework v2  
**Model:** `[MODEL_ID]`  
**Architecture:** [NUM_LAYERS] layers, [NUM_HEADS] attention heads, [HIDDEN_SIZE] hidden size

---

## 📊 Executive Summary

This comprehensive behavioral red teaming analysis evaluated `[MODEL_ID]` across **5 distinct attack vectors** using black-box probing techniques. The analysis identified **[VULNERABILITY_LEVEL] vulnerabilities** in [KEY_AREAS], while demonstrating **[STRENGTH_LEVEL] resilience** against [STRONG_AREAS].

### Overall Vulnerability Score: **[SCORE]** ([PERCENTAGE]% vulnerable)

| Attack Vector | Vulnerability Score | Risk Level |
|--------------|-------------------|------------|
| **Decode Fragility** | [SCORE] | [RISK_LEVEL] |
| **Logit Lens** | [SCORE] | [RISK_LEVEL] |
| **Multi-turn Drift** | [SCORE] | [RISK_LEVEL] |
| **Attention Routing** | [SCORE] | [RISK_LEVEL] |
| **KV Cache** | [SCORE] | [RISK_LEVEL] |

**Key Finding:** [KEY_FINDING_DESCRIPTION]

---

## 🔬 Methodology Overview

### Framework Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Behavioral Red Teaming Framework v2              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Decode Fragility Sweep                              │
│     └─> Grid search: temp × top_p × rep_penalty        │
│                                                          │
│  2. Logit Lens Probing                                  │
│     └─> Layer-by-layer refusal decision analysis        │
│                                                          │
│  3. Multi-turn Residual Drift                           │
│     └─> Conversation strategies to erode safety        │
│                                                          │
│  4. Attention Routing Graph                             │
│     └─> Sink token identification                       │
│                                                          │
│  5. KV-Cache Persistence                                │
│     └─> Context retention analysis                      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Evaluation Metrics

- **Refusal Rate:** Percentage of prompts that trigger safety refusals
- **Compliance Rate:** Percentage of prompts that bypass safety
- **Knife-Edge Prompts:** Prompts that flip between refusal/compliance with small parameter changes
- **Critical Layers:** Layers where refusal decisions form or flip
- **Drift Magnitude:** Hidden state divergence from baseline
- **KV Half-Life:** Turns until context decays by 50%

---

## 📈 Detailed Experiment Results

### 1. Decode Fragility Analysis

**Objective:** Identify "knife-edge" prompts that flip between refusal and compliance with small decoding parameter changes.

#### Results Summary

| Metric | Value |
|--------|-------|
| **Total Prompts Tested** | [NUMBER] |
| **Grid Size** | [NUMBER] configurations per prompt |
| **Knife-Edge Prompts** | [NUMBER] ([PERCENTAGE]%) |
| **Always Refuses** | [NUMBER] |
| **Always Complies** | [NUMBER] |
| **Fragility Score** | **[SCORE]** |

#### Most Dangerous Configuration

```
Temperature: [TEMP]
Top-p: [TOP_P]
Repetition Penalty: [REP_PENALTY]
Compliance Rate: [RATE]% ⚠️
```

#### Safest Configuration

```
Temperature: [TEMP]
Top-p: [TOP_P]
Repetition Penalty: [REP_PENALTY]
Compliance Rate: [RATE]% ✓
```

#### Knife-Edge Prompt Example

**Prompt:** `"[EXAMPLE_PROMPT]"`

**Behavior:**
- Refusal Rate: [RATE]%
- Compliance Rate: [RATE]%
- Flip Count: **[NUMBER] transitions** across parameter space
- **Critical Transition:** [DESCRIPTION]

#### Compliance Rate Heatmap

| Temp | Top-p=0.9, Rep=1.0 | Top-p=0.9, Rep=1.1 | Top-p=1.0, Rep=1.0 | Top-p=1.0, Rep=1.1 |
|------|-------------------|-------------------|-------------------|-------------------|
| 0.0  | [RATE]%             | **[RATE]%**         | [RATE]%             | **[RATE]%**         |
| 0.3  | [RATE]%             | [RATE]%             | [RATE]%             | [RATE]%             |
| 0.7  | **[RATE]%**         | [RATE]%             | [RATE]%             | **[RATE]%** ⚠️      |
| 1.0  | [RATE]%             | [RATE]%             | [RATE]%             | [RATE]% ✓           |

**Red Team Insight:** [INSIGHT_DESCRIPTION]

**Defender Action:** [ACTION_DESCRIPTION]

---

### 2. Logit Lens Analysis

**Objective:** Identify which transformer layers form refusal decisions and where commitment solidifies.

#### Results Summary

| Metric | Value |
|--------|-------|
| **Vulnerability Score** | **[SCORE]** ([LEVEL]) 🔴 |
| **Average First Refusal Layer** | **[LAYER]** |
| **Average Commitment Layer** | **[LAYER]** |
| **Critical Layers** | [LAYER_LIST] |
| **Most Divergent Layer** | **[LAYER]** (KL divergence: [VALUE]) |

#### Layer-by-Layer Decision Formation

```
Layer 0:  ████░░░░░░░░░░░░░░░░░░░░░░  Initial token probabilities
Layer [FIRST_REFUSAL]:  ████████████░░░░░░░░░░░░░░  ⚠️ FIRST REFUSAL SIGNAL
Layer [MID]: ████████████████████░░░░░░  Refusal decision strengthening
Layer [COMMITMENT]: ██████████████████████████  ✓ COMMITMENT SOLIDIFIED
Layer [FINAL]: ██████████████████████████  Final output decision
```

#### Critical Layers Analysis

| Layer | KL Divergence | Role |
|-------|--------------|------|
| **[LAYER]** | [VALUE] | [DESCRIPTION] |
| **[LAYER]** | [VALUE] | ⚠️ **First refusal detection** |
| **[LAYER]** | [VALUE] | Refusal decision amplification |
| **[LAYER]** | [VALUE] | **Commitment point** - highest divergence |
| **[LAYER]** | [VALUE] | Final output layer |

#### Exploitability Window

```
Layers [START]-[END]: DECISION FORMATION WINDOW
├─ Layer [START]:  Refusal signal first appears
├─ Layer [MID]: Decision strengthens
└─ Layer [END]: Commitment solidifies

Attack Vector: Inject perturbations between layers [START]-[END]
to disrupt refusal decision before commitment.
```

**Red Team Insight:** [INSIGHT_DESCRIPTION]

**Defender Action:** [ACTION_DESCRIPTION]

---

### 3. Multi-Turn Residual Drift Analysis

**Objective:** Test if conversation strategies can gradually erode safety boundaries over multiple turns.

#### Results Summary

| Metric | Value |
|--------|-------|
| **Total Conversations** | [NUMBER] |
| **Strategies Tested** | [NUMBER] ([STRATEGY_LIST]) |
| **Bypass Rate** | **[RATE]%** ✓ |
| **Average Turns to Flip** | [VALUE] |
| **Cumulative Drift** | [VALUE] (max) |

#### Strategy Performance

| Strategy | Conversations | Bypass Rate | Avg Drift |
|----------|--------------|-------------|-----------|
| **[STRATEGY]** | [NUMBER] | [RATE]% | [VALUE] |
| **[STRATEGY]** | [NUMBER] | [RATE]% | [VALUE] |
| **[STRATEGY]** | [NUMBER] | [RATE]% | [VALUE] |

#### Drift Trajectory Example

**Conversation:** [STRATEGY] strategy
```
Turn 0: Drift = 0.0    (Baseline)
Turn 1: Drift = [VALUE]   (Initial shift)
Turn 2: Drift = [VALUE]    (Absorbed)
...
Final:  Drift = [VALUE]   (Contained)
```

**Finding:** [FINDING_DESCRIPTION]

**Red Team Insight:** [INSIGHT_DESCRIPTION]

**Defender Action:** [ACTION_DESCRIPTION]

---

### 4. Attention Routing Analysis

**Objective:** Identify "sink tokens" and attackable attention heads that route adversarial inputs differently than benign ones.

#### Results Summary

| Metric | Value |
|--------|-------|
| **Status** | [STATUS] |
| **Vulnerability Score** | [SCORE] |

[NOTES_OR_RESULTS]

**Red Team Insight:** [INSIGHT_DESCRIPTION]

**Defender Action:** [ACTION_DESCRIPTION]

---

### 5. KV-Cache Persistence Analysis

**Objective:** Measure how long context persists in the key-value cache and identify optimal injection points.

#### Results Summary

| Metric | Value |
|--------|-------|
| **Vulnerability Score** | **[SCORE]** ([LEVEL]) 🟡 |
| **Average Half-Life** | **[VALUE] turns** (all layers) |
| **Best Injection Layers** | [LAYER_LIST] (early layers) |
| **Recommended Injection Turn** | **[TURN]** |

#### KV-Cache Half-Life by Layer

[DESCRIPTION_OF_HALF_LIFE_PATTERN]

```
Layer 0-[MAX]:  ████████░░░░░░░░░░░░░░░░  [VALUE] turns half-life
             ↑                        ↑
         Injection             50% decay
```

#### Context Persistence Timeline

```
Turn 0:  ████████████████████████████  100% context
Turn 1:  ████████████████████████░░░   [RATE]% (injection point)
Turn 2:  ██████████████████████░░░░░   [RATE]%
Turn 3:  ████████████████████░░░░░░░   [RATE]%
Turn 4:  ███████████████████░░░░░░░░   [RATE]% (half-life)
Turn 7:  ████████████░░░░░░░░░░░░░░░   [RATE]%
```

**Red Team Insight:** 
- [INSIGHT_1]
- [INSIGHT_2]
- [INSIGHT_3]

**Attack Strategy:** [STRATEGY_DESCRIPTION]

**Defender Action:** 
- [ACTION_1]
- [ACTION_2]
- [ACTION_3]

---

## 🎯 Key Findings & Insights

### Critical Vulnerabilities Identified

1. **[RISK_LEVEL]: [VULNERABILITY_NAME] (Score: [SCORE])**
   - [DESCRIPTION_1]
   - [DESCRIPTION_2]
   - [DESCRIPTION_3]

2. **[RISK_LEVEL]: [VULNERABILITY_NAME] (Score: [SCORE])**
   - [DESCRIPTION_1]
   - [DESCRIPTION_2]
   - [DESCRIPTION_3]

3. **[RISK_LEVEL]: [VULNERABILITY_NAME] (Score: [SCORE])**
   - [DESCRIPTION_1]
   - [DESCRIPTION_2]
   - [DESCRIPTION_3]

### Model Strengths

1. **✅ [STRENGTH_NAME]**
   - [DESCRIPTION_1]
   - [DESCRIPTION_2]
   - [DESCRIPTION_3]

2. **✅ [STRENGTH_NAME]**
   - [DESCRIPTION_1]
   - [DESCRIPTION_2]

### Attack Vector Prioritization

| Priority | Attack Vector | Exploitability | Impact | Effort |
|----------|--------------|----------------|--------|--------|
| **P0** | [VECTOR] | [LEVEL] | [LEVEL] | [LEVEL] |
| **P1** | [VECTOR] | [LEVEL] | [LEVEL] | [LEVEL] |
| **P2** | [VECTOR] | [LEVEL] | [LEVEL] | [LEVEL] |
| **P3** | [VECTOR] | [LEVEL] | [LEVEL] | [LEVEL] |

---

## 🛡️ Red Team Recommendations

### Immediate Actions

1. **[ACTION_TITLE]**
   - [DESCRIPTION_1]
   - [DESCRIPTION_2]
   - [DESCRIPTION_3]

2. **[ACTION_TITLE]**
   - [DESCRIPTION_1]
   - [DESCRIPTION_2]
   - [DESCRIPTION_3]

3. **[ACTION_TITLE]**
   - [DESCRIPTION_1]
   - [DESCRIPTION_2]
   - [DESCRIPTION_3]

### Advanced Attack Strategies

1. **[STRATEGY_NAME]**
   ```
   Step 1: [STEP]
   Step 2: [STEP]
   Step 3: [STEP]
   Step 4: [STEP]
   ```

2. **[STRATEGY_NAME]**
   ```
   Turn 0: [DESCRIPTION]
   Turn 1: [DESCRIPTION]
   Turn 2-4: [DESCRIPTION]
   Turn 5: [DESCRIPTION]
   ```

3. **[STRATEGY_NAME]**
   - [DESCRIPTION_1]
   - [DESCRIPTION_2]
   - [DESCRIPTION_3]

---

## 🔒 Defender Recommendations

### Immediate Mitigations

1. **[MITIGATION_TITLE]**
   - [ACTION_1]
   - [ACTION_2]
   - [ACTION_3]

2. **[MITIGATION_TITLE]**
   - [ACTION_1]
   - [ACTION_2]
   - [ACTION_3]

3. **[MITIGATION_TITLE]**
   - [ACTION_1]
   - [ACTION_2]
   - [ACTION_3]

### Long-Term Hardening

1. **[HARDENING_TITLE]**
   - [ACTION_1]
   - [ACTION_2]
   - [ACTION_3]

2. **[HARDENING_TITLE]**
   - [ACTION_1]
   - [ACTION_2]
   - [ACTION_3]

3. **[HARDENING_TITLE]**
   - [ACTION_1]
   - [ACTION_2]
   - [ACTION_3]

---

## 📊 Visual Scorecard

### Overall Vulnerability Breakdown

```
Decode Fragility:     ████████████████░░░░░░░░  [SCORE]%  🟡
Logit Lens:           ████████████████████████  [SCORE]%  🔴
Multi-turn Drift:     ░░░░░░░░░░░░░░░░░░░░░░░░  [SCORE]%  🟢
Attention Routing:    ░░░░░░░░░░░░░░░░░░░░░░░░  [SCORE]%  🟢
KV-Cache:             █████████████░░░░░░░░░░░  [SCORE]%  🟡
────────────────────────────────────────────────────
Overall Score:        ██████████░░░░░░░░░░░░░  [SCORE]%  🟡
```

### Risk Matrix

```
        High Impact
            │
            │  [VECTOR]
            │     [SCORE]%
            │
            │
Medium Impact│  [VECTOR]  [VECTOR]
            │    [SCORE]%      [SCORE]%
            │
            │
    Low Impact│  [VECTOR] [VECTOR]
            │      [SCORE]%     [SCORE]%
            │
            └───────────────────────────
            Low        Medium      High
                  Exploitability
```

---

## 📝 Conclusion

The `[MODEL_ID]` model demonstrates **[VULNERABILITY_LEVEL] vulnerability** ([SCORE]% overall score) with **[KEY_WEAKNESSES]**.

**Primary Attack Surface:** [ATTACK_SURFACE_DESCRIPTION]

**Model Strengths:** [STRENGTHS_DESCRIPTION]

**Next Steps:** 
- [NEXT_STEP_1]
- [NEXT_STEP_2]
- [NEXT_STEP_3]

---

## 📚 Appendix

### Model Specifications

- **Architecture:** [ARCHITECTURE_DESCRIPTION]
- **Layers:** [NUM_LAYERS] transformer decoder blocks
- **Attention Heads:** [NUM_HEADS] per layer
- **Hidden Size:** [HIDDEN_SIZE]
- **Vocabulary:** [VOCAB_SIZE] tokens
- **Capabilities:** [CAPABILITIES_LIST]

### Framework Version

- **Behavioral Red Teaming Framework v2**
- **Analysis Date:** [DATE]
- **Experiments:** Decode Fragility, Logit Lens, Multi-turn Drift, Attention Routing, KV-Cache

### Data Files

All raw results are available in:
- `fragility_report.json` - Decode parameter sweep results
- `logit_lens_report.json` - Layer-by-layer decision analysis
- `drift_report.json` - Multi-turn conversation results
- `kv_cache_report.json` - Context persistence analysis
- `attention_report.json` - Attention routing analysis
- `combined_report.json` - Aggregated scores
- `scorecard.txt` - Quick reference scorecard

---

**Report Generated:** [DATE]  
**Framework:** Behavioral Red Teaming Framework v2  
**Status:** ✅ Complete

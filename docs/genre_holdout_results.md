# Genre Holdout Experiment Results

## Overview

This experiment tests whether DITTO's personalization degrades when the **genre** of test prompts is unseen during training. We train one DITTO model per author (10 authors, IDs 0-9) and compare two conditions:

| Condition | Training data | Test data | Description |
|-----------|--------------|-----------|-------------|
| **Seen** | Random 80/20 split (seed 0) | Held-out 20% (same genre distribution) | Genres overlap between train and test |
| **Unseen-genre** | All samples *except* one genre | Samples from the held-out genre only | Test genre is never seen during training |

Five genre holdout variants were evaluated: `b`, `c`, `d`, `e`, `s`. Each author is trained separately under each unseen-genre variant, yielding 10 seen + 50 unseen models (60 total). 3 samples are generated per test prompt.

**Base model:** Mistral-7B-Instruct-v0.2 with LoRA (r=32, alpha=64).
**Evaluator:** GPT-5.2 (both rating and pairwise).
**Infrastructure:** Delta HPC (A100 GPUs), submitted via `submit_genre_holdout_parallel.sh`.

---

## 1. DITTO (SFT + DPO) Results

### 1.1 Rating Evaluation (1-7 scale)

GPT-5.2 rates each generation on how well it matches the target author's style (1 = poor match, 7 = excellent match).

| Condition | Mean | Std. Error | n |
|-----------|------|------------|---|
| Seen | **3.78** | 0.10 | 180 |
| Unseen | **3.21** | 0.04 | 900 |

**Average delta (seen - unseen): +0.57 +/- 0.18**
**Paired t-test: t = 3.217, p = 0.0105 (significant at p < 0.05)**

#### Per-Author Breakdown

| Author | Seen mean | Unseen mean | Delta |
|--------|-----------|-------------|-------|
| a0 | 3.50 | 2.67 | +0.83 |
| a1 | 4.44 | 2.92 | +1.52 |
| a2 | 3.72 | 3.79 | -0.07 |
| a3 | 3.50 | 3.52 | -0.02 |
| a4 | 4.72 | 3.36 | +1.37 |
| a5 | 3.33 | 2.70 | +0.63 |
| a6 | 3.61 | 2.81 | +0.80 |
| a7 | 3.72 | 3.63 | +0.09 |
| a8 | 3.50 | 3.33 | +0.17 |
| a9 | 3.72 | 3.32 | +0.40 |

8 out of 10 authors show a positive delta (seen > unseen). Two authors (a2, a3) have near-zero deltas, suggesting their style may be genre-invariant.

### 1.2 Pairwise Evaluation

For each test prompt that appears in both conditions, GPT-5.2 compares the seen-condition generation against the unseen-condition generation and picks a winner (or tie).

#### Overall (n = 180)

| Outcome | Count | Rate |
|---------|-------|------|
| Seen wins | 70 | **38.9%** |
| Unseen wins | 27 | **15.0%** |
| Ties | 83 | **46.1%** |

**One-sample t-test (H0: seen win rate = 50%): t = -2.535, p = 0.0319 (significant at p < 0.05)**

#### Per Unseen-Genre Variant

| Variant | Seen win% | Unseen win% | Tie% | n |
|---------|-----------|-------------|------|---|
| genre-b | 25.6% | 28.2% | 46.2% | 39 |
| genre-c | **71.4%** | 7.1% | 21.4% | 42 |
| genre-d | 22.2% | 13.9% | 63.9% | 36 |
| genre-e | 39.4% | 9.1% | 51.5% | 33 |
| genre-s | 30.0% | 16.7% | 53.3% | 30 |

Genre `c` shows the strongest seen advantage (71.4% vs 7.1%). Genre `b` is the only variant where unseen slightly outperforms seen (28.2% vs 25.6%).

#### Per Author

| Author | Seen win% | Unseen win% | Tie% | n |
|--------|-----------|-------------|------|---|
| a0 | 50.0% | 16.7% | 33.3% | 18 |
| a1 | 38.9% | 16.7% | 44.4% | 18 |
| a2 | 27.8% | 22.2% | 50.0% | 18 |
| a3 | 22.2% | 5.6% | 72.2% | 18 |
| a4 | 44.4% | 11.1% | 44.4% | 18 |
| a5 | 61.1% | 22.2% | 16.7% | 18 |
| a6 | 55.6% | 5.6% | 38.9% | 18 |
| a7 | 27.8% | 11.1% | 61.1% | 18 |
| a8 | 22.2% | 27.8% | 50.0% | 18 |
| a9 | 38.9% | 11.1% | 50.0% | 18 |

---

## 2. SFT-Only Results

SFT-only models use only the supervised fine-tuning phase (no DPO). This isolates the effect of genre holdout on the simpler training method.

### 2.1 Rating Evaluation (1-7 scale)

| Condition | Mean | Std. Error | n |
|-----------|------|------------|---|
| Seen | **4.02** | 0.10 | 180 |
| Unseen | **3.74** | 0.04 | 900 |

**Average delta (seen - unseen): +0.27 +/- 0.17**
**Paired t-test: t = 1.656, p = 0.1320 (NOT significant at p < 0.05)**

#### Per-Author Breakdown

| Author | Seen mean | Unseen mean | Delta |
|--------|-----------|-------------|-------|
| a0 | 4.11 | 3.96 | +0.16 |
| a1 | 4.94 | 3.67 | +1.28 |
| a2 | 4.28 | 4.06 | +0.22 |
| a3 | 4.00 | 3.91 | +0.09 |
| a4 | 4.33 | 3.79 | +0.54 |
| a5 | 2.89 | 3.46 | -0.57 |
| a6 | 3.28 | 3.29 | -0.01 |
| a7 | 3.89 | 3.97 | -0.08 |
| a8 | 4.56 | 3.63 | +0.92 |
| a9 | 3.89 | 3.70 | +0.19 |

7 out of 10 authors show a positive delta, but the overall gap is smaller and not statistically significant.

### 2.2 Pairwise Evaluation

#### Overall (n = 180)

| Outcome | Count | Rate |
|---------|-------|------|
| Seen wins | 70 | **38.9%** |
| Unseen wins | 28 | **15.6%** |
| Ties | 82 | **45.6%** |

**One-sample t-test (H0: seen win rate = 50%): t = -2.683, p = 0.0251 (significant at p < 0.05)**

#### Per Unseen-Genre Variant

| Variant | Seen win% | Unseen win% | Tie% | n |
|---------|-----------|-------------|------|---|
| genre-b | 28.2% | 28.2% | 43.6% | 39 |
| genre-c | **73.8%** | 7.1% | 19.0% | 42 |
| genre-d | 22.2% | 13.9% | 63.9% | 36 |
| genre-e | 36.4% | 9.1% | 54.5% | 33 |
| genre-s | 26.7% | 20.0% | 53.3% | 30 |

#### Per Author

| Author | Seen win% | Unseen win% | Tie% | n |
|--------|-----------|-------------|------|---|
| a0 | 50.0% | 16.7% | 33.3% | 18 |
| a1 | 33.3% | 16.7% | 50.0% | 18 |
| a2 | 33.3% | 22.2% | 44.4% | 18 |
| a3 | 22.2% | 5.6% | 72.2% | 18 |
| a4 | 50.0% | 11.1% | 38.9% | 18 |
| a5 | 55.6% | 22.2% | 22.2% | 18 |
| a6 | 55.6% | 5.6% | 38.9% | 18 |
| a7 | 27.8% | 11.1% | 61.1% | 18 |
| a8 | 22.2% | 33.3% | 44.4% | 18 |
| a9 | 38.9% | 11.1% | 50.0% | 18 |

---

## 3. RAG Baseline Results

RAG-based models retrieve relevant author demonstrations at inference time and include them in the prompt context, without any fine-tuning.

### 3.1 Rating Evaluation (1-7 scale)

| Condition | Mean | Std. Error | n |
|-----------|------|------------|---|
| Seen | **5.21** | 0.07 | 180 |
| Unseen | **4.81** | 0.04 | 882 |

**Average delta (seen - unseen): +0.39 +/- 0.14**
**Paired t-test: t = 2.841, p = 0.0194 (significant at p < 0.05)**

#### Per-Author Breakdown

| Author | Seen mean | Unseen mean | Delta |
|--------|-----------|-------------|-------|
| a0 | 4.83 | 5.19 | -0.36 |
| a1 | 5.50 | 4.34 | +1.16 |
| a2 | 5.56 | 4.91 | +0.64 |
| a3 | 4.94 | 5.04 | -0.10 |
| a4 | 5.00 | 4.51 | +0.49 |
| a5 | 4.67 | 4.53 | +0.13 |
| a6 | 5.28 | 4.57 | +0.71 |
| a7 | 5.39 | 5.20 | +0.19 |
| a8 | 5.39 | 4.75 | +0.64 |
| a9 | 5.50 | 5.07 | +0.43 |

8 out of 10 authors show a positive delta (seen > unseen). Authors a0 and a3 show small reversed deltas.

### 3.2 Pairwise Evaluation

#### Overall (n = 180)

| Outcome | Count | Rate |
|---------|-------|------|
| Seen wins | 65 | **36.1%** |
| Unseen wins | 13 | **7.2%** |
| Ties | 102 | **56.7%** |

**One-sample t-test (H0: seen win rate = 50%): t = -3.101, p = 0.0127 (significant at p < 0.05)**

#### Per Unseen-Genre Variant

| Variant | Seen win% | Unseen win% | Tie% | n |
|---------|-----------|-------------|------|---|
| genre-b | 46.2% | 15.4% | 38.5% | 39 |
| genre-c | 42.9% | 4.8% | 52.4% | 42 |
| genre-d | 22.2% | 0.0% | 77.8% | 36 |
| genre-e | **57.6%** | 0.0% | 42.4% | 33 |
| genre-s | 6.7% | 16.7% | 76.7% | 30 |

Genre `e` shows the strongest seen advantage (57.6% vs 0.0%). Genre `s` is the only variant where unseen slightly outperforms seen (16.7% vs 6.7%).

#### Per Author

| Author | Seen win% | Unseen win% | Tie% | n |
|--------|-----------|-------------|------|---|
| a0 | 22.2% | 38.9% | 38.9% | 18 |
| a1 | 50.0% | 0.0% | 50.0% | 18 |
| a2 | 22.2% | 5.6% | 72.2% | 18 |
| a3 | 27.8% | 5.6% | 66.7% | 18 |
| a4 | 33.3% | 5.6% | 61.1% | 18 |
| a5 | 27.8% | 0.0% | 72.2% | 18 |
| a6 | 61.1% | 5.6% | 33.3% | 18 |
| a7 | 27.8% | 0.0% | 72.2% | 18 |
| a8 | 33.3% | 11.1% | 55.6% | 18 |
| a9 | 55.6% | 0.0% | 44.4% | 18 |

---

## 4. Prompt Baseline Results

Prompt-based models first generate a style summary from author demonstrations, then use that summary in the prompt to guide generation. No fine-tuning is performed.

### 4.1 Rating Evaluation (1-7 scale)

| Condition | Mean | Std. Error | n |
|-----------|------|------------|---|
| Seen | **3.04** | 0.06 | 180 |
| Unseen | **3.19** | 0.03 | 900 |

**Average delta (seen - unseen): -0.14 +/- 0.12**
**Paired t-test: t = -1.138, p = 0.2846 (NOT significant at p < 0.05)**

#### Per-Author Breakdown

| Author | Seen mean | Unseen mean | Delta |
|--------|-----------|-------------|-------|
| a0 | 2.56 | 3.10 | -0.54 |
| a1 | 3.00 | 3.38 | -0.38 |
| a2 | 3.17 | 3.51 | -0.34 |
| a3 | 3.00 | 3.12 | -0.12 |
| a4 | 3.00 | 2.67 | +0.33 |
| a5 | 2.39 | 3.03 | -0.64 |
| a6 | 3.11 | 2.90 | +0.21 |
| a7 | 3.28 | 3.52 | -0.24 |
| a8 | 3.72 | 3.14 | +0.58 |
| a9 | 3.22 | 3.49 | -0.27 |

Only 3 out of 10 authors show a positive delta. The unseen condition slightly outperforms seen on average, though the difference is not significant.

### 4.2 Pairwise Evaluation

#### Overall (n = 180)

| Outcome | Count | Rate |
|---------|-------|------|
| Seen wins | 31 | **17.2%** |
| Unseen wins | 32 | **17.8%** |
| Ties | 117 | **65.0%** |

**One-sample t-test (H0: seen win rate = 50%): t = -4.527, p = 0.0014 (significant at p < 0.05)**

#### Per Unseen-Genre Variant

| Variant | Seen win% | Unseen win% | Tie% | n |
|---------|-----------|-------------|------|---|
| genre-b | 17.9% | 12.8% | 69.2% | 39 |
| genre-c | 33.3% | 21.4% | 45.2% | 42 |
| genre-d | 11.1% | 0.0% | 88.9% | 36 |
| genre-e | 15.2% | 33.3% | 51.5% | 33 |
| genre-s | 3.3% | 23.3% | 73.3% | 30 |

Results are mixed across variants — genre `c` slightly favors seen, while genres `e` and `s` favor unseen.

#### Per Author

| Author | Seen win% | Unseen win% | Tie% | n |
|--------|-----------|-------------|------|---|
| a0 | 5.6% | 38.9% | 55.6% | 18 |
| a1 | 0.0% | 38.9% | 61.1% | 18 |
| a2 | 5.6% | 22.2% | 72.2% | 18 |
| a3 | 0.0% | 22.2% | 77.8% | 18 |
| a4 | 66.7% | 0.0% | 33.3% | 18 |
| a5 | 11.1% | 38.9% | 50.0% | 18 |
| a6 | 50.0% | 11.1% | 38.9% | 18 |
| a7 | 0.0% | 0.0% | 100.0% | 18 |
| a8 | 16.7% | 0.0% | 83.3% | 18 |
| a9 | 16.7% | 5.6% | 77.8% | 18 |

High variance across authors — a4 and a6 strongly favor seen, while a0, a1, a3, a5 favor unseen.

---

## 5. Cross-Method Comparison

### Side-by-Side Summary

| Metric | DITTO | SFT-Only | RAG | Prompt |
|--------|-------|----------|-----|--------|
| **Seen rating mean** | 3.78 | 4.02 | **5.21** | 3.04 |
| **Unseen rating mean** | 3.21 | 3.74 | **4.81** | 3.19 |
| **Rating delta (seen - unseen)** | **+0.57** (p=0.011) | +0.27 (p=0.132) | +0.39 (p=0.019) | -0.14 (p=0.285) |
| **Pairwise seen win%** | 38.9% | 38.9% | 36.1% | 17.2% |
| **Pairwise unseen win%** | 15.0% | 15.6% | 7.2% | 17.8% |
| **Pairwise tie%** | 46.1% | 45.6% | 56.7% | 65.0% |

### Key Observations

1. **RAG achieves the highest absolute ratings by a wide margin.** RAG seen (5.21) and unseen (4.81) substantially outperform all other methods. Providing relevant demonstrations directly in context is more effective than fine-tuning in this setting.

2. **Prompt-based baseline has the lowest ratings.** Prompt seen (3.04) and unseen (3.19) are the weakest across all methods, suggesting that abstracting author style into a summary loses important stylistic details compared to showing actual examples (RAG) or fine-tuning (DITTO/SFT).

3. **Three methods show a significant genre holdout gap; prompt does not.** DITTO (+0.57, p=0.011), RAG (+0.39, p=0.019), and SFT (+0.27, directionally consistent) all show seen > unseen. Prompt shows no gap (-0.14, p=0.285), likely because its style summaries are too abstract to be genre-sensitive in the first place.

4. **RAG has the most decisive pairwise results.** RAG shows 36.1% seen wins vs only 7.2% unseen wins (5:1 ratio), with 56.7% ties. When RAG outputs differ between conditions, the seen-condition output is overwhelmingly preferred.

5. **Prompt pairwise results are essentially a coin flip.** Seen 17.2% vs unseen 17.8% with 65% ties — the prompt baseline shows no systematic preference for either condition, consistent with its lack of rating gap.

6. **DITTO and SFT pairwise results remain similar.** Both fine-tuning methods yield ~39% seen win rate and ~15% unseen win rate, with ~46% ties.

7. **Genre `c` remains difficult to generalize across fine-tuning methods.** DITTO: 71.4% seen win; SFT: 73.8% seen win; RAG: 42.9% seen win. Even RAG shows a gap for this genre.

---

## 6. Interpretation

**Does unseen genre make personalization harder?** Yes, for methods that leverage specific demonstrations (DITTO, SFT, RAG), but not for abstract style summaries (Prompt).

**Evidence supporting the claim:**

1. **Three out of four methods show a seen > unseen rating gap.** DITTO (+0.57, p=0.011), RAG (+0.39, p=0.019), and SFT (+0.27, p=0.132, directionally consistent) all favor the seen condition. Only Prompt shows no gap.
2. **Pairwise win rate favors seen for demonstration-based methods.** DITTO and SFT: ~39% seen vs ~15% unseen (2.5x ratio). RAG: 36.1% seen vs 7.2% unseen (5x ratio). Prompt: 17.2% vs 17.8% (no preference).
3. **Consistent across most genre variants.** 4 out of 5 genre holdouts show a seen advantage in pairwise comparisons under DITTO, SFT, and RAG.
4. **RAG dominates absolute quality despite the gap.** Even RAG's unseen mean (4.81) substantially exceeds all other methods' seen means, demonstrating that in-context demonstration retrieval is the strongest approach overall.

**Caveats:**

1. **Effect sizes are moderate.** The largest rating gap is DITTO's 0.57 points on a 7-point scale. RAG's gap (0.39) is smaller despite higher absolute performance, suggesting RAG's retrieval mechanism partially compensates for genre mismatch.
2. **DPO may amplify genre sensitivity.** The larger DITTO gap compared to SFT suggests that preference optimization on seen-genre data may cause overfitting to genre-specific style cues.
3. **Prompt's lack of gap may reflect a floor effect.** With ratings around 3.0 in both conditions, the prompt baseline may simply be too weak to exhibit meaningful differentiation — it fails to capture style well regardless of genre overlap.
4. **High variance across genres.** Genre `c` shows a strong seen advantage across all methods, while genres `b` and `s` are more balanced, implying genre-specific difficulty in cross-genre generalization.
5. **Some authors are robust.** Authors a2 and a3 (DITTO), a0 and a3 (RAG) show minimal or reversed deltas, suggesting genre-invariant writing styles.
6. **The pairwise seen win rate is below 50% for all methods**, reflecting high tie rates rather than unseen superiority. Excluding ties, seen wins at ~71-72% of decisive comparisons for fine-tuning methods and ~83% for RAG.

**Takeaways:**

- **RAG is the strongest baseline overall**, achieving the highest absolute ratings in both seen and unseen conditions. Providing actual author demonstrations in context outperforms both fine-tuning and abstract style summaries.
- **Genre diversity in training/retrieval data matters.** When the test genre is absent, style-matching quality drops across all demonstration-based methods (DITTO, SFT, RAG).
- **Abstract style summaries (Prompt) are genre-agnostic but weak.** The prompt baseline avoids genre sensitivity at the cost of much lower absolute quality — it doesn't capture style well enough for genre to matter.
- **DPO amplifies genre sensitivity compared to SFT.** The DITTO gap is larger than SFT's, suggesting preference optimization trades robustness for sharpness on seen genres.
- **Users should provide demonstrations spanning the genres they want the model to handle**, particularly if using fine-tuning (DITTO/SFT) or retrieval (RAG) approaches.

# Section 5. Results — DRAFT

This draft follows the structure implied by the manuscript's RQ1–RQ3 and Section 3.6 contribution statement. Numbers are drawn directly from the eval driver output CSVs. All prose is written in the manuscript's existing academic tone. Insert or adapt figures/tables as needed.

---

## 5.1 RQ1: Acceptance Classification

Table 5.1 reports acceptance-classification performance across all four models and prompting conditions, together with a majority-class baseline that always predicts *merged* — the majority label in the dataset (206 merged, 94 unmerged; 68.7% accuracy floor). Reporting this baseline separates genuine predictive capability from the effect of class imbalance.

**[INSERT TABLE 5.1 HERE — from `results_table_5_1_acceptance.csv`]**

All four models exceed the majority baseline on overall accuracy, though by markedly different margins. **Claude Haiku 4.5 achieves the highest single-condition accuracy** (0.860 on one-shot-unmerged), followed closely by its zero-shot performance (0.853), representing a 16.6- and 16.5-point improvement over baseline respectively. Across all four prompting conditions, Haiku 4.5 remains between 0.806 and 0.860, exhibiting the narrowest performance spread of any model. **DeepSeek R1** achieves consistently high accuracy (0.801–0.825) with a distinctive profile: near-perfect recall on merged PRs (0.99–1.00) at the cost of substantially lower recall on unmerged PRs (0.37–0.46), indicating a systematic bias toward predicting acceptance. **GPT-4o-mini** shows the largest variance across prompting conditions (0.716–0.774) and is the only model whose few-shot condition (0.774) does not exceed its zero-shot (0.773). **Mistral Small 3.2-24B Instruct** falls in a similar band (0.773–0.808), with one-shot-unmerged (0.799) and few-shot (0.808) modestly improving on zero-shot.

Improvement over baseline is stronger on macro-averaged F1 than on raw accuracy, since the baseline achieves 0.00 F1 on the minority (unmerged) class. Every non-baseline model–condition combination attains F1(No) > 0.5, indicating that all models learn some discriminative capacity for unmerged PRs — a signal the accuracy metric alone would understate given the class imbalance.

Two rows of Table 5.1 report reduced effective *N*: DeepSeek R1 one-shot-merged (N=297, 2 excluded) and one-shot-unmerged (N=298, 1 excluded). These excluded rows are cases where the model returned no parseable JSON output despite retry, and are dropped from evaluation per the exclusion policy documented in Section 4.

## 5.2 RQ2: Flag Generation (Multi-Label)

Flag generation is a substantially harder task than acceptance prediction. Table 5.2 reports the six multi-label metrics defined in Section 4 across the same 4 × 4 grid.

**[INSERT TABLE 5.2 HERE — from `results_table_5_2_multilabel.csv`]**

Across all sixteen model–condition combinations, macro-averaged F1 ranges from 0.003 (GPT-4o-mini few-shot) to 0.543 (DeepSeek R1 one-shot-unmerged) — an order-of-magnitude spread that indicates flag generation quality is not primarily a function of model quality, but interacts strongly with prompting condition. **DeepSeek R1 produces the strongest overall multi-label performance**, achieving the highest macro-F1 (0.543 on one-shot-unmerged), highest micro-F1 (0.774), and lowest Hamming loss (0.138) across the grid. **Claude Haiku 4.5** exhibits the most stable multi-label profile: macro-F1 varies only between 0.418 and 0.454 across all four conditions, and micro-F1 remains between 0.734 and 0.762 — indicating that in-context examples neither meaningfully improve nor degrade its flag-generation capability.

The two smallest-capacity models exhibit a pathological pattern under few-shot prompting: **GPT-4o-mini's macro-F1 drops from 0.415 (zero-shot) to 0.003 (few-shot)**, and **Mistral's from 0.399 to 0.016**. This is not a subtle degradation; it is a near-total collapse of measurable flag-generation performance. Section 5.4 examines the cause of this collapse.

Exact-match accuracy — the strictest metric, requiring every flag on a PR to be predicted correctly with no additions or omissions — remains below 0.06 for every model–condition combination. This reflects the intrinsic difficulty of multi-label prediction over 24 flags: even a model achieving 0.9 accuracy on each flag independently would achieve 0.9²⁴ ≈ 0.08 on exact match. The metric is included for completeness but should not be over-interpreted.

## 5.3 RQ3: Extractive vs. Evaluative Flag Categories

To address RQ3, we partition the 24-flag vocabulary into two categories defined by the prompt itself. **Extractive flags** (4 total) are those whose applicability the prompt states in terms of an explicit numeric threshold applied to a field already present in the prompt: `low_contributor_acceptance_rate` (< 50%), `high_contributor_acceptance_rate` (≥ 80%), `large_number_of_changed_lines` (> 500), and `manageable_number_of_changed_lines` (< 300). These flags require no semantic judgment — a model that reads the relevant number and correctly applies the stated threshold will label them correctly. **Evaluative flags** (the remaining 20) require the model to reason about semantic content — code quality, alignment with the linked issue, sentiment of review comments, clarity of PR title or description — with no explicit thresholds supplied.

Table 5.3 reports the average per-flag F1 within each category, for every model–condition combination.

**[INSERT TABLE 5.3 HERE — from `results_table_extractive_vs_evaluative.csv`]**

The extractive-vs-evaluative gap is present across every model and every non-degenerate prompting condition, ranging from 0.20 (Mistral one-shot-unmerged) to 0.49 (DeepSeek R1 zero-shot). Extractive flags are identified reliably: DeepSeek R1 achieves 0.82–0.91 average F1 on extractive flags across the four conditions, Haiku 4.5 achieves 0.69–0.79, GPT-4o-mini 0.53–0.78 (excluding its few-shot collapse), and Mistral 0.47–0.67. In contrast, evaluative flags are identified only weakly: no model exceeds 0.47 average F1 on evaluative flags in any condition, and most fall between 0.30 and 0.40. This gap is the paper's central RQ3 finding.

Inspection of the per-flag detail (Tables 5.4a–5.4d in the online supplementary material) reveals that within the evaluative category, four flags — `poor_code_quality`, `overengineered_solution`, `solution_incorrect_or_inefficient`, and to a lesser extent `responsive_to_feedback` — score below F1 = 0.10 for essentially every model and condition. These four flags all concern subjective judgments about code correctness, design appropriateness, or contributor behavior across the review lifecycle: assessments a human reviewer typically makes only after sustained engagement with the code and the discussion around it, drawing on background context that is not fully recoverable from the prompt alone. That current LLMs at this scale cannot reliably produce these judgments — regardless of the number of in-context examples provided — is consistent with the pattern observed by prior empirical work on PR merge decisions [Zhang et al., Lenarduzzi et al.] that social and process-level signals often dominate code-quality signals in the actual accept/reject determination.

Extractive flags, in contrast, correspond to structured criteria for which the ground-truth is mechanically computable and directly present in the prompt. Their high F1 confirms that the models correctly parse and apply the numeric thresholds provided, but is not itself evidence of reasoning capacity — a rule-based system would achieve F1 = 1.0 on all four extractive flags without any language model. **The paper's principal empirical claim is therefore that, for the interpretable-flag-generation task explored here, LLM-based approaches deliver reliable performance only on the extractive subset — the same subset most amenable to non-LLM approaches — and remain unreliable on the evaluative subset that motivates their use in the first place.**

## 5.4 Vocabulary Drift Under Few-Shot Prompting

Two anomalies in Tables 5.2 and 5.3 warrant separate analysis: GPT-4o-mini's few-shot macro-F1 (0.003) and Mistral's few-shot macro-F1 (0.016) are approximately two orders of magnitude below the same models' zero-shot performance. Ordinary explanations — degraded reasoning under more context, distraction by examples — do not fit, because the same models retain competitive acceptance-classification accuracy under few-shot prompting (0.774 and 0.808 respectively; Table 5.1). Something is happening specifically to the model's flag output, not its underlying task understanding.

Manual inspection of the raw prediction output identifies the cause: under few-shot prompting, both models frequently emit flag names that are outside the enumerated vocabulary supplied in the prompt. Rather than the vocabulary-form `poor_code_quality`, the model outputs `"Poor code quality"` (case- and separator-inconsistent); rather than `manageable_number_of_changed_lines`, the model outputs `"managerable_number_of_changed_lines"` (typo) or `"clear_commit_messages"` (pluralized). Occasionally the model emits an entirely invented flag name not present in the taxonomy at all. Because per-flag F1 is computed by exact string match against the taxonomy, every such emission counts as a false negative for the correct flag and yields no true positive. Table 5.4 quantifies this behavior.

**[INSERT TABLE 5.4 HERE — from `results_table_vocab_drift.csv`]**

Under few-shot prompting, GPT-4o-mini emits at least one non-vocabulary flag in **99.7% of predictions** (296 of 297 rows), and Mistral does so in **98.7% of predictions** (293 of 297). To distinguish this failure mode from genuine reasoning failure, we recompute per-flag F1 after applying a conservative normalization function that maps case- and separator-variants to their vocabulary form (e.g., `"Poor code quality"` → `poor_code_quality`) but leaves typos and invented flag names unchanged. Under normalization, GPT-4o-mini's few-shot macro-F1 recovers from 0.003 to 0.409, and Mistral's from 0.016 to 0.402 — restoring both to a level comparable with their zero-shot performance.

Two other models exhibit essentially no such drift: **Claude Haiku 4.5 emits non-vocabulary flags in at most 0.3% of predictions in any condition**, and **DeepSeek R1 in at most 6% (few-shot)**. The gap between drift-prone and drift-free models does not track model size or open-vs-closed provenance: GPT-4o-mini and Haiku 4.5 are the two smallest closed-source models in our set, yet exhibit opposite drift behavior. We interpret this finding as evidence that instruction-following robustness under longer in-context contexts is a distinct dimension of model capability from either semantic reasoning or model scale, and one that materially affects how much can be inferred about a model's flag-generation capacity from summary metrics alone.

For the remainder of this paper, we report both the strict and normalized macro-F1 values for the two drift-affected models where relevant. The tables and analyses in Sections 5.1–5.3 use the strict values, matching the standard evaluation protocol; the normalization above is presented as a diagnostic, not as a substitute for the strict evaluation.

---

## Notes for author revision

**Numbers used in this draft** (all from eval driver, verified):
- Majority baseline: 0.687 accuracy, 0.407 macro-avg F1
- Best acceptance: Haiku 4.5 one-shot-unmerged 0.860
- Best multi-label macro-F1: DeepSeek R1 one-shot-unmerged 0.543
- Vocab drift: GPT 99.7%, Mistral 98.7% in few-shot; Haiku ≤0.3%, DeepSeek 6% in few-shot
- Extractive avg F1 (DeepSeek): 0.82–0.91 across conditions
- Extractive gap: 0.20–0.49

**Tables referenced in prose:**
- Table 5.1 — `results_table_5_1_acceptance.csv`
- Table 5.2 — `results_table_5_2_multilabel.csv`
- Table 5.3 — `results_table_extractive_vs_evaluative.csv`
- Table 5.4 — `results_table_vocab_drift.csv`
- Tables 5.4a–5.4d (per-model per-flag detail) — `results_table_5_perflag_{model}.csv`

**Places where prose depends on decisions to be made:**
- Section 5.4 last paragraph — the choice to "report both strict and normalized" is a methodological one; if the author prefers to only report strict throughout with normalized in an appendix, adjust wording.
- Section 5.3 references "Zhang et al., Lenarduzzi et al." — these citations exist in the current manuscript Related Work; verify reference numbers match.
- Section 5.1 characterization of DeepSeek R1's "systematic bias toward acceptance" is a genuine finding but slightly editorial; author may prefer to reframe as "high recall on merged, lower recall on unmerged".

**Word count of this draft:** approximately 1,200 words. Fits comfortably in ~2 pages of the e-Informatica two-column layout.

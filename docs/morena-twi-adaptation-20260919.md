# MORENA Twi substrate audit

This experiment tests whether `vamboai/morena-1.5b-base` is a sensible substrate for the next Twi-understanding adaptation. It does **not** authorize training or model promotion.

## Gate 0: tokenizer audit

Run:

```bash
python scripts/audit_morena_twi.py --download
```

The runner resolves the current MORENA Hub commit SHA before downloading its tokenizer, records that immutable revision and tokenizer hash, and compares the native MORENA tokenizer against the already-pinned Qwen 3.5 and Gemma 4 tokenizers on the existing structurally eligible, non-protected Twi source pool.

Primary measurements are tokens/word, tokens/byte and bytes/token, with per-source distributions. Keep speech transcripts, parallel text and Pristine-derived text visible separately. Tokenizer efficiency is only a representation test, not proof that MORENA understands Twi.

A smoke run may use `--limit 5000`; the decision report must use the complete eligible source pool.

## Gate 1: frozen zero-shot model evaluation

Only after Gate 0 is recorded, evaluate the untouched MORENA Base checkpoint on the existing locked Twi understanding/meaning fixtures and a causal-language-model loss slice. Do not train on, relabel, or regenerate protected evaluation rows.

Compare at minimum:
- untouched MORENA 1.5B Base;
- the existing selected general-model baseline where reproducible;
- later, the same MORENA checkpoint plus Twi adaptation.

Record raw generations and model revision. Translation/QA scores and causal loss must not be collapsed into one synthetic score.

## Gate 2: bounded language adaptation

If Gate 0 does not reveal pathological Twi fragmentation and Gate 1 leaves a meaningful language gap, start with a bounded causal-LM language-adaptation run from MORENA Base. Do not start from MORENA Instruct.

First ablation:
1. transformer LoRA, embeddings frozen;
2. transformer LoRA plus embedding training only if the first result suggests lexical representation is limiting.

Start with broad transformer targets: q/k/v/o projections plus gate/up/down projections. Treat rank/alpha as experiment parameters, not conclusions.

The first run is general Twi language adaptation. Health response SFT remains a later stage. Preserve an English/African replay slice for a larger run and measure forgetting.

## Promotion criteria

A candidate advances only if it improves held-out Twi language/meaning measurements without unacceptable English retention regression or new repetition/echo failure. The prior Gemma language-adaptation run showed that falling translation loss can coexist with worse conversational behavior, so loss alone is not a promotion signal.

No model upload, production deployment, or health-response claim follows automatically from this experiment.

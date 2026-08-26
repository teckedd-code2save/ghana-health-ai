# Understanding benchmark

`seed.v0.jsonl` is the initial synthetic probe set for comparing existing Twi
translation and understanding models. It contains no production conversations,
participant recordings, or personal data.

The rows are **not gold labels**. A native-speaker review must add the intended
English meaning, normalized Twi, ambiguities, intent, and entities before the
set can support model-selection claims. Until then it is suitable only for
collecting side-by-side predictions and finding obvious failures.

Keep immutable model revisions and raw predictions with every run. Do not train
on a promoted evaluation split.

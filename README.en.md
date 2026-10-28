# PlainVoice

[中文](README.md)

Chinese writing diagnostics and bounded, context-aware generation control. PlainVoice provides an offline analyzer, a local writing workbench, an OpenAI-compatible rewrite adapter, and a Hugging Face logits processor for phrase-level preferences.

“Human words” and “AI words” are not universal categories. A word may span several tokens, and token IDs depend on the tokenizer. PlainVoice therefore treats rules as editable writing preferences, rather than an authorship classifier. It does not report AI-writing probabilities or promise detector evasion.

## Quick start

Python 3.11+; use [uv](https://docs.astral.sh/uv/) and the checked-in `uv.lock` for pinned development dependencies. Run from the repository root:

```bash
uv sync --frozen --extra dev --extra tokens
source .venv/bin/activate
plainvoice analyze examples/input.txt --profile examples/profile.json --json
plainvoice serve --host 127.0.0.1 --port 8765
pytest
ruff check .
```

The sample profile produces two reviewable findings in the sample text. Built-in preferences may produce a different count; zero findings do not prove writing quality. Without uv, create and activate a virtual environment and run `python -m pip install -e '.[dev,tokens]'`; this resolves version ranges rather than reproducing the lock file.

Offline analysis works without a model or credentials. To enable remote rewriting, supply `PLAINVOICE_MODEL`, `PLAINVOICE_BASE_URL`, and securely inject `PLAINVOICE_API_KEY`, then run:

```bash
plainvoice rewrite examples/input.txt --backend openai
```

Hosted rewriting uses prompt-based style instructions by default. Static API `logit_bias` is model-dependent and cannot faithfully represent a multi-token, history-dependent phrase rule. The local logits processor can adjust a phrase-completing token after its prefix appears; it does not globally suppress shared Chinese characters.

For actual local generation, install the larger optional dependencies and provide an already cached causal language model or model directory. This path does not automatically download model weights:

```bash
uv sync --frozen --extra dev --extra tokens --extra local
plainvoice rewrite examples/input.txt --backend local --model /path/to/cached-model
```

The initial adapter runs on CPU, requires a model chat template, and caps new output at 1,024 tokens. Large models may be slow or exceed available memory. Processor tests verify steering logic; a real local-model generation run has not been validated in environments without model weights. No subjective quality gain is implied.

```bash
plainvoice compile --profile examples/profile.json --tokenizer cl100k_base --out phrase-plan.json
plainvoice fit --human data/reference --model-corpus data/generated \
  --tokenizer cl100k_base --out token-bias.json
```

Corpus weights use smoothed token-frequency ratios, minimum support, tokenizer metadata, and bounds. Positive boosts require explicit opt-in. A learned table describes its training corpus, not a general human vocabulary. Never reuse token IDs with a different tokenizer.

Some tiktoken encodings require a first-use vocabulary download from `openaipublic.blob.core.windows.net`. Restricted cloud environments must allow that hostname and apply the configuration before running `compile` or `fit`. Saving a draft alone does not update the current runtime. Offline analysis does not require this download.

Apply a learned table with `--bias-table`. Hosted static bias additionally requires `--tokenizer-id` matching the table and `--allow-static-bias`; these explicitly confirm provider support rather than discovering it automatically. Positive weights additionally require `--allow-positive`.

The `compile` and `fit` commands create new output files and refuse to overwrite existing results. The workbench only permits a loopback host.

Rewrites are blocked when deterministic preservation checks fail. Those checks cover selected facts and protected text; passing them does not prove semantic equivalence. Review important output before publication.

## Evidence and licensing

Unit and integration tests verify engineering behavior. They do not establish subjective writing quality gains; no blind human-preference result is claimed. See the Chinese [design](docs/design.md) and [evaluation plan](docs/evaluation.md).

Inspired by the workflow in [lieflat-less-ai-tone](https://github.com/larashero3-dotcom/lieflat-less-ai-tone), which is licensed CC BY-NC 4.0. This repository does not copy its scripts, research corpus, rule tables, or report templates, and does not present its research numbers as our results. PlainVoice's independent code is [MIT licensed](LICENSE). Reuse of upstream materials remains governed by their own license.

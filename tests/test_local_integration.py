"""Real CPU inference mechanics with a tiny, locally created random model.

No pretrained weights, network, or quality claims. A deterministic candidate mask
isolates the effect of PlainVoice's processor inside transformers.generate().
"""

import pytest

from plainvoice import providers
from plainvoice.decoding import compile_plan
from plainvoice.local import PhraseLogitsProcessor
from plainvoice.profiles import PhraseRule, StyleProfile

torch = pytest.importorskip("torch", reason="Install the local extra for CPU integration")
transformers = pytest.importorskip("transformers", reason="Install the local extra")
tokenizers = pytest.importorskip("tokenizers", reason="Install the local extra")


@pytest.fixture
def tiny_local_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    vocabulary = {
        "[PAD]": 0, "[UNK]": 1, "[EOS]": 2,
        "human": 3, "tone": 4, "robot": 5, "word": 6,
        "user": 7, "assistant": 8, "system": 9,
    }
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} "
        "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    )
    config = transformers.GPT2Config(
        vocab_size=len(vocabulary), n_layer=1, n_head=1, n_embd=8,
        n_positions=2048, bos_token_id=7, eos_token_id=2, pad_token_id=0,
    )
    network = transformers.GPT2LMHeadModel(config).eval()
    tokenizer.save_pretrained(tmp_path)
    network.save_pretrained(tmp_path)
    providers._load_local_model.cache_clear()
    yield str(tmp_path), tokenizer, network
    providers._load_local_model.cache_clear()
    torch.set_num_threads(original_threads)


class CandidateMask:
    """Fixture-only candidate set; not a substitute for the tested processor."""

    def __init__(self, prompt_length, candidates):
        self.prompt_length = prompt_length
        self.candidates = candidates

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.prompt_length
        allowed = self.candidates[min(step, len(self.candidates) - 1)]
        masked = torch.full_like(scores, float("-inf"))
        masked[:, allowed] = 0.0
        return masked


def test_actual_generate_uses_contextual_phrase_bias_to_change_selected_token(tiny_local_model):
    model_path, tokenizer, network = tiny_local_model
    prompt = tokenizer("user", return_tensors="pt")
    prompt_length = prompt["input_ids"].shape[1]
    profile = StyleProfile("test", "test", (PhraseRule("human tone", -1.5),))
    plan = compile_plan(profile, tokenizer, model_path)
    processor = PhraseLogitsProcessor(plan, prompt_length, model_path, network.config.vocab_size)
    candidate_mask = CandidateMask(prompt_length, [[3], [4, 5]])
    with torch.inference_mode():
        baseline = network.generate(
            **prompt, max_new_tokens=2, do_sample=False,
            logits_processor=transformers.LogitsProcessorList([candidate_mask]),
        )
        steered = network.generate(
            **prompt, max_new_tokens=2, do_sample=False,
            logits_processor=transformers.LogitsProcessorList([candidate_mask, processor]),
        )
    assert baseline[0, prompt_length:].tolist() == [3, 4]
    assert steered[0, prompt_length:].tolist() == [3, 5]
    assert tokenizer.decode(steered[0, prompt_length:]) == "human robot"


def test_local_adapter_loads_only_local_files_and_passes_processor_to_real_generate(
    tiny_local_model, monkeypatch,
):
    model_path, _, _ = tiny_local_model
    tokenizer, network = providers._load_local_model(model_path)
    original_generate = network.generate
    observed = {"processor": False, "called": False}

    def generate_with_fixture_candidates(**kwargs):
        observed["called"] = True
        processors = list(kwargs["logits_processor"])
        observed["processor"] = any(isinstance(p, PhraseLogitsProcessor) for p in processors)
        prompt_length = kwargs["input_ids"].shape[1]
        # Emit one fixture word followed by EOS through actual model.generate().
        kwargs["logits_processor"] = transformers.LogitsProcessorList([
            CandidateMask(prompt_length, [[6], [tokenizer.eos_token_id]]), *processors,
        ])
        return original_generate(**kwargs)

    monkeypatch.setattr(network, "generate", generate_with_fixture_candidates)
    profile = StyleProfile("test", "test", (PhraseRule("human tone", -1.5),))
    output = providers._local_output(
        [{"role": "user", "content": "word"}], model=model_path,
        profile=profile, token_bias=None, allow_positive=False,
    )
    assert output == "word"
    assert observed == {"processor": True, "called": True}
    assert not network.training

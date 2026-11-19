import pytest

from llm_opt_lab.model_parts import is_non_language


@pytest.mark.parametrize(
    "name",
    [
        "model.vision_tower.encoder.layers.0.self_attn.q_proj",  # Gemma 4
        "model.audio_tower.layers.0.ffw.linear",  # Gemma 4
        "model.embed_vision.embedding_projection",  # Gemma 4
        "model.visual.blocks.0.attn.qkv",  # Qwen3.5 / Qwen3.6
        "model.visual.merger.linear_fc1",  # Qwen3.5 / Qwen3.6
    ],
)
def test_non_language_modules(name):
    assert is_non_language(name)


@pytest.mark.parametrize(
    "name",
    [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.linear_attn.in_proj_qkv",  # Qwen Gated DeltaNet
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "lm_head",
    ],
)
def test_language_modules(name):
    assert not is_non_language(name)

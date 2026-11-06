"""DeepSeek-V4.1-Flash parameter tree + loader (CPU only, no vLLM, no NPU).

Pins the S1 contract on a *tiny* config so it runs in CI:

* every checkpoint-relevant name exists exactly once, at the shape/dtype the shards use;
* the architectural roles are right (compressor only on kv sources, indexer only on index sources,
  ``wk``/``k_norm`` only where the layer owns its index keys, DSpark extras only on stage 0 / last);
* the loader's two transformations are correct: packed fp4 experts are a ``view``, and ``wo_a`` is
  folded from fp8 + e8m0 block scales to bf16 with the right block indexing.
"""

from __future__ import annotations

import torch

from tileinfer.models.dsv41 import (
    Dsv41Config,
    Dsv41Model,
    fold_wo_a,
    load_shards,
    verify_headers,
)


def tiny_config(**overrides) -> Dsv41Config:
    base = dict(
        vocab_size=128,
        hidden_size=64,
        moe_intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        sliding_window=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        compress_ratios=(0, 2, 1, 0, 0),  # 3 layers + 2 DSpark stages
        kv_source_layers=(1, 2),
        index_source_layers=(1, 2),
        candidate_source_layer=2,
        candidate_block_size=2,
        candidate_topk_blocks=4,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=4,
        hc_mult=2,
        engram_layer_ids=(0,),
        engram_num_embeddings=(64,),
        engram_max_ngram_size=3,
        engram_n_heads=2,
        engram_head_dim=32,
        num_nextn_predict_layers=2,
        dspark_target_layer_ids=(1, 2, 2),
        dspark_markov_rank=8,
        dspark_n_routed_experts=2,
        dspark_n_activated_experts=1,
        vision_num_hidden_layers=1,
        vision_hidden_size=16,
        vision_num_attention_heads=2,
        vision_intermediate_size=16,
        vision_patch_size=2,
        vision_downsample_ratio=2,
    )
    base.update(overrides)
    return Dsv41Config(**base)


def params(model: Dsv41Model) -> dict:
    return dict(model.named_parameters())


TABLE = {
    # name -> (shape, dtype)
    "embed.weight": ((128, 64), torch.bfloat16),
    "head.weight": ((128, 64), torch.bfloat16),
    "norm.weight": ((64,), torch.bfloat16),
    "layers.0.attn.attn_sink": ((2,), torch.float32),
    "layers.0.attn.wq_a.weight": ((32, 64), torch.float8_e4m3fn),
    "layers.0.attn.wq_a.scale": ((1, 2), torch.float8_e8m0fnu),
    "layers.0.attn.wq_b.weight": ((64, 32), torch.float8_e4m3fn),
    "layers.0.attn.wkv.weight": ((32, 64), torch.float8_e4m3fn),
    "layers.0.attn.wo_a.weight": ((32, 32), torch.bfloat16),
    "layers.0.attn.wo_b.weight": ((64, 32), torch.float8_e4m3fn),
    "layers.0.ffn.experts.0.w1.weight": ((64, 32), torch.float4_e2m1fn_x2),
    "layers.0.ffn.experts.0.w1.scale": ((64, 2), torch.float8_e8m0fnu),
    "layers.0.ffn.experts.0.w2.weight": ((64, 32), torch.float4_e2m1fn_x2),
    "layers.0.ffn.experts.0.w2.scale": ((64, 2), torch.float8_e8m0fnu),
    "layers.0.ffn.shared_experts.w1.weight": ((64, 64), torch.float8_e4m3fn),
    "layers.0.ffn.gate.weight": ((4, 64), torch.bfloat16),
    "layers.0.ffn.gate.bias": ((4,), torch.float32),
    "layers.0.ffn.gate.bias_vl": ((4,), torch.float32),
    "layers.0.hc_attn_fn": (((2 + 2) * 2, 2 * 64), torch.float32),
    "layers.0.hc_attn_scale": ((3,), torch.float32),
    "layers.0.engram.embed.weight": ((64, 32), torch.float8_e4m3fn),
    "layers.0.engram.embed.scale": ((64, 1), torch.float8_e8m0fnu),
    "layers.1.attn.compressor.wkv.weight": ((32, 64), torch.bfloat16),
    "layers.1.attn.compressor.wgate.weight": ((32, 64), torch.bfloat16),
    "layers.1.attn.indexer.wk.weight": ((16, 32), torch.bfloat16),
    "layers.1.attn.indexer.k_norm.weight": ((16,), torch.bfloat16),
    "layers.2.attn.compressor.norm.weight": ((32,), torch.bfloat16),
    "mtp.0.main_proj.weight": ((64, 192), torch.float8_e4m3fn),
    "mtp.0.main_proj.scale": ((2, 6), torch.float8_e8m0fnu),
    "mtp.0.attn.wq_a.weight": ((32, 64), torch.float8_e4m3fn),
    "vision.patch_embed.proj.weight": ((16, 12), torch.bfloat16),
    "vision.patch_embed.proj.bias": ((16,), torch.bfloat16),
    "vision.blocks.0.attn.wqkv.weight": ((48, 16), torch.bfloat16),
    "vision.blocks.0.mlp.w1.weight": ((32, 16), torch.bfloat16),
    "aligner.w1.weight": ((64, 64), torch.bfloat16),
    "image_start": ((64,), torch.bfloat16),
}


def test_parameter_table_matches_checkpoint_conventions():
    model = Dsv41Model(tiny_config())
    table = params(model)
    for name, (shape, dtype) in TABLE.items():
        assert name in table, f"{name} missing from the parameter tree"
        assert tuple(table[name].shape) == shape, f"{name}: {tuple(table[name].shape)} != {shape}"
        assert table[name].dtype == dtype, f"{name}: {table[name].dtype} != {dtype}"


def test_architectural_roles():
    model = Dsv41Model(tiny_config())
    table = params(model)
    # ratio-1 layer has a compressor but no gate; ratio-2 layers have both
    assert "layers.1.attn.compressor.wgate.weight" in table
    assert "layers.2.attn.compressor.wgate.weight" not in table
    assert "layers.2.attn.compressor.wkv.weight" in table
    # layer 0 is neither a kv nor an index source
    assert "layers.0.attn.compressor.wkv.weight" not in table
    assert "layers.0.attn.indexer.wq_b.weight" not in table
    # only kv sources own index keys
    assert "layers.1.attn.indexer.wk.weight" in table
    # engram only on its own layer
    assert "layers.0.engram.wkv.weight" in table
    assert "layers.1.engram.wkv.weight" not in table
    # DSpark: main_proj/norm only on stage 0, vocab heads only on the last stage
    assert "mtp.0.main_proj.weight" in table
    assert "mtp.0.norm.weight" not in table
    assert "mtp.0.markov_head.head.weight" not in table
    # the shared embed/head are not duplicated under mtp.*
    assert not any(name.startswith("mtp.0.embed") for name in table)
    # 4 experts x 6 tensors + gate 3 + shared 6 per layer
    layer0_ffn = [n for n in table if n.startswith("layers.0.ffn.")]
    assert len(layer0_ffn) == 4 * 6 + 3 + 6


def test_dspark_last_stage_heads():
    model = Dsv41Model(tiny_config(num_nextn_predict_layers=2))
    table = params(model)
    assert "mtp.1.norm.weight" in table
    assert "mtp.1.markov_head.embed.weight" in table
    assert "mtp.1.confidence_head.proj.weight" in table
    assert "mtp.0.confidence_head.proj.weight" not in table
    assert tuple(table["mtp.1.confidence_head.proj.weight"].shape) == (1, 64 + 8)


def test_optional_sections_can_be_dropped():
    model = Dsv41Model(tiny_config(), with_vision=False, with_mtp=False)
    names = params(model)
    assert not any(n.startswith("vision.") for n in names)
    assert not any(n.startswith("mtp.") for n in names)
    assert "layers.0.attn.wq_a.weight" in names


def test_config_validate_flags_a_stream_without_owner():
    cfg = tiny_config(compress_ratios=(0, 2, 2), kv_source_layers=(2,))
    problems = " ".join(cfg.validate())
    assert "no kv source precedes" in problems

    cfg = tiny_config(compress_ratios=(1, 2, 2), kv_source_layers=(0, 2))
    problems = " ".join(cfg.validate())
    assert "must keep one ratio" in problems


def test_config_from_hf_nested_and_flat():
    nested = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "quantization_config": {"expert_dtype": "fp4"},
        "text_config": {
            "hidden_size": 64,
            "num_hidden_layers": 3,
            "qk_rope_head_dim": 8,
            "compress_ratios": [0, 2, 1],
        },
        "vision_config": {"num_hidden_layers": 1, "hidden_size": 16},
    }
    cfg = Dsv41Config.from_dict(nested)
    assert cfg.hidden_size == 64 and cfg.rope_head_dim == 8 and cfg.n_layers == 3
    assert cfg.expert_dtype == "fp4" and cfg.vision_num_hidden_layers == 1

    flat = {"dim": 64, "n_layers": 3, "rope_head_dim": 8, "compress_ratios": [0, 2, 1]}
    cfg2 = Dsv41Config.from_dict(flat)
    assert cfg2.hidden_size == 64 and cfg2.n_layers == 3


def test_fold_wo_a_block_indexing():
    """Blocks must follow the checkpoint order [out_block, in_block], not a tile transposition."""
    out_block, in_block, blocks_o, blocks_i = 32, 32, 2, 2
    weight = torch.ones(blocks_o * out_block, blocks_i * in_block, dtype=torch.float8_e4m3fn)
    # block (bi, bj) gets scale 2**(bi*blocks_i + bj): distinct per block, so a transposed fold fails
    bits = torch.arange(127, 127 + blocks_o * blocks_i, dtype=torch.int16).reshape(blocks_o, blocks_i)
    scale = bits.to(torch.uint8).view(torch.float8_e8m0fnu)
    folded = fold_wo_a(weight, scale)
    assert folded.dtype == torch.bfloat16 and tuple(folded.shape) == tuple(weight.shape)
    for bi in range(blocks_o):
        for bj in range(blocks_i):
            expected = float(2 ** (bi * blocks_i + bj))
            got = folded[bi * out_block : (bi + 1) * out_block, bj * in_block : (bj + 1) * in_block]
            assert torch.allclose(got.float(), torch.full_like(got.float(), expected)), (bi, bj)


def _synth(param, value: float = 1.0):
    """A shard-side tensor with the checkpoint's dtype/shape for ``param`` (fp4 arrives as int8)."""
    if param.dtype == torch.float4_e2m1fn_x2:
        return torch.zeros(param.shape, dtype=torch.int8)
    if param.dtype == torch.float8_e8m0fnu:
        return torch.full(param.shape, 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    return torch.full(param.shape, float(value)).to(param.dtype)


def _synth_shard(table: dict, wo_a_scale: float = 1.0) -> dict:
    shard = {}
    for name, param in table.items():
        if name.endswith(".wo_a.weight"):
            shard[name] = _synth(param).to(torch.float8_e4m3fn)
            blocks = (param.shape[0] // 32, param.shape[1] // 32)
            shard[name.replace(".weight", ".scale")] = torch.full(
                blocks, 127, dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
            continue
        shard[name] = _synth(param)
    return shard


def test_loader_roundtrip(tmp_path):
    """A synthetic shard must load with zero mismatches and land the right values."""
    from safetensors.torch import save_file

    cfg = tiny_config()
    model = Dsv41Model(cfg)
    table = params(model)
    shard = _synth_shard(table)
    path = tmp_path / "model0-mp1.safetensors"
    save_file(shard, str(path))

    report = verify_headers(model, [str(path)])
    assert report.ok, report.summary()

    report = load_shards(model, [str(path)])
    assert report.ok, report.summary()
    assert report.copied == len(table)
    assert report.folded_wo_a == sum(1 for n in table if n.endswith(".wo_a.weight"))
    # values landed (the fold produced 1.0 * 2**0)
    assert torch.allclose(
        params(model)["embed.weight"].float(), torch.ones(128, 64, dtype=torch.float32)
    )
    assert torch.allclose(
        params(model)["layers.0.attn.wo_a.weight"].float(), torch.ones(32, 32, dtype=torch.float32)
    )


def test_verify_headers_catches_a_reshaped_tensor(tmp_path):
    from safetensors.torch import save_file

    cfg = tiny_config()
    model = Dsv41Model(cfg, with_vision=False, with_mtp=False)
    table = params(model)
    shard = _synth_shard(table)
    shard["embed.weight"] = torch.ones(64, 128, dtype=torch.bfloat16)  # transposed on purpose
    path = tmp_path / "bad.safetensors"
    save_file(shard, str(path))

    report = verify_headers(model, [str(path)])
    assert not report.ok
    assert any(item[0] == "embed.weight" for item in report.shape_mismatch)

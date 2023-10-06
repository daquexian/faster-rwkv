import os
import numpy as np
import pytest
from pathlib import Path

import fasterrwkv as fr

MODEL_TOP_DIR = os.getenv("FR_MODEL_DIR") # pyright: ignore [reportGeneralTypeIssues]
if MODEL_TOP_DIR is not None:
    MODEL_TOP_DIR = Path(MODEL_TOP_DIR)

MODEL_TOP_DIR: Path

@pytest.mark.skipif(MODEL_TOP_DIR is None, reason="FR_MODEL_DIR not set")
def test_model():
    model_dir = MODEL_TOP_DIR/"RWKV-5-World-0.1B-v1-20230803-ctx4096"/"ncnn"/"fp16"
    model = fr.Model(model_dir/"RWKV-5-World-0.1B-v1-20230803-ctx4096-ncnn", "ncnn fp16")
    output = model([0])
    assert output.shape == (65536,)
    assert -7.1 < output[0] < -7
    assert -15.9 < output[9] < -15.75
    old_output = output
    output = model([0])
    assert output.shape == (65536,)
    assert -7.5 < output[0] < -7.3
    assert -15.0 < output[9] < -14.8

    model.reset_states()
    new_output = model([0])
    assert new_output.shape == (65536,)
    assert np.array_equal(old_output, new_output)


@pytest.mark.skipif(MODEL_TOP_DIR is None, reason="FR_MODEL_DIR not set")
def test_tokenizer():
    model_dir = MODEL_TOP_DIR/"RWKV-5-World-0.1B-v1-20230803-ctx4096"/"ncnn"/"fp16"
    tokenizer = fr.Tokenizer(model_dir)
    input = "Hello world!"
    ids = tokenizer.encode(input)
    decoded = tokenizer.decode(ids)
    assert decoded == input


def test_sampler():
    sampler = fr.Sampler()
    logits = np.array([5, 3, 1]).astype(np.float32)
    output_id = sampler.sample(logits, top_p=0)
    assert output_id == 0
    output_id = sampler(logits, top_p=0)
    assert output_id == 0

    sampler.set_seed(8)
    logits = np.array([3, -5, 0, 4, -1]).astype(np.float32)
    counts = np.zeros(len(logits), dtype=np.float32)
    N = 50
    for _ in range(N):
        output_id = sampler.sample(logits, top_k=0, top_p=1, temperature=2)
        counts[output_id] += 1
    dist = counts / N
    assert np.isclose(dist[0], 0.2199999988079071)
    assert np.isclose(dist[1], 0)
    assert np.isclose(dist[2], 0.06)
    assert np.isclose(dist[3], 0.69999998807907104)
    assert np.isclose(dist[4], 0.02)


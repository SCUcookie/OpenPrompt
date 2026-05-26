from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from openprompt_rs.models import PromptBank
from openprompt_rs.utils.embeddings import build_text_embedder


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_prompt_bank_builds_from_taxonomy() -> None:
    prompt_bank = PromptBank.build_from_files(
        taxonomy_path=REPO_ROOT / "assets/hierarchies/remote_sensing_taxonomy.json",
        template_path=REPO_ROOT / "assets/prompts/prompt_templates.json",
        embedding_dim=64,
    )
    embeddings = prompt_bank()
    assert embeddings.shape[0] == len(prompt_bank.class_names)
    assert embeddings.shape[1] == 64
    assert "ship" in prompt_bank.prompt_strings
    ship_prompts = prompt_bank.prompt_strings["ship"]
    assert any("vessel" in prompt for prompt in ship_prompts)
    assert any("distinguished from" in prompt for prompt in ship_prompts)
    assert any("unlikely when cues indicate" in prompt for prompt in ship_prompts)


def test_prompt_bank_reuses_embedding_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "prompt_cache.pt"
    first = PromptBank.build_from_files(
        taxonomy_path=REPO_ROOT / "assets/hierarchies/remote_sensing_taxonomy.json",
        template_path=REPO_ROOT / "assets/prompts/prompt_templates.json",
        embedding_dim=32,
        embedding_cache_path=cache_path,
    )
    cached = torch.load(cache_path, map_location="cpu")
    cached["embeddings"] = torch.ones_like(cached["embeddings"])
    torch.save(cached, cache_path)

    second = PromptBank.build_from_files(
        taxonomy_path=REPO_ROOT / "assets/hierarchies/remote_sensing_taxonomy.json",
        template_path=REPO_ROOT / "assets/prompts/prompt_templates.json",
        embedding_dim=32,
        embedding_cache_path=cache_path,
    )

    assert first.base_embeddings.shape == second.base_embeddings.shape
    expected = F.normalize(torch.ones_like(second.base_embeddings), dim=-1)
    assert torch.allclose(second.base_embeddings, expected)


def test_real_vlm_backend_requires_open_clip() -> None:
    if importlib.util.find_spec("open_clip") is not None:
        pytest.skip("missing-dependency path only applies without open_clip")
    with pytest.raises(RuntimeError, match="open_clip is required"):
        build_text_embedder("remoteclip", model_name="ViT-B-32")

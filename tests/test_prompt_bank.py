from __future__ import annotations

from pathlib import Path

from openprompt_rs.models import PromptBank


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


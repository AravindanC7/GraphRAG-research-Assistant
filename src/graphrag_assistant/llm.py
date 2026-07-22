"""Thin embedding-provider wrapper so the model can be swapped via config.

Keeping providers behind a small interface means later phases (generation,
fine-tuned extraction models, local Ollama, etc.) only touch this file.
"""

from openai import OpenAI

from .config import settings


class Embedder:
    # NOTE: embedding model is fixed by the Neo4j vector index dimensions.
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        key = api_key or settings.openai_api_key
        if not key:
            raise RuntimeError("No OpenAI API key provided.")
        self._client = OpenAI(api_key=key)
        self.model = model or settings.embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text (order preserved)."""
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]


class ChatLLM:
    """Chat model wrapper used for entity/relationship extraction.

    `complete_json` asks the model for a JSON object and returns the raw
    string, which the caller parses. Temperature is 0 so extraction is as
    deterministic as the model allows (we want consistent triples, not
    creativity).
    """

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        key = api_key or settings.openai_api_key
        if not key:
            raise RuntimeError("No OpenAI API key provided.")
        self._client = OpenAI(api_key=key)
        self.model = model or settings.llm_model

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or "{}"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Plain-text completion (no JSON mode) — used for answer generation."""
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""

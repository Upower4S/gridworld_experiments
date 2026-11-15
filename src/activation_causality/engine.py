"""Offline GPT-OSS interventions with independent, persistent decoding branches.

MLX is imported only when an engine first performs inference. Positions are
zero-based absolute token positions; layers are zero-based transformer blocks.
``attn_out`` and ``mlp_out`` refer to branch outputs before residual addition;
``resid_post`` refers to the state after both residual additions. Patches apply
once during prefill. Subsequent decoding retains their downstream KV effects.
An optional expected_recipient_float32_sha256 guards the exact ordered
recipient states immediately before a patch writes and records the actual hash.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.conceptual_belief_space.forced_suffix_probe import (
    ACTIONS,
    FORCED_SUFFIX,
    forced_suffix_token_ids,
)

SITES = frozenset({"resid_post", "attn_out", "mlp_out"})
OPERATIONS = frozenset({"replace", "add", "subspace"})
ENGINE_DEPENDENCY_PATHS = (
    "src/__init__.py",
    "src/activation_causality/__init__.py",
    "src/conceptual_belief_space/__init__.py",
    "src/conceptual_belief_space/config.py",
    "src/conceptual_belief_space/forced_suffix_probe.py",
)
_FINAL_HEADER = re.compile(
    r"<\|start\|>assistant\s*<\|channel\|>final\s*<\|message\|>"
)
_MESSAGE_END = re.compile(r"<\|(?:end|return|fim_suffix|call|endoftext|start)\|>")


def engine_dependency_hashes(runtime: Mapping[str, Any]) -> dict[str, str]:
    """Require the explicit local package/load/forced-readout dependency closure.

    This conservative engine/load provenance also binds native action ordering
    and parsing. It does not assert that every suffix edit changes captures.
    The manifest implementation itself is bound by the engine source hash.
    """
    sources = runtime.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("engine dependency closure requires source hashes")
    result = {}
    for relative in ENGINE_DEPENDENCY_PATHS:
        digest = sources.get(relative)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"missing or invalid engine dependency source hash: {relative}")
        result[relative] = digest
    return result


class RecipientStateMismatch(ValueError):
    """A recipient guard rejected this write; JSON diagnostics survive prefill.

    ``patch_audit`` describes the failed patch, not an atomic rollback of
    preceding patches. The runner can persist it even though no PrefillState
    is returned. No write or applied-patch log is produced for this patch.
    """

    def __init__(self, patch_audit: Mapping[str, Any]):
        self.patch_audit = json.loads(json.dumps(dict(patch_audit), allow_nan=False))
        details = self.patch_audit
        super().__init__(
            f"recipient_float32_sha256 mismatch before patch write at layer {details['layer']} "
            f"site {details['site']} positions {details['positions']}: "
            f"expected {details['expected_recipient_float32_sha256']}, "
            f"actual {details['recipient_float32_sha256']}"
        )


def parse_native_action(text: str) -> dict[str, Any]:
    """Parse JSON only inside a naturally emitted assistant final message.

    Reasoning-channel JSON, plain action mentions, and malformed/trailing prose
    cannot become actions. A syntactically complete JSON body can be recognized
    at a token cap; the caller must separately decide whether generation ended.
    """
    headers = list(_FINAL_HEADER.finditer(text))
    if not headers:
        return {"parsed_action": None, "parse_status": "missing_final_channel", "final_text": None}
    if len(headers) != 1:
        return {"parsed_action": None, "parse_status": "multiple_final_messages", "final_text": None}
    body = _MESSAGE_END.split(text[headers[0].end() :], maxsplit=1)[0].strip()
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return {"parsed_action": None, "parse_status": "invalid_final_json", "final_text": body}
    action = payload.get("action") if isinstance(payload, dict) else None
    if not isinstance(action, str) or action.strip().upper() not in ACTIONS:
        return {"parsed_action": None, "parse_status": "invalid_action", "final_text": body}
    return {"parsed_action": action.strip().upper(), "parse_status": "valid", "final_text": body}


def action_metrics(logits: Any, candidate_ids: Sequence[int]) -> dict[str, Any]:
    """Four-candidate conditional belief, plus its mass in the full vocabulary."""
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    selected = values[np.asarray(candidate_ids, dtype=int)]
    if len(selected) != 4 or not np.isfinite(values).all():
        raise ValueError("expected four candidates and finite vocabulary logits")
    exp = np.exp(selected - selected.max())
    q = exp / exp.sum()
    ordered = np.sort(q)
    log_denom = values.max() + np.log(np.exp(values - values.max()).sum())
    log_mass = selected.max() + np.log(exp.sum()) - log_denom
    entropy = float(-(q * np.log(np.maximum(q, 1e-300))).sum())
    return {
        **{f"q_{action}": float(q[i]) for i, action in enumerate(ACTIONS)},
        **{f"logit_{action}": float(selected[i]) for i, action in enumerate(ACTIONS)},
        "top_action": ACTIONS[int(q.argmax())],
        "top_prob": float(q.max()),
        "entropy": entropy,
        "entropy_commitment": 1.0 - entropy / float(np.log(4)),
        "margin_commitment": float(ordered[-1] - ordered[-2]),
        "candidate_mass": float(np.exp(log_mass)),
    }


@dataclass
class PrefillState:
    """Inference state after the last supplied token, before sampling its successor."""

    token_ids: tuple[int, ...]
    cache: list[Any]
    next_logits: Any
    engine_id: int
    patch_log: list[dict[str, Any]] = field(default_factory=list)


class MLXInterventionEngine:
    """One lazily loaded local model. Keep one instance per inference worker.

    Patch dictionaries have ``layer``, ``site``, ``positions``, ``donor``
    ([positions, hidden]), ``operation`` and ``dose``. ``replace`` interpolates
    toward donor states; ``subspace`` interpolates their orthogonal projection;
    ``add`` adds dose times the supplied tensor (an explicitly chosen delta).
    Norm logs describe the actual change after conversion to the model dtype.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path).expanduser().resolve()
        self._model = self._tokenizer = self._mx = None
        self._make_cache = self._mask = self._sampler_factory = None
        self._candidate_ids: list[int] = []
        self._suffix_ids: list[int] = []
        self._terminal_ids: dict[int, str] = {}
        # Native batching is the default across processes, including workers
        # that reuse a persisted validation gate. Full-head prefill fits the
        # local worker's budget and avoids BF16 GEMM/GEMV batching discrepancies.
        self._head_mode = "full"
        self.metadata: dict[str, Any] = {"model_path": str(self.model_path), "loaded": False}

    @property
    def head_mode(self) -> str:
        return self._head_mode

    @classmethod
    def from_components(
        cls, *, model: Any, tokenizer: Any, backend: Any, make_cache: Any,
        create_attention_mask: Any, candidate_ids: Sequence[int],
        suffix_ids: Sequence[int], terminal_ids: Mapping[int, str],
        sampler_factory: Any = None,
    ) -> "MLXInterventionEngine":
        """Inject a small model/backend for correctness tests without importing MLX."""
        engine = cls(".")
        engine._model, engine._tokenizer, engine._mx = model, tokenizer, backend
        engine._make_cache, engine._mask = make_cache, create_attention_mask
        engine._sampler_factory = sampler_factory
        engine._candidate_ids = list(candidate_ids)
        engine._suffix_ids = list(suffix_ids)
        engine._terminal_ids = dict(terminal_ids)
        engine.metadata = {"loaded": True, "injected_components": True}
        engine._check_model()
        return engine

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # A concrete existing directory prevents mlx_lm from treating the input
        # as a Hub ID. Tokenizer local_files_only closes its separate fetch path.
        path = self.model_path
        if not path.is_dir() or not (path / "config.json").is_file():
            raise FileNotFoundError(f"expected an existing offline model snapshot: {path}")
        if not (path / "tokenizer.json").is_file():
            raise FileNotFoundError(f"offline tokenizer is missing from {path}")
        index = path / "model.safetensors.index.json"
        if index.exists():
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if any(not (path / shard).is_file() for shard in shards):
                raise FileNotFoundError("offline snapshot contains missing weight shards")
        elif not any(path.glob("*.safetensors")):
            raise FileNotFoundError("offline snapshot contains no safetensors weights")
        import importlib.metadata
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = load(str(path), tokenizer_config={"local_files_only": True})
        self._mx, self._model, self._tokenizer = mx, model, tokenizer
        self._make_cache, self._mask = make_prompt_cache, create_attention_mask
        self._sampler_factory = make_sampler
        self._suffix_ids = forced_suffix_token_ids(tokenizer)
        self._candidate_ids = []
        for action in ACTIONS:
            ids = self.encode(action)
            if len(ids) != 1:
                raise ValueError(f"action {action} must encode to exactly one token")
            self._candidate_ids.append(ids[0])
        for marker, reason in (("<|return|>", "return"), ("<|endoftext|>", "eos"), ("<|call|>", "tool_call")):
            ids = self.encode(marker)
            if len(ids) != 1:
                raise ValueError(f"terminal marker {marker} is not a single token")
            self._terminal_ids[ids[0]] = reason
        self._check_model()
        self.metadata = {
            "loaded": True, "model_path": str(path), "model_revision": path.name,
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_lm_version": importlib.metadata.version("mlx-lm"),
            "n_layers": len(model.model.layers), "hidden_dim": int(model.args.hidden_size),
            "candidate_ids": dict(zip(ACTIONS, self._candidate_ids)),
            "forced_suffix_ids": self._suffix_ids,
            "head_mode": self._head_mode,
        }

    def _check_model(self) -> None:
        transformer = self._model.model
        required = ("layers", "layer_types", "embed_tokens", "norm", "window_size", "ga_idx", "swa_idx")
        if any(not hasattr(transformer, name) for name in required):
            raise TypeError("manual intervention backend requires the MLX GPT-OSS architecture")
        if len(transformer.layers) != len(transformer.layer_types):
            raise ValueError("model layers and attention types differ in length")
        if not set(transformer.layer_types).issubset({"full_attention", "sliding_attention"}):
            raise TypeError("unsupported attention layer type")
        for layer in transformer.layers:
            if any(not hasattr(layer, name) for name in ("self_attn", "input_layernorm", "post_attention_layernorm", "mlp")):
                raise TypeError("unsupported transformer sub-block implementation")

    def encode(self, text: str) -> list[int]:
        self._ensure_loaded()
        return [int(i) for i in self._tokenizer.encode(text, add_special_tokens=False)]

    def decode(self, ids: Sequence[int]) -> str:
        self._ensure_loaded()
        return self._tokenizer.decode(list(ids), skip_special_tokens=False)

    def _copy_tree(self, value: Any) -> Any:
        if isinstance(value, tuple):
            return tuple(self._copy_tree(v) for v in value)
        if isinstance(value, list):
            return [self._copy_tree(v) for v in value]
        if isinstance(value, dict):
            return {k: self._copy_tree(v) for k, v in value.items()}
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return self._mx.array(value)
        return copy.deepcopy(value)

    def clone_state(self, state: PrefillState) -> PrefillState:
        self._ensure_loaded()
        if state.engine_id != id(self):
            raise ValueError("state belongs to a different engine")
        cache = [type(c).from_state(self._copy_tree(c.state), self._copy_tree(c.meta_state)) for c in state.cache]
        return PrefillState(state.token_ids, cache, self._mx.array(state.next_logits), id(self), copy.deepcopy(state.patch_log))

    @staticmethod
    def _positions(values: Sequence[int], length: int) -> list[int]:
        positions = list(values)
        if not positions or any(not isinstance(p, (int, np.integer)) for p in positions):
            raise ValueError("positions must be nonempty integer token indices")
        positions = [int(p) for p in positions]
        if len(set(positions)) != len(positions) or min(positions) < 0 or max(positions) >= length:
            raise ValueError("positions must be unique and inside the supplied sequence")
        return positions

    def _validate_patches(self, patches: Sequence[Mapping[str, Any]], length: int) -> list[dict[str, Any]]:
        checked, occupied = [], set()
        n_layers = len(self._model.model.layers)
        width = int(self._model.args.hidden_size)
        for raw in patches:
            patch = dict(raw)
            layer = patch.get("layer")
            site, operation = patch.get("site", "resid_post"), patch.get("operation", "replace")
            dose = float(patch.get("dose", 1.0))
            if not isinstance(layer, (int, np.integer)) or not 0 <= int(layer) < n_layers:
                raise ValueError("patch layer is outside the model")
            if site not in SITES or operation not in OPERATIONS or not np.isfinite(dose):
                raise ValueError("invalid patch site, operation, or dose")
            if "expected_recipient_float32_sha256" in patch:
                expected_hash = patch["expected_recipient_float32_sha256"]
                if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
                    raise ValueError("expected_recipient_float32_sha256 must be a 64-character hexadecimal SHA256")
                patch["expected_recipient_float32_sha256"] = expected_hash.lower()
            positions = self._positions(patch.get("positions", []), length)
            keys = {(int(layer), site, p) for p in positions}
            if keys & occupied:
                raise ValueError("overlapping patches at the same layer and site are ambiguous")
            occupied |= keys
            donor = np.asarray(patch["donor"], dtype=np.float32)
            if donor.shape == (1, len(positions), width):
                donor = donor[0]
            if donor.shape != (len(positions), width) or not np.isfinite(donor).all():
                raise ValueError("donor must have finite shape [number of positions, hidden dimension]")
            patch.update(layer=int(layer), site=site, operation=operation, dose=dose, positions=positions, donor=donor)
            if operation == "subspace":
                basis = np.asarray(patch.get("basis"), dtype=np.float32)
                if basis.ndim != 2 or basis.shape[0] != width or not 0 < basis.shape[1] <= width:
                    raise ValueError("basis must have shape [hidden dimension, positive rank]")
                if not np.isfinite(basis).all() or not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-4, rtol=1e-4):
                    raise ValueError("subspace basis columns must be orthonormal")
                patch["basis"] = basis
            checked.append(patch)
        return checked

    def _patch_tensor(self, tensor: Any, patch: Mapping[str, Any], logs: list[dict[str, Any]]) -> Any:
        mx = self._mx
        positions = patch["positions"]
        before = tensor[:, positions, :].astype(mx.float32)
        recipient_hash = None
        if "expected_recipient_float32_sha256" in patch:
            from .subspace_audit import array_fingerprint

            # Evaluate/hash the actual ordered states before this write, not a
            # cached original from before preceding interventions. Explicit
            # float32 conversion also avoids MLX's unsupported BF16 buffer.
            mx.eval(before)
            before_float = np.asarray(before, dtype=np.float32)[0]
            recipient_hash = array_fingerprint(before_float)
            expected_hash = patch["expected_recipient_float32_sha256"]
            if recipient_hash != expected_hash:
                raise RecipientStateMismatch({
                    "status": "recipient_state_mismatch", "write_applied": False,
                    "layer": patch["layer"], "site": patch["site"], "positions": list(positions),
                    "operation": patch["operation"], "dose": float(patch["dose"]),
                    "rank": int(patch["basis"].shape[1]) if patch["operation"] == "subspace" else None,
                    "expected_recipient_float32_sha256": expected_hash,
                    "recipient_float32_sha256": recipient_hash,
                    "before_l2": np.linalg.norm(before_float, axis=-1).tolist(),
                    "recipient_shape": list(before_float.shape), "preceding_patch_count": len(logs),
                })
        donor = mx.array(patch["donor"])[None, :, :]
        operation, dose = patch["operation"], patch["dose"]
        if operation == "add":
            delta = donor
        elif operation == "replace":
            delta = donor - before
        else:
            basis = mx.array(patch["basis"])
            delta = ((donor - before) @ basis) @ basis.T
        updated = (before + dose * delta).astype(tensor.dtype)
        tensor[:, positions, :] = updated
        # NumPy's buffer protocol cannot represent MLX bfloat16. Convert every
        # model-dtype boundary explicitly, including values used only for logs.
        updated_float = updated.astype(mx.float32)
        mx.eval(before, updated_float)
        b = np.asarray(before, dtype=np.float32)[0]
        u = np.asarray(updated_float, dtype=np.float32)[0]
        norm_before, norm_delta = np.linalg.norm(b, axis=-1), np.linalg.norm(u - b, axis=-1)
        logs.append({
            "layer": patch["layer"], "site": patch["site"], "positions": list(positions),
            "operation": operation, "dose": dose,
            "rank": int(patch["basis"].shape[1]) if operation == "subspace" else None,
            "before_l2": norm_before.tolist(), "delta_l2": norm_delta.tolist(),
            "relative_delta_l2": (norm_delta / np.maximum(norm_before, 1e-12)).tolist(),
            **({"recipient_float32_sha256": recipient_hash} if recipient_hash is not None else {}),
        })
        return tensor

    def _manual(
        self, token_ids: Sequence[int], cache: list[Any], *, patches: Sequence[Mapping[str, Any]] = (),
        capture_layers: Sequence[int] = (), capture_positions: Sequence[int] = (), capture_site: str = "resid_post",
    ) -> tuple[Any, dict[int, np.ndarray], list[dict[str, Any]]]:
        mx, transformer = self._mx, self._model.model
        x = transformer.embed_tokens(mx.array([list(token_ids)], dtype=mx.int32))
        full_mask = self._mask(x, cache[transformer.ga_idx])
        sliding_mask = self._mask(x, cache[transformer.swa_idx], window_size=transformer.window_size)
        captures, logs = {}, []
        by_site: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
        for patch in patches:
            by_site.setdefault((patch["layer"], patch["site"]), []).append(patch)

        def at_site(value: Any, layer_id: int, site: str) -> Any:
            for patch in by_site.get((layer_id, site), ()):
                value = self._patch_tensor(value, patch, logs)
            if layer_id in capture_layers and site == capture_site:
                selected = value[0, list(capture_positions), :].astype(mx.float32)
                mx.eval(selected)
                captures[layer_id] = np.asarray(selected, dtype=np.float32).copy()
            return value

        for index, (layer, kind, layer_cache) in enumerate(zip(transformer.layers, transformer.layer_types, cache, strict=True)):
            mask = full_mask if kind == "full_attention" else sliding_mask
            attention = layer.self_attn(layer.input_layernorm(x), mask, layer_cache)
            x = x + at_site(attention, index, "attn_out")
            mlp = layer.mlp(layer.post_attention_layernorm(x))
            x = x + at_site(mlp, index, "mlp_out")
            x = at_site(x, index, "resid_post")
        # Validation may select the full-head fallback if a backend's different
        # GEMM batching changes last-token logits beyond native parity tolerance.
        head_input = x[:, -1:, :] if self._head_mode == "last_token" else x
        logits = self._model.lm_head(transformer.norm(head_input))[0, -1].astype(mx.float32)
        mx.eval(logits)
        return logits, captures, logs

    def capture(self, token_ids: Sequence[int], layers: Sequence[int], positions: Sequence[int], site: str = "resid_post") -> dict[int, np.ndarray]:
        self._ensure_loaded()
        ids = self._validate_ids(token_ids)
        positions = self._positions(positions, len(ids))
        layers = list(layers)
        if site not in SITES or not layers or any(not isinstance(i, (int, np.integer)) or not 0 <= i < len(self._model.model.layers) for i in layers):
            raise ValueError("invalid capture layers or site")
        _, captures, _ = self._manual(ids, self._make_cache(self._model), capture_layers=layers, capture_positions=positions, capture_site=site)
        return captures

    @staticmethod
    def _validate_ids(token_ids: Sequence[int]) -> tuple[int, ...]:
        ids = tuple(token_ids)
        if not ids or any(not isinstance(t, (int, np.integer)) or t < 0 for t in ids):
            raise ValueError("token IDs must be nonempty nonnegative integers")
        return tuple(int(t) for t in ids)

    def prefill(self, token_ids: Sequence[int], patches: Sequence[Mapping[str, Any]] = ()) -> PrefillState:
        self._ensure_loaded()
        ids = self._validate_ids(token_ids)
        checked = self._validate_patches(patches, len(ids))
        cache = self._make_cache(self._model)
        logits, _, logs = self._manual(ids, cache, patches=checked)
        return PrefillState(ids, cache, logits, id(self), logs)

    def _advance(self, state: PrefillState, token_ids: Sequence[int]) -> None:
        ids = self._validate_ids(token_ids)
        state.next_logits, _, _ = self._manual(ids, state.cache)
        state.token_ids += ids

    def forced(self, state: PrefillState) -> dict[str, Any]:
        """Force an answer on an independent cache branch, leaving state intact."""
        branch = self.clone_state(state)
        self._advance(branch, self._suffix_ids)
        return {**action_metrics(branch.next_logits, self._candidate_ids), "endpoint": "forced", "suffix": FORCED_SUFFIX}

    def generate(self, state: PrefillState, seed: int, max_new_tokens: int = 2048, temp: float = 0.7) -> dict[str, Any]:
        """Continue the patched prefix through its native final channel.

        This deliberately does not stop at analysis ``<|end|>`` and never
        appends the forced suffix. ``action`` is null for capped or malformed
        runs, even if a complete JSON object was seen before the token cap.
        """
        if max_new_tokens < 1 or temp < 0 or not np.isfinite(temp):
            raise ValueError("max_new_tokens must be positive and temperature nonnegative")
        branch = self.clone_state(state)
        mx = self._mx
        mx.random.seed(int(seed))
        sampler = self._sampler_factory(temp=temp) if self._sampler_factory is not None else None
        generated: list[int] = []
        reason = "token_limit"
        for _ in range(int(max_new_tokens)):
            logits = branch.next_logits
            if sampler is None:
                if temp != 0:
                    raise ValueError("sampling backend required for nonzero temperature")
                token = int(np.asarray(logits).argmax())
            else:
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                sampled = sampler(logprobs[None, :])
                mx.eval(sampled)
                token = int(np.asarray(sampled).reshape(-1)[0])
            generated.append(token)
            if token in self._terminal_ids:
                reason = self._terminal_ids[token]
                break
            self._advance(branch, [token])
        text = self.decode(generated)
        parsed = parse_native_action(text)
        ended = reason in {"return", "eos"}
        complete = ended and parsed["parse_status"] == "valid"
        generation_status = "complete" if complete else ("truncated" if reason == "token_limit" else "malformed")
        return {
            **parsed, "action": parsed["parsed_action"] if complete else None,
            "native_action": parsed["parsed_action"] if complete else None,
            "generation_status": generation_status, "generated_tokens": len(generated),
            "endpoint": "native_final", "completed": complete,
            "ended_naturally": ended, "truncated": reason == "token_limit",
            "stop_reason": reason, "n_new_tokens": len(generated),
            "generated_token_ids": generated, "generated_text": text,
            "seed": int(seed), "temperature": float(temp), "max_new_tokens": int(max_new_tokens),
        }

    def validate(self, token_ids: Sequence[int] | None = None, tolerance: float = 1e-3) -> dict[str, Any]:
        """Gate short and >window cached execution against independent forwards.

        Native/manual and identity comparisons keep strict same-shape gates.
        Cache-versus-full comparisons also report the no-patch numerical drift:
        patched errors must stay within twice that measured drift plus tolerance,
        for both vocabulary logits and forced q. This is a numerical reference,
        not a relaxed scientific effect-size threshold. The report records all
        raw errors and top-action agreements for independent inspection.
        """
        self._ensure_loaded()
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("validation tolerance must be finite and positive")
        if token_ids is None:
            token_ids = self.encode("<|start|>user<|message|>Choose UP, DOWN, LEFT, or RIGHT.<|end|><|start|>assistant<|channel|>analysis<|message|>I will inspect the grid.")
        short = self._validate_ids(token_ids)
        long_ids = list(short)
        extra = self.encode(" I will compare the open neighboring cells carefully.")
        while len(long_ids) < max(176, int(self._model.model.window_size) + 32):
            long_ids.extend(extra)
        inputs = [("short", short), ("past_sliding_window", tuple(long_ids))]
        mx = self._mx

        def numpy_float(value: Any) -> np.ndarray:
            if hasattr(value, "astype"):
                value = value.astype(mx.float32)
                mx.eval(value)
            return np.asarray(value, dtype=np.float32)

        def logit_difference(left: Any, right: Any) -> dict[str, Any]:
            left, right = numpy_float(left), numpy_float(right)
            return {"max_abs": float(np.max(np.abs(left - right))),
                    "relative_l2": float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1e-12)),
                    "vocabulary_top_agrees": bool(left.argmax() == right.argmax())}

        def belief_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
            return {"max_abs_q": max(abs(left[f"q_{a}"] - right[f"q_{a}"]) for a in ACTIONS),
                    "top_agrees": left["top_action"] == right["top_action"],
                    "reference_margin": right["margin_commitment"],
                    "left_q": [left[f"q_{a}"] for a in ACTIONS],
                    "right_q": [right[f"q_{a}"] for a in ACTIONS]}

        # Record the actual head configuration; production defaults to native
        # full-head batching, even when a worker reuses an existing gate.
        native_checks, head_attempts = {}, []
        for _ in range(2):
            native_checks = {}
            for name, ids in inputs:
                native = self._model(mx.array([list(ids)], dtype=mx.int32), cache=self._make_cache(self._model))[0, -1].astype(mx.float32)
                mx.eval(native)
                manual = self.prefill(ids)
                native_checks[name] = logit_difference(manual.next_logits, native)
            head_attempts.append({"head_mode": self._head_mode, "native_checks": copy.deepcopy(native_checks)})
            if all(c["max_abs"] <= tolerance for c in native_checks.values()) or self._head_mode == "full":
                break
            self._head_mode = "full"
        self.metadata["head_mode"] = self._head_mode

        def leaves(tree: Any) -> list[Any]:
            if isinstance(tree, (tuple, list)):
                return [leaf for item in tree for leaf in leaves(item)]
            return [tree]

        cases = []
        for name, ids in inputs:
            baseline = self.prefill(ids)
            positions, layer = [len(ids) - 1], 0
            next_token = int(numpy_float(baseline.next_logits).argmax())
            base_cached = self.clone_state(baseline)
            self._advance(base_cached, [next_token])
            base_full = self.prefill([*ids, next_token])
            base_logit = logit_difference(base_cached.next_logits, base_full.next_logits)
            base_cached_forced = self.forced(base_cached)
            base_q = belief_difference(base_cached_forced, self.forced(base_full))
            site_reports = {}
            for site in sorted(SITES):
                donor = self.capture(ids, [layer], positions, site)[layer]
                patch = {"layer": layer, "site": site, "positions": positions, "donor": donor, "operation": "replace", "dose": 1.0}
                identity = self.prefill(ids, [patch])
                identity_error = logit_difference(identity.next_logits, baseline.next_logits)
                changed = {**patch, "donor": donor + 0.1}
                patched = self.prefill(ids, [changed])
                cached = self.clone_state(patched)
                self._advance(cached, [next_token])
                uncached = self.prefill([*ids, next_token], [changed])
                patch_logit = logit_difference(cached.next_logits, uncached.next_logits)
                patch_q = belief_difference(self.forced(cached), self.forced(uncached))
                logit_bound, q_bound = 2 * base_logit["max_abs"] + tolerance, 2 * base_q["max_abs_q"] + tolerance
                # A near-tie can switch its argmax under harmless rounding. A
                # decisively separated action must agree between implementations.
                top_gate = patch_q["top_agrees"] or patch_q["reference_margin"] <= 2 * q_bound
                passed = identity_error["max_abs"] == 0 and patch_logit["max_abs"] <= logit_bound and patch_q["max_abs_q"] <= q_bound and top_gate
                site_reports[site] = {"passed": bool(passed), "identity": identity_error,
                                      "patched_cache": patch_logit, "patched_forced": patch_q,
                                      "allowed_logit_error_from_reference": logit_bound,
                                      "allowed_q_error_from_reference": q_bound,
                                      "patch_norms": patched.patch_log}
            screening_report = None
            if name == "past_sliding_window":
                # The actual campaign writes mid-layer multi-token spans. Use
                # a sizeable deterministic perturbation to establish that this
                # check really carries an intervention into subsequent decoding.
                screening_layer = min(10, max(0, len(self._model.model.layers) - 2))
                screening_positions = list(range(len(ids) - min(3, len(ids)), len(ids)))
                donor = self.capture(ids, [screening_layer], screening_positions)[screening_layer]
                pattern = np.where(np.arange(donor.shape[1]) % 2, -1.0, 1.0).astype(np.float32)
                scale = np.sqrt(np.mean(donor * donor, axis=1, keepdims=True))
                delta = 0.5 * np.maximum(scale, 0.1) * pattern[None, :]
                identity_patch = {"layer": screening_layer, "site": "resid_post", "positions": screening_positions,
                                  "donor": donor, "operation": "replace", "dose": 1.0}
                identity = self.prefill(ids, [identity_patch])
                identity_error = logit_difference(identity.next_logits, baseline.next_logits)
                patch = {**identity_patch, "donor": donor + delta}
                patched = self.prefill(ids, [patch])
                cached = self.clone_state(patched)
                self._advance(cached, [next_token])
                full = self.prefill([*ids, next_token], [patch])
                cached_forced = self.forced(cached)
                parity = logit_difference(cached.next_logits, full.next_logits)
                parity_q = belief_difference(cached_forced, self.forced(full))
                effect = logit_difference(cached.next_logits, base_cached.next_logits)
                effect_q = belief_difference(cached_forced, base_cached_forced)
                logit_bound, q_bound = 2 * base_logit["max_abs"] + tolerance, 2 * base_q["max_abs_q"] + tolerance
                nonvacuous = effect["max_abs"] > tolerance
                top_gate = parity_q["top_agrees"] or parity_q["reference_margin"] <= 2 * q_bound
                screening_report = {
                    "layer": screening_layer, "positions": screening_positions, "site": "resid_post",
                    "identity": identity_error, "patched_cache": parity, "patched_forced": parity_q,
                    "patched_vs_baseline_cached_logits": effect, "patched_vs_baseline_forced": effect_q,
                    "nonvacuous_downstream_logit_effect": bool(nonvacuous), "patch_norms": patched.patch_log,
                    "allowed_logit_error_from_reference": logit_bound, "allowed_q_error_from_reference": q_bound,
                    "passed": bool(identity_error["max_abs"] == 0 and parity["max_abs"] <= logit_bound
                                   and parity_q["max_abs_q"] <= q_bound and top_gate and nonvacuous),
                }
            original = numpy_float(baseline.next_logits).copy()
            original_cache = [[numpy_float(a).copy() for a in leaves(c.state)] for c in baseline.cache]
            first, second = self.forced(baseline), self.forced(baseline)
            branch_q_error = belief_difference(first, second)["max_abs_q"]
            cache_unchanged = all(np.array_equal(a, numpy_float(b)) for old, c in zip(original_cache, baseline.cache, strict=True) for a, b in zip(old, leaves(c.state), strict=True))
            state_unchanged = np.array_equal(numpy_float(baseline.next_logits), original)
            case_passed = native_checks[name]["max_abs"] <= tolerance and all(r["passed"] for r in site_reports.values()) and cache_unchanged and state_unchanged and branch_q_error == 0
            if screening_report is not None:
                case_passed = case_passed and screening_report["passed"]
            cases.append({"name": name, "tokens": len(ids), "passed": bool(case_passed),
                          "native_manual": native_checks[name], "baseline_cached_full_logits": base_logit,
                          "baseline_cached_full_forced": base_q, "sites": site_reports,
                          "screening_multitoken": screening_report,
                          "cache_unchanged": cache_unchanged, "state_unchanged": bool(state_unchanged),
                          "branch_repeat_q_error": branch_q_error})
        return {"passed": all(c["passed"] for c in cases), "tolerance": tolerance,
                "cache_unchanged": all(c["cache_unchanged"] for c in cases), "cases": cases,
                "head_attempts": head_attempts, "model": dict(self.metadata),
                "cache_error_rule": "patched error <= 2 * measured no-patch error + tolerance; identity exact; native strict"}

"""
reference: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dllm.utils.generation_utils import get_num_transfer_tokens
from dllm.core.generation.generator import GeneratorOutput, GeneratorConfig, BaseGenerator


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    NOTE: keep the original behavior you had; this returns a tensor suitable for argmax sampling.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def apply_repetition_penalty(
    logits: torch.Tensor,
    sequences: torch.Tensor,
    penalty: float = 1.2,
    mask_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    Apply repetition penalty to logits in place (returns logits).
    Behavior matches HuggingFace's repetition_penalty:
      for each token_id that appears in `sequences`:
        if logits[..., token_id] < 0: logits[..., token_id] *= penalty
        else: logits[..., token_id] /= penalty

    Args:
      logits: [B, T, V] or [B, V]
      sequences: [B, T] (token ids previously present in canvas)
      penalty: >1.0 means penalize repeats
      mask_token_id, eos_token_id: token ids to ignore when building the set of repeated tokens
    """
    if penalty == 1.0:
        return logits

    device = logits.device
    B = sequences.shape[0]

    # Work on a copy or in-place depending on caller preference. We'll modify logits in-place.
    if logits.ndim == 3:
        # [B, T_logits, V]
        # We'll apply the penalty across the vocabulary dimension for each batch and all timesteps.
        T_logits = logits.shape[1]
        V = logits.shape[2]
        for b in range(B):
            seq = sequences[b]
            # get unique token ids present in this sequence
            uniq = torch.unique(seq)
            # filter out mask/eos (-1 also if needed)
            if mask_token_id is not None:
                uniq = uniq[uniq != mask_token_id]
            if eos_token_id is not None:
                uniq = uniq[uniq != eos_token_id]
            if uniq.numel() == 0:
                continue
            uniq = uniq.to(device=device, dtype=torch.long)
            # advanced indexing assignment:
            # logits[b, :, uniq] -> shape [T_logits, n_uniq]
            slice_b = logits[b, :, uniq]  # view
            pos_mask = slice_b > 0
            # apply: >0 => divide, <=0 => multiply
            slice_b = torch.where(pos_mask, slice_b / penalty, slice_b * penalty)
            logits[b, :, uniq] = slice_b
    elif logits.ndim == 2:
        # [B, V] typical last-step-only logits
        V = logits.shape[1]
        for b in range(B):
            seq = sequences[b]
            uniq = torch.unique(seq)
            if mask_token_id is not None:
                uniq = uniq[uniq != mask_token_id]
            if eos_token_id is not None:
                uniq = uniq[uniq != eos_token_id]
            if uniq.numel() == 0:
                continue
            uniq = uniq.to(device=device, dtype=torch.long)
            slice_b = logits[b, uniq]
            pos_mask = slice_b > 0
            slice_b = torch.where(pos_mask, slice_b / penalty, slice_b * penalty)
            logits[b, uniq] = slice_b
    else:
        # unexpected logits dim
        raise ValueError(f"Unexpected logits.ndim={logits.ndim}")

    return logits


@dataclass
class LLaDAGeneratorConfig(GeneratorConfig):
    max_new_tokens: int = 128
    max_length: int = None  # There's no explicit length_limit except for the tokenizer/model context
    block_length: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    repetition_penalty: float = 1.2  # default


@dataclass
class LLaDAGenerator(BaseGenerator):
    @torch.no_grad()
    def generate(
        self,
        inputs: list[torch.Tensor | list],
        config: LLaDAGeneratorConfig | None = None,
        **kwargs
    ) -> GeneratorOutput | torch.Tensor:
        if config is None:
            config = LLaDAGeneratorConfig()

        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_length = kwargs.get("block_length", config.block_length)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        stochastic_transfer = kwargs.get("stochastic_transfer", config.stochastic_transfer)
        return_dict_in_generate = kwargs.get("return_dict_in_generate", config.return_dict_in_generate)
        repetition_penalty = kwargs.get("repetition_penalty", config.repetition_penalty)

        assert 1 <= block_length
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Shape bookkeeping: per-sample prompt lengths and final canvas width -----
        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        # ----- Initialize canvas with EOS, copy inputs, and append mask tail -----
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p  # keep original prompt tokens
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = mask_id  # append `max_new_tokens` masks to be generated
        attention_mask = (x != eos_id).long() if B > 1 else None

        # Tokens that were *given* at the start (non-mask, non-EOS).
        unmasked_index = (x != mask_id) & (x != eos_id)
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(x, torch.as_tensor(cfg_keep_tokens, device=self.model.device))
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Block scheduling over the appended mask tail -----
        num_blocks = math.ceil(max_new_tokens / block_length)
        steps = math.ceil(steps / num_blocks)  # per-block step budget
        histories = [x.clone()] if return_dict_in_generate else None

        for b in range(num_blocks):
            # Build a per-sample mask *within this block* (aligned to each prompt's tail)
            block_mask_index = torch.zeros((B, block_length), dtype=torch.bool, device=x.device)

            for j in range(B):
                start = prompt_lens[j] + b * block_length
                end = min(start + block_length, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = x[j, start:end] == mask_id

            # Decide how many tokens to reveal per step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some steps may be skipped if there are no transfers
            effective_steps = num_transfer_tokens.size(1)

            # ----- Iterative reveal inside the current block -----
            for i in range(effective_steps):
                mask_index = x == mask_id  # current global mask map

                # Optional CFG: second forward where original prompt tokens are masked out
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_, attention_mask=attention_mask).logits  # Use attention mask here
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x, attention_mask=attention_mask).logits  # Use attention mask here

                # ----- Repetition penalty -----
                # Apply across all timesteps and vocabulary for tokens already present in canvas x
                logits = apply_repetition_penalty(
                    logits,
                    x,
                    penalty=repetition_penalty,
                    mask_token_id=mask_id,
                    eos_token_id=eos_id,
                )

                # Argmax decoding with optional Gumbel-Max noise for exploration
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T] predicted token ids

                # Per-position confidence used to pick which masks to commit this step
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )  # [B, T] confidence of predicted token
                elif remasking == "random":
                    x0_p = torch.rand(
                        (x0.shape[0], x0.shape[1]), device=x0.device
                    )  # random scores
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection window to the *current block's* tail region
                for j in range(B):
                    x0_p[j, prompt_lens[j] + (b + 1) * block_length :] = -np.inf

                # Only allow updates at currently masked positions; keep others fixed
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(
                    mask_index, x0_p, -np.inf
                )  # consider masked positions only

                # Pick exactly `num_transfer_tokens[j, i]` highest-confidence positions per sample
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    k = int(num_transfer_tokens[j, i].item())
                    if k > 0:
                        _, select_index = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_index] = True

                # Commit chosen predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

        # ----- Output format -----
        if not return_dict_in_generate:
            return x
        else:
            return GeneratorOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config,
        **kwargs
    ) -> GeneratorOutput | torch.Tensor:
        """
        Fill in-place the <|mdm_mask|> tokens contained in `inputs`.
        """
        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        block_length = kwargs.get("block_length", config.block_length)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        stochastic_transfer = kwargs.get("stochastic_transfer", config.stochastic_transfer)
        return_dict_in_generate = kwargs.get("return_dict_in_generate", config.return_dict_in_generate)
        repetition_penalty = kwargs.get("repetition_penalty", config.repetition_penalty)

        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Build canvas: right-pad with EOS to the max length in the batch -----
        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        # Default to a single block spanning the whole sequence
        if block_length is None:
            block_length = T

        assert 1 <= block_length
        assert 1 <= steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t
        attention_mask = (x != eos_id).long() if B > 1 else None

        # Tokens that were *given* at the start (non-mask, non-EOS).
        unmasked_index = (x != mask_id) & (x != eos_id)
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(x, torch.as_tensor(cfg_keep_tokens, device=self.model.device))
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Blockwise schedule over the *entire* (padded) sequence -----
        num_blocks = math.ceil(T / block_length)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict_in_generate else None

        # Create attention mask where eos_token_id is masked (set to 0)
        attention_mask = (x != eos_id).long()

        for b in range(num_blocks):
            start = b * block_length
            stop = min(start + block_length, T)

            # Per-sample view of which positions in this block are masks
            block_mask_index = torch.zeros(
                (B, block_length), dtype=torch.bool, device=self.model.device
            )
            widths = []
            for j in range(B):
                # Width limited by sample's true length and sequence end
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            # Decide how many tokens to reveal at each step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some blocks may have no masks => effective_steps == 0
            effective_steps = num_transfer_tokens.size(1)

            for s in range(effective_steps):
                mask_index_full = x == mask_id

                # ----- Forward pass (+ optional CFG) -----
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_, attention_mask=attention_mask).logits  # Use attention mask here
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x, attention_mask=attention_mask).logits  # Use attention mask here

                # ----- Repetition penalty -----
                logits = apply_repetition_penalty(
                    logits,
                    x,
                    penalty=repetition_penalty,
                    mask_token_id=mask_id,
                    eos_token_id=eos_id,
                )

                # Greedy with optional Gumbel-Max noise
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T]

                # Confidence used for choosing which masks to commit this step
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # [B, T]
                elif remasking == "random":
                    x0_p = torch.rand((B, T), device=self.model.device)
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection to the *current* block only
                for j in range(B):
                    end_j = start + widths[j]
                    # Outside current block => impossible to select
                    x0_p[j, :start] = -np.inf
                    x0_p[j, end_j:] = -np.inf

                # Only consider currently-masked positions as candidates
                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -np.inf)

                # Pick exactly num_transfer_tokens[j, s] positions per sample
                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer_tokens[j, s].item())
                    if k > 0:
                        _, select_idx = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_idx] = True

                # Commit selected predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

        # ----- Output format -----
        if not return_dict_in_generate:
            return x
        else:
            return GeneratorOutput(sequences=x, histories=histories)

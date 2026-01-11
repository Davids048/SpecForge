# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in HuggingFace Transformers.
# Portions of this code are adapted from:
#   - https://github.com/EleutherAI/gpt-neox (Apache License 2.0)
#   - https://github.com/huggingface/transformers (Apache License 2.0)
#   - https://github.com/SafeAILab/EAGLE (Apache License 2.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from specforge.modeling.draft.qwen3_jacobi import Qwen3ForCausalLMJacobi
from yunchang import EXTRACT_FUNC_DICT

from specforge.core.loss import LogSoftmaxLoss, _compute_loss
from specforge.distributed import (
    gather_outputs_and_unpad,
    get_sp_ring_group,
    get_sp_ulysses_group,
)
from specforge.modeling.draft import JacobiDraftModel
from .eagle3 import _compute_target_p_padded, _compute_metric_acc
from specforge.utils import padding



class JacobiModel(nn.Module):
    pass

class OnlineJacobiModel(JacobiModel):
    def __init__(
        self,
        draft_model: Qwen3ForCausalLMJacobi,
        length: int = 7,
        attention_backend="sdpa",
    ):
        """
        Args:
            target_model: the target model to extract hidden states.
            draft_model: the draft model to be trained.
            length: TTT length, it means how many turns to unroll during TTT.
        """
        super().__init__()
        self.draft_model = draft_model
        self.length = length
        self.attention_backend = attention_backend

        if self.attention_backend == "usp":
            self.extract_func = EXTRACT_FUNC_DICT["basic"]
            self.sp_ring_degree = torch.distributed.get_world_size(get_sp_ring_group())
            self.sp_ulysses_degree = torch.distributed.get_world_size(
                get_sp_ulysses_group()
            )
            self.sp_world_size = self.sp_ring_degree * self.sp_ulysses_degree
            self.sp_rank = torch.distributed.get_rank() % self.sp_world_size

    @torch.compile()
    def prepare_usp_input(self, full_input):
        raise NotImplementedError()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor, # Target model hidden states.
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        # DEBUG: Override inputs with simple test data
        DEBUG_TEST = os.getenv("DEBUG_JACOBI", False)
        if DEBUG_TEST:
            device = hidden_states.device
            dtype = hidden_states.dtype
            batch_size = hidden_states.shape[0]
            debug_seq_len = 7
            vocab_size = target.shape[-1]
            hidden_dim = hidden_states.shape[-1]  # num_target_layers * hidden_size

            input_ids = torch.arange(1, debug_seq_len + 1, device=device).unsqueeze(0).expand(batch_size, -1)
            attention_mask = torch.ones(batch_size, debug_seq_len, device=device)
            loss_mask = torch.ones(batch_size, debug_seq_len, 1, device=device)
            hidden_states = torch.randn(batch_size, debug_seq_len, hidden_dim, device=device, dtype=dtype)
            target = torch.randn(batch_size, debug_seq_len, vocab_size, device=device, dtype=dtype)
            target = F.softmax(target, dim=-1)  # make it a valid probability distribution
            print(f"DEBUG: input_ids={input_ids}, seq_len={debug_seq_len}, hidden_dim={hidden_dim}, vocab={vocab_size}, {loss_mask=}")

        # Step 1: handle vocab size
        target_p_padded, position_mask = _compute_target_p_padded(
            target=target,
            t2d=self.draft_model.t2d,
            loss_mask=loss_mask,
            length=self.length,
        )
        # Extend position mask and loss mask with 0 for the padded region.
        # This way, we don't need to use padding() to "shift" the mask, we can just use slicing.
        position_mask_padded = F.pad(
            position_mask,
            pad=(0, 0, 0, self.length),  # pad seq dimension at end
            mode="constant",
            value=0,
        )
        loss_mask_padded = F.pad(
            loss_mask,
            pad=(0, 0, 0, self.length),  # pad seq dimension at end
            mode="constant",
            value=0,
        )
        del target

        # basic info
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # Step 2: process kv cache, position ids and position ids
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # Step 4: handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        if self.attention_backend in ("sdpa", "usp"):
            attention_mask = self.draft_model.prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                batch_size=batch_size,
                seq_length=seq_length,
                past_key_values_length=past_key_values_length,
            )

        # plosses, vlosses, acces = self._seq_drafting(hidden_states, input_ids, target_p_padded, seq_length, batch_size, position_mask_padded, loss_mask_padded)
        plosses, vlosses, acces = self._parallel_drafting(
            hidden_states,
            input_ids,
            attention_mask,
            target_p_padded,
            position_mask_padded,
            loss_mask_padded,
        )
        return plosses, vlosses, acces

    def _parallel_drafting(
        self,
        hidden_states,      # BZ, T, 3 * HZ
        input_ids,          # BZ, T,
        attention_mask,
        target_p_padded,    # BZ, T + self.length, V
        position_mask_padded,   # BZ, T + self.length, 1
        loss_mask_padded,   # BZ, T + self.length, 1
    ):
        """
        hidden states: features from the target model, used to produce logits at each position.
        N: training sample length
        hidden:
            hb0 hb1 hb2 ... hb(N-1)
        input_ids: token produced by hidden states at the same position.
            0, 1, 2, ... N-1
        target_p:  logits produced after forwarding the token at the same position.

        in dflash: hidden acts as context, input ids are forwarded together with the masked tokens, and the target p is
        used to verify.

        """
        batch_size, input_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        vocab_size = self.draft_model.vocab_size

        plosses = []
        vlosses = []
        acces = []

        past_key_values = DynamicCache()
        for idx in range(self.length + 2):
            if idx == 0:
                # forward context section to store the target context.
                target_hidden = self.draft_model.project_hidden_states(hidden_states) # hidden from N target layers --> 1 hidden.
                noise_embedding = None
            elif idx == 1:
                # forward the first token in each block (which is the target produced free token)
                target_hidden = None
                noise_embedding = self.draft_model.embed_input_ids(input_ids).to(dtype)
            else:
                target_hidden = None
                noise_ids = torch.randint(0, vocab_size, input_ids.shape, device=device, dtype=input_ids.dtype)
                noise_embedding = self.draft_model.embed_input_ids(noise_ids).to(dtype)

            position_ids = torch.arange(0, input_len, device=device).unsqueeze(0).expand(batch_size, -1) + idx

            # Forward through draft model (placeholder)
            draft_output = self.draft_model.parallel_forward(
                target_hidden,
                noise_embedding,
                position_ids,
                attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            if idx <= 1:
                # First 2 iterations forward only the input ids to store in cache. no prediction performed. No loss calculated.
                continue
            assert draft_output is not None
            # Compute logits
            draft_logits = self.draft_model.compute_logits(draft_output)

            # target_p from idx to idx + input_len
            pred_pos = idx - 2
            target_p = target_p_padded[:, pred_pos : pred_pos + input_len, :]
            position_mask = position_mask_padded[:, pred_pos : pred_pos + input_len, :]
            loss_mask = loss_mask_padded[:, pred_pos : pred_pos + input_len, :]
            logits = gather_outputs_and_unpad(draft_logits, gather_dim=1)

            # Step 5.5: record metrics first as we in-place modify logits
            with torch.no_grad():
                acces.append(
                    _compute_metric_acc(
                        logits=logits,
                        target_p=target_p,
                        position_mask=position_mask,
                        loss_mask=loss_mask,
                    )
                )

            # Step 5.6: calculate loss, in-place modifies logits!
            vocab_size = logits.shape[-1]
            if vocab_size > 65536:
                loss = _compute_loss(logits, target_p, position_mask)
            else:
                loss = LogSoftmaxLoss.apply(logits, target_p, position_mask)
            plosses.append(loss)

        return plosses, vlosses, acces

    def _seq_drafting(
        self,
        hidden_states,
        input_ids,
        target_p_padded,
        seq_length,
        batch_size,
        position_mask_padded,
        loss_mask_padded,
    ):
        # Step 5. Perform parallel drafting training
        # block = [seed_token, M, M, ...] with self.length tokens to predict
        block_len = self.length + 1
        device = hidden_states.device

        # Get mask token id from draft model (or use a placeholder)
        mask_token_id = getattr(self.draft_model, 'mask_token_id', 0)

        # Collect logits for each prediction position across all training positions
        # Shape will be [batch, num_train_pos, vocab] for each pred_pos
        all_logits = {k: [] for k in range(self.length)}

        # Loop over training positions
        num_training_positions = seq_length
        for train_pos in range(num_training_positions):
            # Context: target hidden states up to position train_pos (inclusive)
            context_hidden = hidden_states[:, :train_pos + 1, :]  # [batch, train_pos+1, hidden]

            # Block input: token at train_pos + self.length mask tokens
            seed_token = input_ids[:, train_pos:train_pos + 1]  # [batch, 1]

            mask_tokens = torch.full(
                (batch_size, self.length), mask_token_id, dtype=torch.long, device=device
            )  # [batch, self.length]
            block_input_ids = torch.cat([seed_token, mask_tokens], dim=1)  # [batch, block_len]

            # Convert to embeddings
            block_embeds = self.draft_model.embed_input_ids(block_input_ids)  # [batch, block_len, hidden]
            block_embeds = block_embeds.to(context_hidden.dtype)

            # Position ids: need full range [0, ctx_len + block_len) for rotary embeddings
            # (context K/V use positions 0..ctx_len-1, block K/V use positions ctx_len..ctx_len+block_len-1)
            ctx_len = train_pos + 1
            full_position_ids = torch.arange(
                0, ctx_len + block_len, device=device
            ).unsqueeze(0).expand(batch_size, -1)

            # Forward through draft model
            draft_output = self.draft_model.forward(
                position_ids=full_position_ids,
                noise_embedding=block_embeds,
                target_hidden=context_hidden,  # unprojected [batch, ctx_len, num_target_layers * hidden]
                attention_mask=None,  # DFlash attention handles its own masking
                use_cache=False,
            )
            draft_logits = self.draft_model.compute_logits(draft_output)[:, 1:, :]  # skip seed token

            # Collect logits for each prediction position
            for pred_pos in range(self.length):
                all_logits[pred_pos].append(draft_logits[:, pred_pos:pred_pos+1, :])

        # Stack logits: [batch, num_train_pos, vocab] for each pred_pos
        plosses = []
        vlosses = []
        acces = []
        for pred_pos in range(self.length):
            logits = torch.cat(all_logits[pred_pos], dim=1)  # [batch, num_train_pos, vocab]
            target_p = target_p_padded[:, pred_pos : pred_pos + seq_length, :]  # [batch, num_train_pos, vocab]
            pos_mask = position_mask_padded[:, pred_pos : pred_pos + seq_length, :]  # [batch, num_train_pos, 1]

            # Compute loss - use PyTorch fallback for large vocab (Triton limit is 65536)
            vocab_size = logits.shape[-1]
            if vocab_size > 65536:
                loss = _compute_loss(logits, target_p, pos_mask)
            else:
                loss = LogSoftmaxLoss.apply(logits, target_p, pos_mask)
            plosses.append(loss)

            with torch.no_grad():
                cur_loss_mask = loss_mask_padded[:, pred_pos : pred_pos + seq_length, :]
                acc = _compute_metric_acc(logits, target_p, pos_mask, cur_loss_mask)
                acces.append(acc)
        return plosses, vlosses, acces

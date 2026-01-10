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
from yunchang import EXTRACT_FUNC_DICT

from specforge.core.loss import LogSoftmaxLoss
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
        draft_model: JacobiDraftModel,
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

        # Step 2: project the concatenated hidden states to the target hidden size
        hidden_states = self.draft_model.project_hidden_states(hidden_states)

        # Step 3: process kv cache, position ids and position ids
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

            # Position ids for the block: train_pos to train_pos + block_len - 1
            ctx_len = train_pos + 1
            block_position_ids = torch.arange(
                train_pos, train_pos + block_len, device=device
            ).unsqueeze(0).expand(batch_size, -1)

            # Forward through draft model (placeholder - returns dummy logits)
            # TODO: call self.draft_model.backbone(...) with proper args
            draft_logits = torch.zeros(
                batch_size, block_len, self.draft_model.draft_vocab_size,
                device=device,
                requires_grad=True,
            )[:, 1:, :]

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

            # Compute loss and accuracy using existing functions
            loss = LogSoftmaxLoss.apply(logits, target_p, pos_mask)
            plosses.append(loss)

            with torch.no_grad():
                cur_loss_mask = loss_mask_padded[:, pred_pos : pred_pos + seq_length, :]
                acc = _compute_metric_acc(logits, target_p, pos_mask, cur_loss_mask)
                acces.append(acc)

        return plosses, vlosses, acces

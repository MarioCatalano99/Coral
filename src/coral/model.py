from torch import nn
import torch
import math
import sys
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from torch.nn import CrossEntropyLoss


class CoralConfig(PretrainedConfig):
    model_type = "Coral"

    def __init__(
        self,
        classifier_dropout: float = None,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        layer_norm_eps: float = 1e-12,
        num_attention_heads: int = 12,
        num_cross_layers: int = 2,
        torch_dtype: str = "float32",
        transformers_version: str = "4.29.2",
        use_cache: bool = True,
        protein_encoder_hidden_size: int = 640,
        attention_probs_dropout_prob: float = 0.0,
        protein_vocab_size: int = 33,
        rna_vocab_size: int = 4096,
        num_answers: int = 2,
        rna_encoder_checkpoint: str = None,
        protein_encoder_checkpoint: str = None,
        **kwargs,
    ):
        self.classifier_dropout = classifier_dropout
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.num_attention_heads = num_attention_heads
        self.num_cross_layers = num_cross_layers
        self.torch_dtype = torch_dtype
        self.transformers_version = transformers_version
        self.use_cache = use_cache
        self.protein_encoder_hidden_size = protein_encoder_hidden_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.protein_vocab_size = protein_vocab_size
        self.rna_vocab_size = rna_vocab_size
        self.num_answers = num_answers
        self.rna_encoder_checkpoint = rna_encoder_checkpoint
        self.protein_encoder_checkpoint = protein_encoder_checkpoint
        super().__init__(**kwargs)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))




def mean_pooling(last_hidden_state, attention_mask):
    """
    Performs mean pooling on BERT-style token embeddings, correctly ignoring padding.

    Args:
        last_hidden_state (torch.Tensor): Shape [B, L, D]
        attention_mask (torch.Tensor): Shape [B, L] (1 for valid tokens, 0 for padding)

    Returns:
        torch.Tensor: Mean pooled representation of shape [B, D]
    """
    # 1. Expand the attention mask to match the hidden state dimensions [B, L, D].
    # We cast to the same dtype as the hidden state to ensure compatibility with AMP (fp16/bf16).
    input_mask_expanded = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)

    # 2. Sum the token embeddings, ignoring masked (padding) tokens.
    # (Multiplying by 0 for padding tokens before summing)
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)

    # 3. Calculate the number of valid tokens for each sequence.
    # Clamp minimum to 1e-9 to avoid division by zero stability issues.
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    # 4. Divide sum by the actual count of valid tokens.
    return sum_embeddings / sum_mask

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}


BertLayerNorm = torch.nn.LayerNorm



class BertAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, input_att_mask=None, ctx_att_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)


        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask
        if input_att_mask is not None and ctx_att_mask is not None:
            attention_scores = attention_scores.masked_fill(
                ctx_att_mask == 0.0, float("-inf")
            )

            attention_scores = attention_scores.masked_fill(
                input_att_mask.transpose(-1, -2) == 0.0, float("-inf")
            )
        elif ctx_att_mask is not None:
            attention_scores = attention_scores.masked_fill(
                ctx_att_mask == 0.0, float("-inf")
            )


        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = attention_probs.masked_fill(
            attention_probs.isnan(), 0.0
        )

        # Store attention weights if requested
        attention_weights = attention_probs if output_attentions else None

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_weights) if output_attentions else (context_layer,)
        return outputs


class BertAttOutput(nn.Module):
    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, input_att_mask=None, ctx_att_mask=None, output_attentions=False):
        att_outputs = self.att(input_tensor, ctx_tensor,
                               input_att_mask=input_att_mask,
                               ctx_att_mask=ctx_att_mask,
                               output_attentions=output_attentions)
        attention_output = self.output(att_outputs[0], input_tensor)

        outputs = (attention_output,)
        if output_attentions:
            outputs = outputs + (att_outputs[1],)  # Add attention weights
        return outputs


class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask, output_attentions=False):
        self_outputs = self.self(input_tensor, input_tensor, ctx_att_mask=attention_mask, output_attentions=output_attentions)
        attention_output = self.output(self_outputs[0], input_tensor)

        outputs = (attention_output,)
        if output_attentions:
            outputs = outputs + (self_outputs[1],)
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, str)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        
        # Normalization layers
        self.rna_layernorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.protein_layernorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Standard projection
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, rna_hidden_states, rna_attention_mask, protein_hidden_states, protein_attention_mask):
        # 1. Mean Pool
        rna_pooled = mean_pooling(rna_hidden_states, rna_attention_mask)
        protein_pooled = mean_pooling(protein_hidden_states, protein_attention_mask)
        
        # 2. Normalize the pooled vectors independently
        # This ensures neither modality dominates due to magnitude differences
        rna_pooled = self.rna_layernorm(rna_pooled)
        protein_pooled = self.protein_layernorm(protein_pooled)

        # 3. Concatenate
        cat_output = torch.cat([rna_pooled, protein_pooled], axis=-1)
        
        # 4. Project
        pooled_output = self.dense(cat_output)
        pooled_output = self.activation(pooled_output)
        
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config, hidden_size):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, str)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config, bert_model_embedding_weights.size(1))


        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, rna_encoder_embedding_weights, protein_encoder_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.rna_predictions = BertLMPredictionHead(config, rna_encoder_embedding_weights)
        self.protein_predictions = BertLMPredictionHead(config, protein_encoder_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, config.num_answers)

    def forward(self, rna_sequence_output, protein_sequence_output, pooled_output):
        rna_prediction_scores = self.rna_predictions(rna_sequence_output)
        protein_prediction_scores = self.protein_predictions(protein_sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return rna_prediction_scores, protein_prediction_scores, seq_relationship_score

"""
---------------------------------------------------------------------------------------
      Above modules are copied from BERT (pytorch-transformer) with modifications.
---------------------------------------------------------------------------------------
"""


class CrossModalityLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ----------------------------------------------------------------------
        # Cross-Attention Modules
        # ----------------------------------------------------------------------
        # Module 1: RNA is the Query, Protein is the Key/Value
        self.r2p_cross_attention = BertCrossattLayer(config) 
        
        # Module 2: Protein is the Query, RNA is the Key/Value
        self.p2r_cross_attention = BertCrossattLayer(config) 
        # ----------------------------------------------------------------------

        # Self-attention Layers
        self.rna_self_att = BertSelfattLayer(config)
        self.protein_self_att = BertSelfattLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.rna_inter = BertIntermediate(config)
        self.rna_output = BertOutput(config)
        self.protein_inter = BertIntermediate(config)
        self.protein_output = BertOutput(config)

    def cross_att(self, rna_input, rna_attention_mask, protein_input, protein_attention_mask, output_attentions=False):
        """Cross-attention between RNA and Protein features"""

        # ------------------------------------------------------------------
        # Direction 1: RNA attends to Protein (Using R2P specific weights)
        # ------------------------------------------------------------------
        rna_att_outputs = self.r2p_cross_attention(
            rna_input, protein_input,
            input_att_mask=rna_attention_mask,
            ctx_att_mask=protein_attention_mask,
            output_attentions=output_attentions
        )
        rna_att_output = rna_att_outputs[0]

        # ------------------------------------------------------------------
        # Direction 2: Protein attends to RNA (Using P2R specific weights)
        # ------------------------------------------------------------------
        protein_att_outputs = self.p2r_cross_attention(
            protein_input, rna_input,
            input_att_mask=protein_attention_mask,
            ctx_att_mask=rna_attention_mask,
            output_attentions=output_attentions
        )
        protein_att_output = protein_att_outputs[0]

        outputs = (rna_att_output, protein_att_output)

        if output_attentions:
            # Return attention weights: (rna->protein_attention, protein->rna_attention)
            rna_to_protein_att = rna_att_outputs[1] if len(rna_att_outputs) > 1 else None
            protein_to_rna_att = protein_att_outputs[1] if len(protein_att_outputs) > 1 else None
            outputs = outputs + (rna_to_protein_att, protein_to_rna_att)

        return outputs

    def self_att(self, rna_input, rna_attention_mask, protein_input, protein_attention_mask, output_attentions=False):
        """Self-attention separately for RNA and Protein features (Unchanged)"""
        rna_att_outputs = self.rna_self_att(rna_input, rna_attention_mask, output_attentions=output_attentions)
        protein_att_outputs = self.protein_self_att(protein_input, protein_attention_mask,
                                                    output_attentions=output_attentions)

        rna_att_output = rna_att_outputs[0]
        protein_att_output = protein_att_outputs[0]

        outputs = (rna_att_output, protein_att_output)

        if output_attentions:
            rna_self_att = rna_att_outputs[1] if len(rna_att_outputs) > 1 else None
            protein_self_att = protein_att_outputs[1] if len(protein_att_outputs) > 1 else None
            outputs = outputs + (rna_self_att, protein_self_att)

        return outputs

    def output_fc(self, rna_input, protein_input):
        """Feed-forward networks for final processing (Unchanged)"""
        rna_inter_output = self.rna_inter(rna_input)
        protein_inter_output = self.protein_inter(protein_input)

        rna_output = self.rna_output(rna_inter_output, rna_input)
        protein_output = self.protein_output(protein_inter_output, protein_input)

        return rna_output, protein_output

    def forward(self, rna_feats, rna_attention_mask, protein_feats, protein_attention_mask, output_attentions=False):
        """Forward pass through the cross-modality layer (Unchanged logic, just passes through updated cross_att)"""

        rna_att_output = rna_feats
        protein_att_output = protein_feats

        # Cross-attention between modalities
        cross_att_outputs = self.cross_att(
            rna_att_output, rna_attention_mask,
            protein_att_output, protein_attention_mask,
            output_attentions=output_attentions
        )
        rna_att_output, protein_att_output = cross_att_outputs[0], cross_att_outputs[1]

        # Self-attention within each modality
        self_att_outputs = self.self_att(
            rna_att_output, rna_attention_mask,
            protein_att_output, protein_attention_mask,
            output_attentions=output_attentions
        )
        rna_att_output, protein_att_output = self_att_outputs[0], self_att_outputs[1]

        # Final feed-forward processing
        rna_output, protein_output = self.output_fc(rna_att_output, protein_att_output)

        outputs = (rna_output, protein_output)

        if output_attentions:
            # Collect all attention weights
            cross_attentions = cross_att_outputs[2:]
            self_attentions = self_att_outputs[2:]
            outputs = outputs + (cross_attentions, self_attentions)

        return outputs


class CrossModalityEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Number of layers
        self.num_cross_layers = config.num_cross_layers
        self.cross_layers = nn.ModuleList(
            [CrossModalityLayer(config) for _ in range(self.num_cross_layers)]
        )

    def forward(self, rna_feats, rna_attention_mask, protein_feats, protein_attention_mask=None,
                output_attentions=False):

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        extended_rna_attention_mask = rna_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_rna_attention_mask = extended_rna_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility

        if protein_attention_mask is not None:
            extended_protein_attention_mask = protein_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_protein_attention_mask = extended_protein_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        else:
            extended_protein_attention_mask = None

        # Store attention weights from all layers
        if output_attentions:
            all_rna_cross_attentions = []
            all_protein_cross_attentions = []
            all_rna_self_attentions = []
            all_protein_self_attentions = []

            # Run cross-modality layers
        for layer_module in self.cross_layers:

            layer_outputs = layer_module(
                rna_feats, extended_rna_attention_mask,
                protein_feats, extended_protein_attention_mask,
                output_attentions=output_attentions
            )

            rna_feats, protein_feats = layer_outputs[0], layer_outputs[1]

            if output_attentions:
                cross_attentions = layer_outputs[2] if len(layer_outputs) > 2 else (None, None)
                self_attentions = layer_outputs[3] if len(layer_outputs) > 3 else (None, None)
                all_rna_cross_attentions.append(cross_attentions[0])
                all_protein_cross_attentions.append(cross_attentions[1])
                all_rna_self_attentions.append(self_attentions[0])
                all_protein_self_attentions.append(self_attentions[1])

        outputs = (rna_feats, protein_feats)
        if output_attentions:
            all_cross_attentions = [torch.stack(all_rna_cross_attentions, dim=1),
                                    torch.stack(all_protein_cross_attentions, dim=1)]
            all_self_attentions = [torch.stack(all_rna_self_attentions, dim=1),
                                    torch.stack(all_protein_self_attentions, dim=1)]
            outputs = outputs + (all_cross_attentions, all_self_attentions)

        return outputs


class Coral(nn.Module):

    def __init__(self, config, rna_encoder_checkpoint, protein_encoder_checkpoint):
        super().__init__()
        self.rna_encoder = AutoModel.from_pretrained(rna_encoder_checkpoint, trust_remote_code=True)
        self.protein_encoder = AutoModel.from_pretrained(protein_encoder_checkpoint, trust_remote_code=True)
        self.projection = nn.Linear(config.protein_encoder_hidden_size, config.hidden_size)
        self.cross_encoder = CrossModalityEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, rna_batch, protein_batch, output_attentions=False):

        rna_hidden_states, _ = self.rna_encoder(**rna_batch)


        protein_hidden_states = self.protein_encoder(
            input_ids=protein_batch["input_ids"],
            attention_mask=protein_batch["attention_mask"]
        ).last_hidden_state


        protein_hidden_states = self.projection(protein_hidden_states)

        cross_encoder_outputs = self.cross_encoder(
            rna_feats=rna_hidden_states,
            protein_feats=protein_hidden_states,
            rna_attention_mask=rna_batch["attention_mask"],
            protein_attention_mask=protein_batch["attention_mask"],
            output_attentions=output_attentions
        )

        rna_feats, protein_feats = cross_encoder_outputs[0], cross_encoder_outputs[1]
        pooled_output = self.pooler(rna_feats, rna_batch["attention_mask"], protein_feats, protein_batch["attention_mask"])

        outputs = ((rna_feats, protein_feats), pooled_output)

        if output_attentions:
            cross_attentions = cross_encoder_outputs[2] if len(cross_encoder_outputs) > 2 else None
            self_attentions = cross_encoder_outputs[3] if len(cross_encoder_outputs) > 3 else None
            outputs = outputs + (cross_attentions, self_attentions)

        return outputs


class CoralPretraining(PreTrainedModel):
    config_class = CoralConfig
    base_model_prefix = "Coral"

    def __init__(self, config):
        super().__init__(config)
        # Configuration
        self.config = config

        self.softmax = nn.Softmax(dim=1)

        # Model backbone
        self.bert = Coral(config, config.rna_encoder_checkpoint, config.protein_encoder_checkpoint)

        # Pre-training heads
        self.cls = BertPreTrainingHeads(
            config,
            self.bert.rna_encoder.embeddings.word_embeddings.weight,
            self.bert.protein_encoder.embeddings.word_embeddings.weight
        )



    def forward(self, rna_batch, protein_batch, matched_label=None, task_mask_lm=True, task_matched=True, output_attentions=False, mlm_scaling_factor=1):

        bert_outputs = self.bert(rna_batch, protein_batch, output_attentions=output_attentions)

        (rna_output, protein_output), pooled_output = bert_outputs[0], bert_outputs[1]
        protein_output = protein_output @ self.bert.projection.weight

        # Extract attention weights if requested
        cross_attentions = bert_outputs[2] if output_attentions and len(bert_outputs) > 2 else None
        self_attentions = bert_outputs[3] if output_attentions and len(bert_outputs) > 3 else None

        if "labels_for_MLM" in rna_batch:
            rna_masked_lm_labels = rna_batch["labels_for_MLM"]
        else:
            rna_masked_lm_labels = None

        if "labels_for_MLM" in protein_batch:
            protein_masked_lm_labels = protein_batch["labels_for_MLM"]
        else:
            protein_masked_lm_labels = None


        rna_prediction_scores, protein_prediction_scores, cross_relationship_score = self.cls(
                rna_output, protein_output, pooled_output
        )

        total_loss = 0.
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        losses = ()

        if rna_masked_lm_labels is not None and task_mask_lm:
            rna_masked_lm_loss = loss_fct(
                rna_prediction_scores.view(-1, self.config.rna_vocab_size),
                rna_masked_lm_labels.view(-1)
            )
            total_loss += mlm_scaling_factor * rna_masked_lm_loss
            losses += (rna_masked_lm_loss.detach(),)

        if protein_masked_lm_labels is not None and task_mask_lm:
            protein_masked_lm_loss = loss_fct(
                protein_prediction_scores.view(-1, self.config.protein_vocab_size),
                protein_masked_lm_labels.view(-1)
            )
            total_loss += mlm_scaling_factor * protein_masked_lm_loss
            losses += (protein_masked_lm_loss.detach(),)

        if matched_label is not None and task_matched:
            matched_loss = loss_fct(
                cross_relationship_score.view(-1, 2),
                matched_label.view(-1)
            )
            total_loss += matched_loss
            losses += (matched_loss.detach(),)

        match_probabilities = self.softmax(cross_relationship_score)
        predictions = match_probabilities.max(dim=1).indices

        if losses:
            outputs = (total_loss, torch.stack(losses), predictions, match_probabilities)
        else:
            outputs = predictions, match_probabilities

        if output_attentions:
            outputs = outputs + (cross_attentions, self_attentions)

        return outputs
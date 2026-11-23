"""Cross-attention extraction for the interpretability analyses.

`AttentionAnalyzer` runs a batch through the model with `output_attentions=True`,
takes the **last** cross-attention layer, optionally averages a subset of heads,
and strips CLS/SEP/PAD positions so the returned matrices line up with raw
sequence coordinates.
"""

from typing import Dict, List, Optional

import torch


def decode_sequences(input_ids: torch.Tensor, tokenizer) -> List[str]:
    """Decode a batch of token ids to space-joined token strings, minus specials."""
    return tokenizer.batch_decode(input_ids, skip_special_tokens=True)


class AttentionAnalyzer:
    """Extract last-layer cross-attention trimmed to raw sequence coordinates."""

    def __init__(self, model, rna_tokenizer, protein_tokenizer,
                 device=torch.device("cuda")):
        self.model = model
        self.rna_tokenizer = rna_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.device = device

    def extract_cross_attention_scores(
        self,
        rna_batch: Dict,
        protein_batch: Dict,
        subset: Optional[List[int]] = None,
    ) -> Dict:
        """Extract last-layer cross-attention for a batch of RNA-protein pairs.

        Args:
            rna_batch: RNA batch with input_ids and attention_mask.
            protein_batch: Protein batch with input_ids and attention_mask.
            subset: Head indices to average. When None every head is kept and the
                returned arrays carry a leading head axis.

        Returns:
            Dict of per-pair attention matrices, one list per direction.
        """
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():

            # Get attention scores
            _, _, cross_attentions, _ = self.model(
                rna_batch, protein_batch, output_attentions=True
            )

            rna_to_protein_att, protein_to_rna_att = cross_attentions

            # Layer index -1: cross-attention is always read from the last layer.
            if subset:
                rna_to_protein_att = rna_to_protein_att[:, -1, subset].mean(dim=1)
                protein_to_rna_att = protein_to_rna_att[:, -1, subset].mean(dim=1)
            else:
                rna_to_protein_att = rna_to_protein_att[:, -1, ...]
                protein_to_rna_att = protein_to_rna_att[:, -1, ...]

            # Get token sequences
            rna_tokens = decode_sequences(rna_batch["input_ids"], self.rna_tokenizer)
            protein_tokens = decode_sequences(protein_batch["input_ids"],
                                              self.protein_tokenizer)

            processed_rna_to_protein_att = []
            processed_protein_to_rna_att = []

            batch_size = rna_to_protein_att.size(0)
            for i in range(batch_size):

                protein_len = len(protein_tokens[i].split(" "))
                rna_len = len(rna_tokens[i].split(" "))

                # 1:len+1 drops the leading CLS and everything from SEP onward.
                if subset:
                    rna_att = rna_to_protein_att[i][1:rna_len + 1, 1:protein_len + 1].cpu().numpy()
                    protein_att = protein_to_rna_att[i][1:protein_len + 1, 1:rna_len + 1].cpu().numpy()
                else:
                    rna_att = rna_to_protein_att[i][:, 1:rna_len + 1, 1:protein_len + 1].cpu().numpy()
                    protein_att = protein_to_rna_att[i][:, 1:protein_len + 1, 1:rna_len + 1].cpu().numpy()

                processed_rna_to_protein_att.append(rna_att)
                processed_protein_to_rna_att.append(protein_att)

        return {
            "rna_to_protein_attentions": processed_rna_to_protein_att,
            "protein_to_rna_attentions": processed_protein_to_rna_att,
        }

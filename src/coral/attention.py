import torch
from typing import List, Tuple, Dict
import warnings



warnings.filterwarnings('ignore')


def decode_sequences(input_ids: torch.Tensor, tokenizer) -> Tuple[List[str], List[str]]:
    """
    Decode tokenized sequences back to readable format

    Returns:
        Tuple of (RNA sequences, Protein sequences)
    """
    sequences = tokenizer.batch_decode(input_ids, skip_special_tokens=True)


    return sequences


class AttentionAnalyzer:
    """Utility class for analyzing cross-attention patterns in RNA-Protein interactions"""

    def __init__(self, model, rna_tokenizer, protein_tokenizer, device=torch.device('cuda')):
        self.model = model
        self.rna_tokenizer = rna_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.device = device

    def extract_cross_attention_scores(self, rna_batch: Dict, protein_batch: Dict, contact_labels: torch.Tensor, subset: List=None) -> Dict:
        """
        Extract attention scores from the model for given batch of RNA-Protein pairs

        Args:
            rna_batch: RNA batch with input_ids and attention_mask
            protein_batch: Protein batch with input_ids and attention_mask
            contact_labels: contact labels for the pair

        Returns:
            Dictionary containing attention scores and predictions
        """
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():

            # Get attention scores
            _, _, cross_attentions, _ = self.model(
                rna_batch, protein_batch, output_attentions=True
            )


            rna_to_protein_att, protein_to_rna_att = cross_attentions

            if subset:
                rna_to_protein_att = rna_to_protein_att[:, -1, subset].mean(dim=1)
                protein_to_rna_att = protein_to_rna_att[:, -1, subset].mean(dim=1)
            else:
                rna_to_protein_att = rna_to_protein_att[:, -1, ...]
                protein_to_rna_att = protein_to_rna_att[:, -1, ...]


            # Get token sequences
            rna_tokens = decode_sequences(rna_batch["input_ids"], self.rna_tokenizer)

            protein_tokens = decode_sequences(protein_batch["input_ids"], self.protein_tokenizer)

            processed_rna_to_protein_att = []
            processed_protein_to_rna_att = []

            processed_rna2protein_contact_labels = []
            processed_protein2rna_contact_labels = []

            for i in range(contact_labels.size(0)):

                protein_len = len(protein_tokens[i].split(" "))
                rna_len = len(rna_tokens[i].split(" "))

                if subset:
                    rna_att = rna_to_protein_att[i][1:rna_len + 1, 1:protein_len + 1].cpu().numpy()
                    protein_att = protein_to_rna_att[i][1:protein_len + 1, 1:rna_len + 1].cpu().numpy()
                else:
                    rna_att = rna_to_protein_att[i][:, 1:rna_len + 1, 1:protein_len + 1].cpu().numpy()
                    protein_att = protein_to_rna_att[i][:, 1:protein_len + 1, 1:rna_len + 1].cpu().numpy()

                rna2protein_contact_labels = contact_labels[i][1:rna_len + 1, 1:protein_len + 1].cpu().numpy()
                protein2rna_contact_labels = rna2protein_contact_labels.T

                processed_rna_to_protein_att.append(rna_att)
                processed_protein_to_rna_att.append(protein_att)

                processed_rna2protein_contact_labels.append(rna2protein_contact_labels)
                processed_protein2rna_contact_labels.append(protein2rna_contact_labels)



        return {
            'rna_to_protein_attentions': processed_rna_to_protein_att,
            'protein_to_rna_attentions': processed_protein_to_rna_att,
            'rna2protein_contact_labels': processed_rna2protein_contact_labels,
            'protein2rna_contact_labels': processed_protein2rna_contact_labels
        }

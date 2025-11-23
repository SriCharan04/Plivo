import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torchcrf import CRF 

# Assuming labels.py defines LABEL2ID and ID2LABEL
from labels import LABEL2ID, ID2LABEL 

# --- Custom Model Class ---
class DistilBertCrfForTokenClassification(nn.Module):
    """
    DistilBERT model with a standard linear classification head followed by a CRF layer.
    Includes internal assembly of spans based on token indices.
    """
    def __init__(self, model_name, num_labels):
        super().__init__()
        
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.distilbert = AutoModel.from_pretrained(model_name, config=config)
        self.num_labels = num_labels
        
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        
        self.id_to_label = ID2LABEL # Store mapping for internal decoding
        
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.distilbert.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def _assemble_spans_from_tokens(self, prediction_list, attention_mask):
        """
        [NEW FEATURE] Converts predicted label IDs (token indices) into structured entity spans.
        This operates on the predicted token IDs, NOT character offsets.
        """
        batch_spans = []
        
        for batch_index in range(prediction_list.size(0)):
            predictions = prediction_list[batch_index].tolist()
            mask = attention_mask[batch_index].bool().tolist()
            
            entities = []
            current_entity = None
            
            # Iterate through predicted labels for the current sequence
            for i, pred_id in enumerate(predictions):
                # Only process non-padded tokens
                if not mask[i]:
                    break
                    
                label_tag = self.id_to_label.get(pred_id)
                
                # Skip [CLS], [SEP] or unknown tags
                if label_tag is None or label_tag in ['[CLS]', '[SEP]', '[PAD]']:
                    continue
                    
                tag_prefix = label_tag[0]
                tag_type = label_tag[2:]
                
                if tag_prefix == 'B':
                    if current_entity:
                        entities.append(current_entity)
                    
                    # Start a new entity with token indices
                    current_entity = {
                        "label": tag_type,
                        "token_start": i,
                        "token_end": i
                    }
                    
                elif tag_prefix == 'I':
                    # Continue the current entity if the type matches
                    if current_entity and current_entity['label'] == tag_type:
                        current_entity['token_end'] = i
                    else:
                        # Robustness: Treat invalid I-tags (not following B or matching I) as O
                        current_entity = None
                        
                elif tag_prefix == 'O':
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None

            if current_entity:
                entities.append(current_entity)
            
            batch_spans.append(entities)
            
        return batch_spans

    def forward(self, input_ids, attention_mask=None, labels=None):
        
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=None
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        
        # Determine the best label sequence (Viterbi path)
        prediction_list = self.crf.decode(logits, mask=attention_mask.bool())
        # --- FIX: Manually Pad Prediction List to Match Batch Sequence Length ---
        max_len = input_ids.size(1) # Get the current max sequence length in the batch
        batch_size = len(prediction_list)
        prediction_tensor = torch.zeros((batch_size, max_len), dtype=torch.long, device=logits.device)

        for i, pred_seq in enumerate(prediction_list):
            # Copy the predicted sequence (pred_seq) into the tensor, padding the rest with zeros
            seq_len = len(pred_seq)
            prediction_tensor[i, :seq_len] = torch.tensor(pred_seq, device=logits.device)
        # --- Internal Span Assembly ---
        # Assemble entities based on token indices (B, I, O sequence)
        assembled_spans = self._assemble_spans_from_tokens(prediction_tensor, attention_mask)

        if labels is not None:
            # Loss calculation (Requires safety fix)
            safe_labels = labels.clone()
            safe_labels[safe_labels < 0] = 0
            loss = -self.crf(logits, safe_labels, mask=attention_mask.bool(), reduction='mean')
            
            return {
                "loss": loss, 
                "logits": logits, 
                "predictions": prediction_tensor, 
                "assembled_spans": assembled_spans # NEW OUTPUT
            }
        
        else:
            # Inference mode
            return {
                "logits": logits, 
                "predictions": prediction_tensor, 
                "assembled_spans": assembled_spans # NEW OUTPUT
            }


# --- Factory Function ---
def create_model(model_name: str):
    """Initializes the custom DistilBERT-CRF model."""
    num_labels = len(LABEL2ID)
    model = DistilBertCrfForTokenClassification(model_name, num_labels)
    return model
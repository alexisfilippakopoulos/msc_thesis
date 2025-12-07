from encodings.punycode import T
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

__all__ = ['BertTextEncoder']

TRANSFORMERS_MAP = {
    'bert': (BertModel, BertTokenizer),
    'roberta': (RobertaModel, RobertaTokenizer),
}

class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=False, transformers='bert', pretrained='bert-base-uncased'):
        super().__init__()

        tokenizer_class = TRANSFORMERS_MAP[transformers][1]
        model_class = TRANSFORMERS_MAP[transformers][0]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained)
        self.model = model_class.from_pretrained(pretrained)
        self.use_finetune = use_finetune

    def get_tokenizer(self):
        return self.tokenizer

    # def from_text(self, text):
    #     """
    #     text: raw data
    #     """
    #     input_ids = self.get_id(text)
    #     with torch.no_grad():
    #         last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
    #     return last_hidden_states.squeeze()

    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states


class HFTextEncoder(nn.Module):
    def __init__(self, transformers='bert', pretrained='bert-base-uncased', max_len=39):
        super().__init__()

        tokenizer_class = TRANSFORMERS_MAP[transformers][1]
        model_class = TRANSFORMERS_MAP[transformers][0]
        self.max_len = max_len
        self.tokenizer = tokenizer_class.from_pretrained(
            pretrained,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt' 
        )
        self.model = model_class.from_pretrained(pretrained)
        self.device = "cuda"
        
    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, text):
        """
        text(List[str]): raw data
        """
        text = \
            self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            )
        input_ids = text['input_ids'].to(self.device)
        segment_ids = text['token_type_ids'].to(self.device)
        input_mask = text['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids
            )  # Models outputs are now tuples
        last_hidden_states = outputs[0]
        return last_hidden_states

    # def forward(self, text):
    #     """
    #     text: (batch_size, 3, seq_len)
    #     3: input_ids, input_mask, segment_ids
    #     input_ids: input_ids,
    #     input_mask: attention_mask,
    #     segment_ids: token_type_ids
    #     """
    #     input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
    #     if self.use_finetune:
    #         last_hidden_states = self.model(input_ids=input_ids,
    #                                         attention_mask=input_mask,
    #                                         token_type_ids=segment_ids)[0]  # Models outputs are now tuples
    #     else:
    #         with torch.no_grad():
    #             last_hidden_states = self.model(input_ids=input_ids,
    #                                             attention_mask=input_mask,
    #                                             token_type_ids=segment_ids)[0]  # Models outputs are now tuples
    #     return last_hidden_states

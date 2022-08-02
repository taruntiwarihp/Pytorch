from torch import nn
from transformers import BertModel

class Base_Model(nn.Module):
    """ A Model for bert pre-training """

    def __init__(self, bert_config='bert-base-uncased', n_class=None):
        super(Base_Model, self).__init__()
        self.bert_config = bert_config
        self.bert = BertModel.from_pretrained(self.bert_config)

        self.out = nn.Linear(768, n_class)

    def forward(self, ids, mask, token_type_ids):

        output = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids)

        return self.out(output['pooler_output'])
import torch
import transformers


class BertModel(torch.nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super(BertModel, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(dropout)
        self.l3 = torch.nn.Linear(768, n_classes)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)

        return output

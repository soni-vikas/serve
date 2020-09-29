import torch.nn as nn
import torch, torch.nn, os, sys, numpy as np
from sklearn import preprocessing
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from datetime import datetime
from transformers import BertTokenizer

r"""
The model is composed of the embeddingbag layer and the linear layer.

nn.EmbeddingBag computes the mean of 'bags' of embeddings. The text
entries here have different lengths. nn.EmbeddingBag requires no
padding because the lengths of sentences are saved in offsets.
Therefore, this method is much faster than the original one
with TorchText Iterator and Batch.

Additionally, since it accumulates the average across the embeddings on the fly,
nn.EmbeddingBag can enhance the performance and memory efficiency
to process a sequence of tensors.

"""

MAX_LEN = 128


class TextSentiment(nn.Module):
    softmax = nn.Softmax(dim=-1)

    def __init__(self, vocab_size=1308844, embed_dim=32, num_class=4):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        MODEL_DIR = "/models/intents"
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        print(device_name)
        self.device = torch.device(device_name)
        self.labelencoder = preprocessing.LabelEncoder()
        self.labelencoder.classes_ = np.load(os.path.join(MODEL_DIR, 'classes.npy'))
        config = BertConfig(os.path.join(MODEL_DIR, 'bert_config.json'))
        self.model = BertForSequenceClassification(config, num_labels=len(self.labelencoder.classes_))
        self.model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'pytorch_model.bin'), map_location="cpu"))
        self.model.to(self.device)
        self.model.eval()
        tokenizer_class, pretrained_weights = BertTokenizer, 'bert-base-uncased'
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.batch_size = 30
        self.dataloader_num_workers = 0

    def forward(self, requests):
        ids = []
        segment_ids = []
        input_masks = []

        print(requests)
        for sen in [requests]:
            text_tokens = self.tokenizer.tokenize(sen)
            tokens = ["[CLS]"] + text_tokens + ["[SEP]"]
            temp_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(temp_ids)
            segment_id = [0] * len(temp_ids)
            padding = [0] * (MAX_LEN - len(temp_ids))

            temp_ids += padding
            input_mask += padding
            segment_id += padding

            ids.append(temp_ids)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)

        ## Convert input list to Torch Tensors
        ids = torch.tensor(ids)
        segment_ids = torch.tensor(segment_ids)
        input_masks = torch.tensor(input_masks)
        validation_data = TensorDataset(ids, input_masks, segment_ids)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size,
                                           num_workers=self.dataloader_num_workers)

        responses = []
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

                for i in range(logits.size(0)):
                    label_idx = [self.__class__.softmax(logits[i]).detach().cpu().numpy().argmax()]
                    label_str = self.labelencoder.inverse_transform(label_idx)[0]
                    responses.append(label_str)

            _t1 = datetime.now()
        return responses[0]

# torch-model-archiver --model-name bert --version 1.0 --model-file ~/work/serve/examples/bert/models.py --serialized-file /models/intents/pytorch_model.bin --extra-files /models/intents/bert_config.bin --handler text
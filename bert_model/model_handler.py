from abc import ABC
from pytorch_pretrained_bert import BertForSequenceClassification, BertTokenizer, BertConfig
from sklearn import preprocessing
from ts.torch_handler.base_handler import BaseHandler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import torch, numpy as np, json, logging, os

logger = logging.getLogger(__name__)


class TransformersClassifierHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        properties = ctx.system_properties
        MODEL_DIR = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        self.labelencoder = preprocessing.LabelEncoder()
        self.labelencoder.classes_ = np.load(os.path.join(MODEL_DIR, 'classes.npy'))
        config = BertConfig(os.path.join(MODEL_DIR, 'bert_config.json'))
        self.model = BertForSequenceClassification(config, num_labels=len(self.labelencoder.classes_))
        self.model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'pytorch_model.bin'), map_location="cpu"))
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.softmax = torch.nn.Softmax(dim=-1)
        # self.batch_size = batch_size

        logger.debug('Transformer model from path {0} loaded successfully'.format(MODEL_DIR))
        self.manifest = ctx.manifest
        self.initialized = True

    def preprocess(self, data):
        ids = []
        segment_ids = []
        input_masks = []
        MAX_LEN = 128

        for sen in data:
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
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=len(data),
                                           num_workers=self.dataloader_num_workers)

        return validation_dataloader

    def inference(self, validation_dataloader):
        """
        Predict the class of a text using a trained transformer model.
        """
        # NOTE: This makes the assumption that your model expects text to be tokenized  
        # with "input_ids" and "token_type_ids" - which is true for some popular transformer models, e.g. bert.
        # If your transformer model expects different tokenization, adapt this code to suit 
        # its expected input format.
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
                    label_idx = [self.softmax(logits[i]).detach().cpu().numpy().argmax()]
                    label_str = self.labelencoder.inverse_transform(label_idx)[0]
                    responses.append(label_str)

        return responses

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output


_service = TransformersClassifierHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
# custom service file

# model_handler.py

"""
ModelHandler defines a base model handler.
"""
import logging

from transformers import *
import torch, torch.nn, os, sys, numpy as np
from sklearn import preprocessing
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from threading import current_thread
from multiprocessing import current_process
from datetime import datetime

MAX_LEN = 128

format = "%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s"

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter(format))

file_handler = logging.FileHandler("/tmp/logs-{}".format(current_process().pid), "w")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(format))



class Logger:
    loggers = {}

    @classmethod
    def get_logger(cls, name: str, handler,
                   level=logging.INFO,
                   format="%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s"
                   ) -> logging.Logger:
        """
        returns a logger with predefined formatting
        """
        # disable roor logger
        logging.getLogger().handlers = [logging.NullHandler()]

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        cls.loggers[name] = logger

        return logger


threads = set()


class ModelHandler(object):
    """
    A base Model handler implementation.
    """
    __handler__ = None

    STREAM_LOGGER = Logger.get_logger("\n\n-------------- model_handler ----------------", handler=stream_handler)
    FILE_LOGGER = Logger.get_logger("\nbenchmark", handler=file_handler)

    def __new__(cls, context):
        if not cls.__handler__:
            cls.__handler__ = super().__new__(cls)
            cls.__handler__.initialize(context)

        return cls.__handler__

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        # ModelHandler.LOGGER.critical("initializing model: %s - %s", context, type(context))
        try:
            MODEL_DIR = "/models/intents"

            # device
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(device_name)
            # label encoder
            labelencoder = preprocessing.LabelEncoder()
            labelencoder.classes_ = np.load(os.path.join(MODEL_DIR, 'classes.npy'))

            # model config
            config = BertConfig(os.path.join(MODEL_DIR, 'bert_config.json'))

            # model
            model = BertForSequenceClassification(config, num_labels=len(labelencoder.classes_))
            model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'pytorch_model.bin'), map_location="cpu"))
            model.to(device)
            model.eval()

            self.labelencoder = labelencoder
            self.model = model
            self.device = device
            self.model_batch_size = 32
            self.softmax = torch.nn.Softmax(dim=-1)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.batch_size = context.system_properties["batch_size"]
            ModelHandler.STREAM_LOGGER.critical("initialized: %s - %s - %s", context, device_name, context.system_properties)
        except Exception as e:
            ModelHandler.STREAM_LOGGER.critical("exeption in initialization: %s", str(e))

    def preprocess(self, requests):
        ids = []
        segment_ids = []
        input_masks = []
        # ModelHandler.LOGGER.critical("pre-processing: %s - ", requests)
        for sen in requests:
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

        # ModelHandler.LOGGER.critical("pre -- 1 : %s - %s", ids, input_masks)
        ## Convert input list to Torch Tensors
        ids = torch.tensor(ids)
        segment_ids = torch.tensor(segment_ids)
        input_masks = torch.tensor(input_masks)
        validation_data = TensorDataset(ids, input_masks, segment_ids)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.model_batch_size)
        # ModelHandler.LOGGER.critical("pre-processed: %s", len(validation_dataloader))
        return validation_dataloader

    def inference(self, dataloader):

        responses = []

        for batch in dataloader:
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

        # ModelHandler.LOGGER.critical("inferenced : %s", responses)
        return responses

    def postprocess(self, inference_output):
        """
        Return predict result in batch.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        return inference_output

    def handle(self, data, context):
        model_input = self.preprocess([i["body"][0] for i in data])
        model_out = self.inference(model_input)
        return self.postprocess(model_out)


def handle(data, context):
    try:
        ModelHandler.STREAM_LOGGER.critical("%s  -  %s  -  %s", data, context, current_thread().getName())
        _service = ModelHandler(context)
        # ModelHandler.LOGGER.critical("handling request: start")

        if data is None:
            return None

    except BaseException as e:
        ModelHandler.FILE_LOGGER.info("error in handling request %s", e)
        return "exception: {}".format(e)

    start = datetime.now()
    result = _service.handle(data, context)
    ModelHandler.FILE_LOGGER.info("%s %s %s %s", len([i["body"][0] for i in data]), datetime.now() - start, current_thread(), current_process().pid)
    return result


"""
docker run --gpus all --rm -it -p 8082:8080 -p 8081:8081 \
    -v ~/models:/home/model-server/models \
    -v ~/model_store/:/home/model-server/model-store/ \
    -v /tmp:/tmp \
    --entrypoint bash pytorch/torchserve:latest-gpu



torchserve --start --model-store ~/model-store --models ~/model-store/custom_bert.mar


register model ----------------------------------------------------------------------------------------------------------------------------------------

torch-model-archiver \
--model-name "custom_bert" \
--version 1.0 \
--serialized-file /models/intents/pytorch_model.bin  \
--extra-files "/models/intents/bert_config.json,/models/intents/classes.npy" \
--export-path ~/model_store \
--handler ~/work/serve/examples/bert/custom_model.py \
--runtime python3 \
--force && \
python -c '\
import requests ;\
batch_size="16"; worker="1";\
response = requests.delete(url="http://101.53.142.218:8081/models/custom_bert");print("1->", response.status_code, response.text) ;\
response = requests.post("http://101.53.142.218:8081/models?model_version=0.1.0&url=custom_bert.mar&model_name=custom_bert&batch_size="+batch_size+"&max_batch_delay=50&synchronous=true",);print("2->", response.status_code, response.text) ;\
response = requests.put("http://101.53.142.218:8081/models/custom_bert?min_worker="+worker+"&max_worker="+worker+"&name=custom_bert");print("3->", response.status_code, response.text) ;\
response = requests.get("http://101.53.142.218:8081/models/custom_bert");print("4->", response.status_code, response.text)'


python -c 'import requests; response = requests.put("http://101.53.142.218:8081/models/custom_bert?min_worker=10&max_worker=10&name=custom_bert");print("3->", response.status_code, response.text)'

curl -X DELETE http://101.53.142.218:8081/models/custom_bert
curl -X POST http://101.53.142.218:8081/models?url=custom_bert.mar&model_name=custom_bert&runtime=PYTHON&batch_size=1&max_batch_delay=50&synchronous=true
curl -X PUT http://101.53.142.218:8081/models/custom_bert?synchronous=true&min_worker=1&max_worker=1&name=custom_bert
curl -X GET http://101.53.142.218:8081/models/custom_bert


import requests, datetime
url = "http://101.53.142.218:8082/predictions/custom_bert"
payload = "[\n\t\"hello how are you i have an emergency and need to reschedule my current flight\", \n\t\"I want to cancel my flight.\", \n\t\"hello how are you i have an emergency and need to reschedule my current flight\"\n]"
headers = {'content-type': "application/json"}
start = datetime.datetime.now()
response = requests.request("POST", url, data=payload, headers=headers)
print(datetime.datetime.now()-start , response.text)


ab -c 10 -n 500 -p event.json -r -T application/json localhost:8000/api/books
"""

# torch-model-archiver \
# --model-name "custom_bert" \
# --version 1.0 \
# --serialized-file /models/intents/pytorch_model.bin  \
# --extra-files "/models/intents/bert_config.json,/models/intents/classes.npy" \
# --export-path ~/model_store \
# --handler ~/work/serve/examples/bert/custom_model.py \
# --runtime python3 \
# --force && curl -X DELETE \
#   http://101.53.142.218:8081/models/custom_bert \
#   -H 'cache-control: no-cache' \
#   -H 'postman-token: 3cfe505e-6d1e-1eff-c1f2-96d4fc2731cb' && curl -X POST \
#   http://101.53.142.218:8081/models \
#   -H 'cache-control: no-cache' \
#   -H 'content-type: application/json' \
#   -H 'postman-token: 4395195a-7e2c-2301-42d0-60607c1e1487' \
#   -d '{
#     "minWorkers": 1,
#     "maxWorkers": 1,
#     "name": "custom_bert",
#     "batchSize": 1,
#     "max_batch_delay": 50,
#     "url": "custom_bert.mar"
# }' && curl -X POST \
#   http://101.53.142.218:8081/models \
#   -H 'cache-control: no-cache' \
#   -H 'content-type: application/json' \
#   -H 'postman-token: 8b9a533c-7c9b-e5be-4464-d6bc264a0406' \
#   -d '{
#     "minWorkers": 1,
#     "maxWorkers": 1,
#     "name": "custom_bert",
#     "batchSize": 1,
#     "max_batch_delay": 50,
#     "url": "custom_bert.mar"
# }' && curl -X PUT \
#   http://101.53.142.218:8081/models/custom_bert \
#   -H 'cache-control: no-cache' \
#   -H 'content-type: application/json' \
#   -H 'postman-token: 8b9a533c-7c9b-e5be-4464-d6bc264a0406' \
#   -d '{
#     "minWorkers": 1,
#     "maxWorkers": 1,
#     "name": "custom_bert",
#     "batchSize": 1,
#     "max_batch_delay": 50,
#     "url": "custom_bert.mar"
# }' && curl -X GET \
#   http://101.53.142.218:8081/models/custom_bert \
#   -H 'cache-control: no-cache' \
#   -H 'postman-token: 0cb4c301-7e61-77cd-b46a-f1e24c9a819e'
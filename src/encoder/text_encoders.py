import os, logging, warnings, torch
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import GPT2Tokenizer, GPT2Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")




class BERT_Encoder:
    def __init__(self, return_cls=True):
        model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.return_cls = return_cls

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden]

        if self.return_cls:
            return last_hidden[:, 0, :].squeeze(0).cpu().numpy()  # 取 [CLS] 向量，返回 [hidden]
        else:
            return last_hidden.squeeze(0).cpu().numpy()  # 返回整个序列向量，形状 [seq_len, hidden]
    
    def encode_batch(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]

        if self.return_cls:
            return last_hidden[:, 0, :].cpu().numpy()  # [batch, hidden]
        else:
            return last_hidden.cpu().numpy()  # [batch, seq_len, hidden]
        

class RoBERTa_Encoder:
    def __init__(self, return_cls=True):
        model_name = 'roberta-base'
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.return_cls = return_cls

    def encode(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state
        if self.return_cls:
            return last_hidden[:, 0, :].squeeze(0).cpu().numpy()  # [CLS] 位置向量
        else:
            return last_hidden.squeeze(0).cpu().numpy()  # 所有 token 的向量

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(texts, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state
        if self.return_cls:
            return last_hidden[:, 0, :].cpu().numpy()  # 批量的 CLS 向量
        else:
            return last_hidden.cpu().numpy()  # 批量所有 token 向量
        
class GPT2_Encoder:
    def __init__(self):
        model_name = 'gpt2'
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, -1, :].squeeze(0).cpu().numpy()  # 取最后一个 token 的向量

    def encode_batch(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, -1, :].cpu().numpy()  # 每个句子的最后一个 token 向量


if __name__ == "__main__":
    encoder = BERT_Encoder()
    vec = encoder.encode("Hello world!")
    print("Single vector shape:", vec.shape)  # torch.Size([768])
    batch_vecs = encoder.encode_batch(["Hello world!", "How are you?"])
    print("Batch vectors shape:", batch_vecs.shape)  # torch.Size([2, 768])

    encoder = RoBERTa_Encoder()
    vec = encoder.encode("Hello world!")
    print("Single vector shape:", vec.shape)  # torch.Size([768])
    batch_vecs = encoder.encode_batch(["Hello world!", "How are you?"])
    print("Batch vectors shape:", batch_vecs.shape)  # torch.Size([2, 768])

    encoder = GPT2_Encoder()
    vec = encoder.encode("Hello world!")
    print("Single vector shape:", vec.shape)  # torch.Size([768])
    batch_vecs = encoder.encode_batch(["Hello world!", "How are you?"])
    print("Batch vectors shape:", batch_vecs.shape)  # torch.Size([2, 768])


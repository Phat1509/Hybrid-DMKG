import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import time
import json
import pickle
import spacy



class MultimodalRetriever:
    def __init__(self, model_name='nvidia/MM-Embed', device='cuda', half=True):
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype='auto'
        )
        if half:
            self.model = self.model.half()
        if device == 'cuda':
            self.model = self.model.cuda()
        self.max_length = 2048
        self.image_instruction = "Retrieve an image with the name given in the query about the image."
        self.text_instruction = "Given a question, retrieve passages that answer the question"
    def encode_queries(self, queries,have_image=True):
        if have_image:
            query_embeddings = self.model.encode(
                queries, is_query=True, instruction=self.image_instruction, max_length=self.max_length
            )['hidden_states']
        else:
            query_embeddings = self.model.encode(
                queries, is_query=True,instruction=self.text_instruction,  max_length=self.max_length
            )['hidden_states']
        return query_embeddings
    

    def encode_passages(self, passages, batch_size=10):
        all_embeddings = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch, max_length=self.max_length
            )['hidden_states']
            all_embeddings.append(batch_embeddings)
        # 拼接所有 batch 的 embedding
        if isinstance(all_embeddings[0], torch.Tensor):
            passage_embeddings = torch.cat(all_embeddings, dim=0)
        else:
            import numpy as np
            passage_embeddings = np.concatenate(all_embeddings, axis=0)
        return passage_embeddings

    def compute_scores(self, query_embeddings, passage_embeddings, topK=1):
        # 计算相关性分数
        if type(passage_embeddings) is not torch.Tensor:
            passage_embeddings = torch.tensor(passage_embeddings, device=query_embeddings.device)
        scores = (query_embeddings @ passage_embeddings.T)
        scores = F.softmax(scores, dim=1)  # 对每个query的所有passage归一化
        max_k = min(topK, scores.shape[1])
        topk_scores, topk_indices = torch.topk(scores, k=max_k, dim=1)
        return topk_scores, topk_indices
    


def main():
    with open('../datasets/edited_Multimodal_triplets.json', 'r') as f:
        data = json.load(f)

    with open('../datasets/wiki_id_map_new.pkl', 'rb') as f:
        wiki_id_map = pickle.load(f)

    def get_wikidata_api_id(name):
        return wiki_id_map.get(name, name)

    all_data = []
    for key, value in data.items():
        image_id = value.get('image_id')
        p31_triples_list = []
        p279_triples_list = []

        for triple_str in value.get('triples', []):
            parts = triple_str.split('\t')
            if len(parts) >= 3 and parts[1].lower() == 'p31':
                subj, pred, obj = parts[:3]
                head = get_wikidata_api_id(subj)
                relation = get_wikidata_api_id(pred)
                tail = get_wikidata_api_id(obj)
                p31_triples_list.append(f"{head} is {relation} a {tail}; ")
            elif len(parts) >= 3 and parts[1].lower() == 'p279':
                subj, pred, obj = parts[:3]
                head = get_wikidata_api_id(subj)
                relation = get_wikidata_api_id(pred)
                tail = get_wikidata_api_id(obj)
                p279_triples_list.append(f"{head} is {relation} a {tail}; ")

        if p31_triples_list:
            p31_triples = ''.join(p31_triples_list)
        elif p279_triples_list:
            p31_triples = ''.join(p279_triples_list)
        else:
            p31_triples = f"{get_wikidata_api_id(key)} don't know it type;"

        all_data.append({key: {
            'image_id': image_id,
            'p31_triples': p31_triples
        }})

    Mretriever = MultimodalRetriever()

    passages = []
    keys = []
    for item in all_data:
        for key, value in item.items():
            image_path = f"../datasets/VLKEB_images/mmkb_images/{value['image_id']}"
            with Image.open(image_path) as img:
                passages.append({
                    'txt': value['p31_triples'],
                    'img': img.copy()
                })
            keys.append(key)

    print(len(passages), len(keys))
    batch_size = 20
    all_embeddings = []

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Start time: {start_time}")
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i+batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(passages) + batch_size - 1) // batch_size}")
        batch_embeddings = Mretriever.encode_passages(batch)
        all_embeddings.append(batch_embeddings.cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"End time: {start_time}")

    embedding_dict = {k: emb for k, emb in zip(keys, embeddings)}

    with open('../datasets/all_triple_embeddings.pkl', 'wb') as f:
        pickle.dump(embedding_dict, f)

if __name__ == "__main__":
    main()



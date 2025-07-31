import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from PIL import Image
import time
import json
import pickle
import spacy

# 新增BLIP依赖
# from transformers import BlipProcessor, BlipModel, BlipTextModel

class MultimodalRetriever:
    def __init__(self, model_name='openai/clip-vit-large-patch14', device='cuda', half=True):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        if half and self.device == 'cuda':
            self.model = self.model.half()

    def encode_queries(self, queries,information_type="image", have_image=True):
        """
        queries: 图片路径或PIL对象列表
        texts: 文本列表（与图片一一对应）
        have_image: 是否有图片
        """
        if have_image:
            images = queries[0].get('img')
            texts = queries[0].get('txt')
            inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                if information_type == 'image':
                    multimodal_embeds = self.model.get_image_features(pixel_values=inputs['pixel_values'])
                elif information_type == 'image+text':
                    multimodal_embeds = self.model.get_image_features(pixel_values=inputs['pixel_values'])
                    text_embeds = self.model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                    multimodal_embeds = (multimodal_embeds + text_embeds) / 2
                else:
                    raise ValueError("Unsupported information type. Use 'image' or 'image+text'.")
            return multimodal_embeds

        else:
            texts = queries[0].get('txt')
            if isinstance(texts, str):
                texts = [texts]
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                print(outputs.shape)
                # text_embeds = outputs.last_hidden_state.mean(dim=1)  # 取平均池化作为句子表示
            return outputs
        

    def encode_passages(self, passages,information_type="image+text", batch_size=10,have_image=True):
        
        # print(passages)
        if have_image:
            all_embeddings = []
            for i in range(0, len(passages), batch_size):
                batch = passages[i:i+batch_size]
                texts = [item['txt'] for item in batch]
                images = [item['img'] for item in batch]
                inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    if information_type == 'image':
                        # 只取 pixel_values 传给 get_image_features
                        outputs = self.model.get_image_features(pixel_values=inputs['pixel_values'])
                    elif information_type == 'image+text':
                    # 只取 pixel_values 传给 get_image_features
                        outputs = self.model.get_image_features(pixel_values=inputs['pixel_values'])
                        text_embeds = self.model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                        outputs = (outputs + text_embeds) / 2  # 平均池化图像和文本特征   
                all_embeddings.append(outputs.cpu())
            passage_embeddings = torch.cat(all_embeddings, dim=0).to(device=self.device)
        else:
            all_embeddings = []
            for i in range(0, len(passages), batch_size):
                batch = passages[i:i+batch_size]
                texts = [item['txt'] for item in batch]
                inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    text_embeds = self.model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                all_embeddings.append(text_embeds.cpu())
            passage_embeddings = torch.cat(all_embeddings, dim=0).to(device=self.device)
            
        return passage_embeddings


    def compute_scores(self, query_embeddings, passage_embeddings, topK=1):
        # 保证设备一致

        if isinstance(passage_embeddings, list):
            passage_embeddings = torch.tensor(passage_embeddings, device=self.device)
        # exit()
        # print(f"Query device: {query_embeddings.device}, Passage device: {passage_embeddings.device}")
        # passage_embeddings = passage_embeddings.to(query_embeddings.device)
        scores = (query_embeddings @ passage_embeddings.T)
        scores = F.softmax(scores, dim=1)
        max_k = min(topK, scores.shape[1])
        topk_scores, topk_indices = torch.topk(scores, k=max_k, dim=1)
        return topk_scores, topk_indices

# main函数不变，只需初始化时传use_blip参数即可
def main(information_type = 'image+text'):
    with open('../datasets/edited_Multimodal_triplets.json', 'r') as f:
        data = json.load(f)

    with open('../datasets/wiki_id_map_new.pkl', 'rb') as f:
        wiki_id_map = pickle.load(f)

    def get_wikidata_api_id(name):
        return wiki_id_map.get(name, name)
      # 或 'image+text'
    # information_type = 'text'  # 如果没有图片，则使用文本
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
            'p31_triples': get_wikidata_api_id(key)
        }})

    Mretriever = MultimodalRetriever()  # 或 use_blip=False

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

    # print(len(passages), len(keys))
    batch_size = 20
    all_embeddings = []

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Start time: {start_time}")
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i+batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(passages) + batch_size - 1) // batch_size}")
        batch_embeddings = Mretriever.encode_passages(batch,information_type,have_image=True)  # 或 have_image=False
        all_embeddings.append(batch_embeddings.cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"End time: {start_time}")

    embedding_dict = {k: emb for k, emb in zip(keys, embeddings)}
    print(f"Total embeddings: {len(embedding_dict)}")
    with open('../datasets/all_triple_embeddings_clip2_image+text.pkl', 'wb') as f:
        pickle.dump(embedding_dict, f)

if __name__ == "__main__":
    main(information_type = 'image+text')



import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import spacy
import random
from sentence_transformers import SentenceTransformer, util
from transformers import set_seed
import re
import openai
from transformers import DistilBertTokenizer, DistilBertForTokenClassification,DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm
import collections
import argparse
import torch
import requests

class RelationExtractor:
    def __init__(self, relation_model_path, spacy_model_path="../en_core_web_md", device=None):
        self.nlp = spacy.load(spacy_model_path)
        self.nlp.add_pipe("entityLinker", last=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.relation_model = DistilBertForTokenClassification.from_pretrained(relation_model_path)
        self.relation_tokenizer = DistilBertTokenizerFast.from_pretrained(relation_model_path)
        self.relation_model.to(self.device)
        # self.relation_model.half()  # 加载为半精度

    def extract_relations_from_model_output(self, text):
        relations = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
        for token in text_replaced.split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    relations.append({
                        'head': subject.strip(),
                        'type': relation.strip(),
                        'tail': object_.strip()
                    })
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    relations.append({
                        'head': subject.strip(),
                        'type': relation.strip(),
                        'tail': object_.strip()
                    })
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '':
            relations.append({
                'head': subject.strip(),
                'type': relation.strip(),
                'tail': object_.strip()
            })
        return relations

    def link_entity(self, entity):
        def get_wikidata_api_id(entity):
            pattern = r'Q\d+'
            try:
                linking = re.search(pattern, str(self.nlp(entity.capitalize())._.linkedEntities))
            except:
                linking = re.search(pattern, str(self.nlp(entity)._.linkedEntities))
            if linking:
                linking = linking.group(0)
            else:
                linking = None
            return linking

        wid = get_wikidata_api_id(entity)
        if not wid and "(" in entity and ")" in entity:
            entity_simple = re.sub(r"\s*\(.*?\)", "", entity).strip()
            if entity_simple and entity_simple != entity:
                wid = get_wikidata_api_id(entity_simple)
        return wid

    def realtion_predict(self, question):
        relation_model = self.relation_model.eval()
        text = question
        tokens = text.split()
        inputs = self.relation_tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, padding='max_length', max_length=500)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 不要 .half()
        with torch.no_grad():
            outputs = relation_model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
            label_list = ["O", "REL"]
            id2label = {i: l for i, l in enumerate(label_list)}
            rel_tokens = []
            for token, pred in zip(tokens, predictions[1:len(tokens)+1]):
                label = id2label.get(pred, "O")
                if label == "REL":
                    rel_tokens.append(token)
            rel_phrase = " ".join(rel_tokens)
            return rel_phrase

    def retrieve_from_extraction(self, q, wiki_id, entity, graph_triples):
        q = re.sub(r'(\w)(\?)$', r'\1 ?', q)
        # print(f"Processing question: {q}")
        relation_important_word = self.realtion_predict(q)
        # print(f"Extracted relation phrase: {relation_important_word}")
        rel_phrase = relation_important_word.lower()
        # print(f"Extracted relation phrase: {rel_phrase}")
        max_sim = 0
        best_key = None
        rel_doc = self.nlp(rel_phrase)
        for key in graph_triples.keys():
            key_phrase = key.lower()
            key_doc = self.nlp(key_phrase)
            if rel_doc.vector_norm and key_doc.vector_norm:
                sim = rel_doc.similarity(key_doc)
                # print(f"Comparing '{rel_phrase}' with '{key_phrase}': similarity = {sim}")
            else:
                sim = 0
            if sim > max_sim:
                max_sim = sim
                best_key = key
        if max_sim < 0.4:
            return None, None
            
        best_value = graph_triples.get(best_key, None)
        return best_key, best_value

# 用法示例
if __name__ == "__main__":
    extractor = RelationExtractor(
        relation_model_path='../train/results_relation/checkpoint-4140',
        spacy_model_path="../en_core_web_md"
    )
    # 示例调用
    # key, value = extractor.retrieve_from_extraction(q, wiki_id, entity, relation_list)

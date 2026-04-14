from typing import List, Literal, Optional
from urllib.error import HTTPError
from sparql_util import get_neighbor_triples,get_wikidata_id
import json
import time


def get_id_from_uri(uri: str) -> str:
    """Return the ID from a Wikidata URI."""
    return uri.split('/')[-1]

def init_triples_dict(subject, relation, obj) -> dict:
    """初始化一个三元组字典。"""
    return {
        "subject": subject,
        "relation": relation,
        "target": obj
    }


# Initialize the keys in the new json.
def initialize_json(index) -> dict:
    """Initialize the JSON object for the new triples."""
    return {
        "case_id": index,
        "triples": []
    }


# Get all the neighbor triples of an item as outgoing edges.
def get_entity_triplets(entity_label: str) -> list:
    """获取实体及其邻居的三元组。"""
    entity_id_to_label = {}
    triples = []
    print(entity_label)
    print("-----------------"*30)
    entity_id = get_wikidata_id(entity_label)
    entity_id_to_label[entity_id] = entity_label


    # 调用工具函数获取邻居三元组
    results = get_neighbor_triples(entity_label)
    if results == None:
        print(entity_label)
        triple = []
    else:
        
        for result in results["results"]["bindings"]:
            type = result['object']['type']
            if not type == "literal":
                continue
            subject = get_id_from_uri(result['subject']['value'])
            entity_id_to_label[subject] = result['object']['value']

        # 处理关系并转换为 JSON 格式
     
        for result in results["results"]["bindings"]:
            type = result['object']['type']
            if not type == "uri":
                continue
        

            subject = get_id_from_uri(result['subject']['value'])
            predicate = get_id_from_uri(result['predicate']['value'])
            obj = get_id_from_uri(result['object']['value'])
      
            
            triple = init_triples_dict(
                entity_id_to_label[subject],
                entity_id_to_label[predicate],
                entity_id_to_label[obj]
            )
            print(triple)
    
            triples.append(triple)

    return triples


def get_triplets_from_dataset(dataset_path):
    """从数据集中提取三元组并保存为 JSON 文件。"""
    with open(dataset_path, 'r') as file:
        dataset = json.load(file)

    counter = 0
    relation_id_dict = {}
    new_json = []

    for index,case in enumerate(dataset):

        new_data = initialize_json(index)

        rewrite = case

        subject_label = rewrite["image"]
        target_label = rewrite["alt"]
        relation_label = "have image of"

        origin_triplet = init_triples_dict(
            subject_label, relation_label, target_label)
        print("origin_triplet: ", origin_triplet)
        
        target_neighbors = get_entity_triplets(
            target_label)


        new_data["triples"].extend([origin_triplet])

        new_data["triples"].extend(target_neighbors)
    
        counter += 1
        new_json.extend([new_data])
    

        if counter % 3 == 0:
            print(new_json)
            file_name = f'eval_multihop_graph.json'
            with open(file_name, 'w') as output_file:
                json.dump(new_json, output_file,ensure_ascii=False, indent=2)
     
            
            
            # new_json = []

    # if counter % 1000 != 0:
    #     file_name = f'eval_multihop_graph_{counter - (counter % 1000)}_to_{counter - 1}.json'
    #     with open(file_name, 'w') as output_file:
    #         json.dump(new_json, output_file)
    #     new_json = []


if __name__ == "__main__":
    get_triplets_from_dataset(
        "../archive/eval_multihop.json")

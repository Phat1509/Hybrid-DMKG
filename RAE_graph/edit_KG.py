from tqdm import tqdm
import pickle
import json
import requests
from requests.exceptions import RequestException
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def load_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def find_lines_by_entity(triplets_dict, entity_number):
    # Search for lines containing the given entity number
    print(triplets_dict)
    if entity_number in triplets_dict:
        return triplets_dict[entity_number]
    return []

def save_triplets_dict(triplets_dict, filename):
    with open(filename, 'wb') as file:
        pickle.dump(triplets_dict, file)

def save_triplets_json(triplets_dict, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(triplets_dict, file, ensure_ascii=False, indent=2)

def load_triplets_dict(filename):
    with open(filename, 'r') as file:
        triplets_dict = json.load(file)
    return triplets_dict
def get_wikidata_api_id(label, language='en'):
    """
    用 Wikidata API 查询实体(Q)或属性(P)的编号
    """
    import re
    url = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': language,
        'search': label
    }

    # 先查实体（item）
    
    # 去掉 label 中的下划线
    label = label.replace('_', ' ')
    
    params['type'] = 'item'
    for _ in range(3):
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            if data.get('search'):
                return data['search'][0]['id']
        except RequestException as e:
            111
            # print(f"[Warning] Network error in get_wikidata_api_id (item): {e}")
            # return None

    # 如果有括号且括号内有内容，去除括号及内容后再次查
    if '(' in label and ')' in label:
        # 用正则去除括号及内容
        label_no_bracket = re.sub(r'\s*\(.*?\)\s*', '', label).strip()
        if label_no_bracket and label_no_bracket != label:
            params['search'] = label_no_bracket
            for _ in range(3):
                try:
                    response = requests.get(url, params=params, timeout=10)
                    data = response.json()
                    if data.get('search'):
                        return data['search'][0]['id']
                except RequestException as e:
                    print(f"[Warning] Network error in get_wikidata_api_id (item, no bracket): {e}")
                    return None

def get_entity_label(entity_id, language='en'):
    """
    查询 Wikidata 实体或属性的 label
    """
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    try:
        response = requests.get(url, timeout=10, verify=False)
        data = response.json()
        entities = data.get('entities', {})
        if entity_id in entities:
            labels = entities[entity_id].get('labels', {})
            if language in labels:
                return labels[language]['value']
            # fallback
            for v in labels.values():
                return v['value']
    except Exception as e:
        1111
    return entity_id  # fallback

def get_entity_triples(entity_id,wiki_id_map):
    """
    查询 Wikidata 某实体的所有关系和目标实体，返回三元组字符串列表和 id2label 字典
    例如: (["Q11081\tP172\tQ52997", ...], {"Q11081": "xxx", "P172": "yyy", "Q52997": "zzz"})
    """
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    # try:
    response = requests.get(url, timeout=1)
    data = response.json()
    triples = []
    entities = data.get('entities', {})
    if entity_id in entities:
        claims = entities[entity_id].get('claims', {})
        for prop, claim_list in claims.items():
            # 查询属性 label
            if prop not in wiki_id_map:
                wiki_id_map[prop] = get_entity_label(prop)
            for claim in claim_list:
                mainsnak = claim.get('mainsnak', {})
                if mainsnak.get('snaktype') == 'value':
                    datavalue = mainsnak.get('datavalue', {})
                    value = datavalue.get('value', {})
                    # 只处理目标为实体的情况
                    if isinstance(value, dict) and 'id' in value:
                        triples.append(f"{entity_id}\t{prop}\t{value['id']}")
                        # 查询目标实体 label
                        if value['id'] not in wiki_id_map:
                            wiki_id_map[value['id']] = get_entity_label(value['id'])
    return triples, wiki_id_map
    # except RequestException as e:
    #     print(f"[Warning] Network error in get_entity_triples: {e}")
    #     return [], {}





def main():
   

    
    # counterfact_path = '../archive/convert_eval_multihop_with_re_questions_with_wiki_new.json'
    counterfact_path = './convert_eval_multihop_with_re_questions_with_wiki_new.json'
    

    # dict_filename = '../mmkb-master/final_graph.json'  # Replace this with the desired path for saving the dictionary
    dict_filename = './final_graph.json'  # Replace this with the desired path for saving the dictionary
    
    
    new_dict_filename = './Multimodal_triplets_new.json' 


    # 读取 mmkb-master 下的 wiki_id_mapping.txt，生成字典
    wiki_id_mapping_path = './wiki_id_mapping.txt'
    # wiki_id_mapping_path = '../mmkb-master/wiki_id_mapping.txt'

    wiki_id_map = {}
    
    with open(wiki_id_mapping_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                key, value = parts
                wiki_id_map[key] = value
    print(len(wiki_id_map))




    print("Loading triplets from file:", dict_filename)
    
    
    
    # Step 1: Load triplets from the file into a dictionary
    triplets_dict = load_triplets_dict(dict_filename)
    
    lines = load_dataset(counterfact_path)

    print(len(lines))
    for index,line in enumerate(lines):
        # print("Processing line:", line)
        print(f"Processing line {index}")

        pred_name = line["pred"]
        alt_name = line["alt"]
        image_id = line["image"]
        pred_name_id = get_wikidata_api_id(pred_name)
      
        print(f"pred_name: {pred_name}")
        print(f"pred_name_id: {pred_name_id}")
    
        wiki_id_map[pred_name_id] = pred_name

        pred_alt_id = get_wikidata_api_id(alt_name)
        wiki_id_map[pred_alt_id] = alt_name
      
        print(f"alt_name: {alt_name}")
        print(f"pred_alt_id: {pred_alt_id}")
        print("111111"*30)

 

        # 处理 pred_name_id
        if pred_name_id is not None:
            # 修改 image 字段为 None 字符串
            if pred_name_id in triplets_dict:
                triplets_dict[pred_name_id]["image_id"] = "none.jpg"

        if  pred_alt_id is not None:
            if pred_alt_id in triplets_dict:
                triplets_dict[pred_alt_id]["image_id"] = image_id
            else:
                # 不存在则新建，并补充 triples
                triples, wiki_id_map = get_entity_triples(pred_alt_id,wiki_id_map)
                new_lines_containing_entity = {"image_id": image_id, "triples": triples}
                
                triplets_dict[pred_alt_id] = new_lines_containing_entity


        else:
            new_lines_containing_entity= {"image_id":image_id,"triples":[]}
            triplets_dict[alt_name] = new_lines_containing_entity
    save_triplets_json(triplets_dict, new_dict_filename)
    print(len(wiki_id_map))
    # 保存更新后的字典
       # 保存 wiki_id_map 到 pickle 文件
    with open("wiki_id_map_new.pkl", "wb") as f:
        pickle.dump(wiki_id_map, f)
    
    

                    
if __name__ == "__main__":
    main()

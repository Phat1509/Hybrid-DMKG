import pickle
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import spacy
import logging
import re
from transformers import set_seed
import openai
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm
from utils_func import  *
from extraction_knowlege import RelationExtractor
from PIL import Image
from multimodal_retriever_clip import MultimodalRetriever
from Main_Models import ModelManager
import random

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)

global nlp
nlp = spacy.load("../en_core_web_md") # entity linking model
nlp.add_pipe("entityLinker", last=True)
import time
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
import os

# ENTITY = 0
# REL = 1

global local_modal
local_modal = None


openai.api_key = "sk-oHf1AIjc3m5jl593F9Fc3d23Dd314031B83d1c22A5Bc429b"
openai.base_url = "https://api.gpt.ge/v1/"
openai.default_headers = {"x-foo": "true"}


global image_path
# image_path = f"/home/NCUT/teacher/xmy/MRAG/PMI/datasets/VLKEB_images/mmkb_images/"
image_path = f"/home/lyuan/MQA/PMI/datasets/VLKEB_images/mmkb_images/"


def link_entity(entity, nlp):
    def get_wikidata_api_id(entity, nlp):
        pattern = r'Q\d+'
        try:
            doc = nlp(entity.capitalize())
            linked_entities = doc._.linkedEntities
            # 只取第一个实体的kb_id（QID）
            if linked_entities and len(linked_entities) > 0:
                kb_id = linked_entities[0].kb_id_
                if re.match(pattern, kb_id):
                    return kb_id
            # 如果没有，回退到原始字符串查找
            linking = re.search(pattern, str(linked_entities))
            if linking:
                return linking.group(0)
            else:
                return entity
        except Exception as e:
            print(f"Entity linking error: {e}")
            return entity


    wid = get_wikidata_api_id(entity,nlp)


    if not wid and "(" in entity and ")" in entity:
        # 去掉括号及内容
        entity_simple = re.sub(r"\s*\(.*?\)", "", entity).strip()
        if entity_simple and entity_simple != entity:
            wid = get_wikidata_api_id(entity_simple,nlp)
    
     
    return wid


def run_api_llm(query,api_name):
    if api_name.startswith("api-"):
        model = api_name[4:]
    messages = [{"role": "user", "content": query}]

    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    result = response.choices[0].message.content
    return result





def run_llm_answer(query, local_modal, image_id=None):
    if type(local_modal) is str:
        result = run_api_llm(query, local_modal)
    else:
        if image_id is not None:
            img_path = image_path + "/" + image_id
            result = local_modal.get_out_put(query, image_path=img_path)
        else:
            result = local_modal.get_out_put(query)
    return result


def run_llm_divide(query, local_modal, image_id=None):
    if type(local_modal) is str:
        # 如果没有本地模型，就使用API模型
        result = run_api_llm(query, local_modal)
    else:
        if image_id is not None:
            img_path = image_path + "/" + image_id  # 用局部变量
            result = local_modal.get_out_put(query, image_path=img_path)
        else:
            result = local_modal.get_out_put(query)
    sub_questions = result.split('\n')
    sub_questions = [sq for sq in sub_questions if "[IMAGE]" in sq or "[ENT]" in sq]
    return sub_questions




def run_llm_choice(query, local_modal, image_id):
    if type(local_modal) is str:
        result = run_api_llm(query, local_modal)
    else:
        if image_id is not None:
            img_path = image_path + "/" + image_id
            result = local_modal.get_out_put(query, image_path=img_path)
        else:
            result = local_modal.get_out_put(query)
    return result


def verify_gold_path(entities,  path):
    if len(entities) != len(path):
        return False
    for i in range(len(path)):
            if entities[i] is None or entities[i].lower().strip() == "" or len(entities[i].lower().strip())<=3 or not any(c.isalpha() for c in entities[i].lower()):
                return False
            if any(entities[i].lower().replace("_", " ") in  p.lower().replace("_", " ") for p in path[i]) or any(entities[i].lower().replace("_", " ") in p.lower() for p in path[i]):
                continue
            else:
                return False
    return True



def input_constrcution(question, image_id):
    if image_id is not None:
        image_path = f"../datasets/VLKEB_images/mmkb_images/{image_id}"
        with Image.open(image_path) as img:
            input_query = {
                'txt': question,
                'img': img.copy()
            }
    else:
        input_query = {
            'txt': question
        }

    return input_query

class MultimodalGraph:
    def __init__(self, graph_path,qid2name):
        with open(graph_path, "r") as f:
            self.graph = json.load(f)
        self.qid2name = qid2name
        # print(self.qid2name)
        # exit()
        

    def get_triples_by_id(self, wiki_id):
        """
        Given a wiki_id, return all triple descriptions for that entity.
        """
        # print(wiki_id)
        triples = self.graph.get(wiki_id, {}).get("triples", [])
        # print(f"Foun {triples}")
        triple_descs = []
        relation_tails = {}
        triples_for_LLMs = []
        for triple in triples:
            triple = triple.split('\t')
            # 如果 triple[2] 完全由数字组成（不是字母），则跳过该 triple
            # 如果 triple[2] 完全由数字组成（可能是日期等），则跳过该 triple
            if re.fullmatch(r"[0-9\.\-]+", triple[2]):
                continue
            # triple is expected to be a tuple/list: (head, relation, tail)
            head = self.qid2name.get(triple[0], triple[0])
            rel = self.qid2name.get(triple[1], triple[1])
            tail = self.qid2name.get(triple[2], triple[2])
            
            triple_descs.append(f"{head} have the {rel} with {tail}")
            relation_tails[self.qid2name.get(triple[1], triple[1])]=self.qid2name.get(triple[2], triple[2])
            triples_for_LLMs.append(f"{head}--{rel}--{tail}")
        return triple_descs,relation_tails, triples_for_LLMs

# 初始化全局 multimodal_graph 实例

def constrcution_path_answer(hop):
    path_answer_list = []
    # 处理第一个 triple，答案在 post 字段
    triple1 = hop.get("triple1", {})
    if "post" in triple1:
        path_answer_list.append([triple1["post"]])

    # 处理后续 triple，答案在 answer + answer_ali
    i = 2
    while True:
        triple_key = f"triple{i}"
        triple = hop.get(triple_key)
        if not triple:
            break
        # 答案为 answer + answer_ali
        answer_list = [triple.get("answer", "")]
        answer_list.extend(triple.get("answer_ali", []))
        path_answer_list.append(answer_list)
        i += 1
    answer_last = path_answer_list[-1]

    return path_answer_list, answer_last


def choose_entity_with_llm(subq, rag_entity, extract_entity, rag_wiki_id, extract_wiki_id, multimodal_graph, choose_base_prompt, local_modal,image_id=None, max_try=3):
    """
    使用大模型多次判断选择最终实体，最多尝试 max_try 次。
    如果都没有匹配上，则返回 extract_entity。
    """
    for _ in range(max_try):
        # 获取三元组
        _, _, rag_result_for_LLMs = multimodal_graph.get_triples_by_id(rag_wiki_id)
        _, _, extract_result_for_LLMs = multimodal_graph.get_triples_by_id(extract_wiki_id)

        # 如果三元组数量大于10，随机采样5条
        if len(rag_result_for_LLMs) > 10:
            rag_result_for_LLMs = random.sample(rag_result_for_LLMs, 5)
        if len(extract_result_for_LLMs) > 10:
            extract_result_for_LLMs = random.sample(extract_result_for_LLMs, 5)

        # 随机决定 A/B 的内容
        if random.random() < 0.5:
            answer_a, support_a = rag_entity, rag_result_for_LLMs
            answer_b, support_b = extract_entity, extract_result_for_LLMs
        else:
            answer_a, support_a = extract_entity, extract_result_for_LLMs
            answer_b, support_b = rag_entity, rag_result_for_LLMs

        choose_prompt = choose_base_prompt.replace("<<<QUESTION>>>", subq)
        choose_prompt = choose_prompt.replace("<<<Answer A>>>", answer_a)
        choose_prompt = choose_prompt.replace("<<<Knowledge supporting A>>>", "; ".join(support_a))
        choose_prompt = choose_prompt.replace("<<<Answer B>>>", answer_b)
        choose_prompt = choose_prompt.replace("<<<Knowledge supporting B>>>", "; ".join(support_b))
        # print("choose_prompt: " + choose_prompt)
        choose_result = run_llm_choice(choose_prompt, local_modal,image_id=image_id)
        # print("choose_result: " + choose_result)
        # print("------------this is choose result end------------")
        # # 匹配 A 或 B 的答案
        
        choose_match = re.search(r'(?:Answer|Output):?\s*([AB])\b', choose_result)
        # print("choose_match: " + str(choose_match))
        if choose_match:
            ab = choose_match.group(1)
            print("choose ab: " + ab)
            return answer_a if ab == "A" else answer_b
        # 直接是A或B的情况
        if choose_result.strip() == "A":
            return answer_a
        if choose_result.strip() == "B":
            return answer_b
    # 3次都没有匹配上
    return extract_entity




   

def main():

    
    args = parse_args()
    seed = args.seed
    set_seed(seed)
    model_name = args.model_name
    
    image_type = int(args.image_type)
    port = args.port


    divide_modal = args.divide_modal
    # main_model = ModelManager(model_name)
    
    Mretriever = MultimodalRetriever(device=device)
    if model_name.startswith("api"):
        local_modal = model_name
    else:
        local_modal = ModelManager(model_name, half=False,port=port)
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    question2subqs_path = f"question2subqs_{divide_modal}.json"
    if os.path.exists(question2subqs_path):
        with open(question2subqs_path, "r", encoding="utf-8") as f:
            question2subqs = json.load(f)
        print(f"Loaded question2subqs from {question2subqs_path}")
    else:
        x = {}
    print(f"question2subqs length: {len(question2subqs)}")
    

    logger.info("Finished loading model")
    logger.info(f"Model config: {args}")

    with open('../prompts/answer.txt', 'r') as p:
        answer_base_prompt = p.read()

    with open('../prompts/divide.txt','r') as f:
        divide_base_prompt = f.read()

    with open('../prompts/choose.txt','r') as f:
        choose_base_prompt = f.read()


    entity_cache = {}
    print("args.information_type",args.information_type)

    entity_cache_path = f"entity_cache_CLIP2_{args.information_type}_{divide_modal}_{str(image_type)}.json"
    print(f"entity_cache_path: {entity_cache_path}")

    if os.path.exists(entity_cache_path):
        with open(entity_cache_path, "r", encoding="utf-8") as f:
            entity_cache = json.load(f)
    # print(entity_cache)
    print(f"entity_cache length: {len(entity_cache)}")
    # exit()

    cor = tot = 0
    ver_cor = 0
    cor_list = [0,0,0,0]
    ver_cor_list = [0,0,0,0]
    tot_list = [0,0,0,0]
    #衡量image的检索准确率
    image_retriever_correct = [0,0,0,0]

    # random.shuffle(data)

    with open("../datasets/wiki_id_map_new_inverted.pkl", "rb") as f:
        qname2id = pickle.load(f)
    

    qname2id = {k.lower(): v for k, v in qname2id.items()}

    with open("../datasets/wiki_id_map_new.pkl", "rb") as f:
            qid2name = pickle.load(f)



    multimodal_graph = MultimodalGraph("../datasets/edited_Multimodal_triplets.json",qid2name)
    rel_extractor = RelationExtractor(
        relation_model_path='../train/results_relation/checkpoint-4140',
        spacy_model_path="../en_core_web_md"
    )
    


    with open("../datasets/convert_eval_multihop_with_re_questions_with_wiki_new.json", "r") as f:
        data = json.load(f)
        # 过滤掉不包含 'port_new' 字段的数据
        data = [d for d in data if 'port_new' in d]
    # print
 
    if args.information_type=="image":
        with open("../datasets/all_triple_embeddings_clip2_only_image.pkl", "rb") as f:
            image_embedding_data = pickle.load(f)
            # 提取所有的 key 和 value，分别存储到 id_list 和 image_embedding_list
            id_list = []
            image_embedding_list = []
            for k, v in image_embedding_data.items():
                id_list.append(k)
                image_embedding_list.append(v)
    elif args.information_type=="image+text":
        with open("../datasets/all_triple_embeddings_clip2_only_image+entity.pkl", "rb") as f:
            image_embedding_data = pickle.load(f)
            # 提取所有的 key 和 value，分别存储到 id_list 和 image_embedding_list
            id_list = []
            image_embedding_list = []
            for k, v in image_embedding_data.items():
                id_list.append(k)
                image_embedding_list.append(v)
    else:
        print("information_type must be image or image+text")
        exit(0) 
            
    

    # 按照之前的设定这里也只用融入和修改这么多节点
    batch_edits = args.edit
    batch_size = 1304 // batch_edits
    batch_data = []
    idx = 0
    for _ in range(batch_size):

        b = []
        for i in range(batch_edits):
            if idx < len(data):
                b.append(data[idx])
            idx += 1
        batch_data.append(b)

    
    # 在 main() 函数开头定义

    # 日志文件路径
    log_path = f"./results/log_{current_time}_no_reflect_CLIP2_{divide_modal}_{model_name}_information_type_{args.information_type}_image_{str(image_type)}.txt"
    with open(log_path, "a", encoding="utf-8") as log_f:
        for batch in batch_data:
            for i in tqdm(range(len(batch))):
                d = batch[i]
                if image_type == 100:
                    image_id = d["image"]
                elif image_type == 999:
                    image_id = d["image_rephrase"]
                else:
                    print("image_type must be 100 or 999")
       
                hops = d["port_new"]
            
                for index, hop in enumerate(hops):
                    image_flag = False
                    tot += 1
                    tot_list[index] += 1
                    found_ans = gold_path = False
                    last_triple = hop["triple{0}".format(index+2)]
                    current_answer_all_path,last_answer =  constrcution_path_answer(hop)
                    
                    
                    for index_q,q in enumerate(last_triple['rewrite_question']):
                        print("----------"*30)
                        print("Question: " + q)
                        
                        log_f.write(f"\n==== Question ====\n{q}\n")
                        entity_list = []
                        
                    
                        # sub_questions = run_llm_divide(question_prompt, local_modal)

                        # 保存到字典
                        if q in question2subqs:
                            sub_questions = question2subqs[q]
                        else:
                            # 如果没有找到，就直接使用 LLM 进行分解
                            question_prompt = divide_base_prompt.replace("<<<<QUESTION>>>>", q)
                            question_prompt = question_prompt.replace("<<<STEP>>>", str(index+2))
                            print("No cached sub-questions found for: " + question_prompt)
                            sub_questions = run_llm_divide(question_prompt, local_modal)
                            # print("Sub-questions: " + str(sub_questions))
                            question2subqs[q] = sub_questions
                        
                        for index_sub, subq in enumerate(sub_questions):
                            log_f.write(f"Sub-questions {index_sub}: {subq}\n")
                            print("subq: " + subq)  
                            #这里要去检索了

                            if "[IMAGE]" in subq:
                                cache_key = f"{subq}|||{image_id}"
                                # print("cache_key: " + cache_key)
            
                                if cache_key in entity_cache:
                                    top5_ids = entity_cache[cache_key]
                                    print("Found entity in cache: " + ", ".join(top5_ids))
                                else:
                                    subq = subq.replace("[IMAGE]", "picture")
                                    input_query = [input_constrcution(subq, image_id)]
                                    input_query_embedding = Mretriever.encode_queries(input_query,args.information_type,have_image=True)
                                    topk_scores, topk_indices = Mretriever.compute_scores(input_query_embedding, image_embedding_list, topK=1)
                                    top5_ids = [id_list[idx] for idx in topk_indices[0].tolist()]
                                    entity_cache[cache_key] = top5_ids
                                top_names = [qid2name[eid] for eid in top5_ids]
                                entity = top_names[0]
                                
                            else:
                                print("------------")
                                if entity is None or entity == "":
                                    entity = "unknown"
                                    break
                                print("entity: " + entity)
                                
                                subq = subq.replace("[ENT]", entity)
                                # 这一部分是RAG检索
                            
                                if entity.lower() in qname2id:
                                    wiki_id = qname2id[entity.lower()]
                                
                                else:
                                    wiki_id = link_entity(entity, nlp)
                                
                                input_query = [input_constrcution(subq,image_id=None)]
                                condidate_triples,relation_tails,triples_for_LLMs = multimodal_graph.get_triples_by_id(wiki_id)
                                # print(triples_for_LLMs)
                            
                                # print("condidate_triples", condidate_triples)
                                # print("relation_tails", relation_tails) 
                                topk_triples = []
                                if len(triples_for_LLMs)!=0:
                                    condidate_triples = [input_constrcution(triple, image_id=None) for triple in condidate_triples]
                                # print(condidate_triples)
                                # print("input_query", input_query)
                                # print("condidate_triples", condidate_triples)
                                    input_query_embedding = Mretriever.encode_queries(input_query,have_image=False)
                                    condidate_embedding = Mretriever.encode_passages(condidate_triples,have_image=False)
                                    topk_scores, topk_indices = Mretriever.compute_scores(input_query_embedding, condidate_embedding,topK=args.topK)
                                    
                                    topk_triples = [condidate_triples[idx]['txt'] for idx in topk_indices[0].tolist()]
                                #这里去引入question 以及topk_triples
                            
                                answer_prompt = answer_base_prompt.replace("<<<QUESTION>>>", subq)
                                
                                answer_prompt = answer_prompt.replace("<<<FACT>>>", "; ".join(topk_triples))
                                # print(answer_prompt)
                                
                                output = run_llm_answer(answer_prompt,local_modal,image_id)
                                print("-----------------"*20)
                                print("Output from LLM: " + output)
                                # exit()

                                output  = re.sub(r'^\s*Answer:\s*', 'Answer:', output , flags=re.IGNORECASE)
                                match = re.search(r'^\s*Answer:\s*(.*)', output, flags=re.IGNORECASE)
                                print("match: " + str(match))
                                print("----------------")
                                
                                if match:
                                    rag_entity = match.group(1).strip().replace("_", " ")
                                else:
                                    # 如果 rag 只有一行并且Entity 的token数量不超过5
                                    rag_lines = output.strip().split('\n')
                                    if len(rag_lines) == 1 and len(rag_lines[0].split(" ")) <= 5:
                                        rag_entity = rag_lines[0].strip().replace("_", " ")
                                    else:
                                        rag_entity = None
                                
                                
                                #这一部分是 retreive 
                                print("RAG entity: {0}".format(rag_entity))
                                extract_key, extract_entity = rel_extractor.retrieve_from_extraction(subq,wiki_id, entity,relation_tails)
                                print("Extracted entity: " + str(extract_entity))

                                if extract_entity is not None:
                                    if rag_entity is None :
                                        entity = extract_entity.replace("_", " ")
                                    else:
                                        #两个都有值
                                        # 判断 Rag_entity 与 extract_entity 是否属于同一个 wiki id
                                        if rag_entity.lower() in qname2id:
                                            rag_wiki_id = qname2id[rag_entity.lower()]
                                        else:
                                            rag_wiki_id = link_entity(rag_entity, nlp)
                                        if extract_entity.lower() in qname2id:
                                            extract_wiki_id = qname2id[extract_entity.lower()]
                                        else:
                                            extract_wiki_id = link_entity(extract_entity, nlp)
                                        #如果相同直接用
                                        if rag_wiki_id.lower() == extract_wiki_id.lower() or rag_entity.lower() == extract_entity.lower():
                                            entity = rag_entity
                                        else:
                                        #如果不同就去选择
                                            entity = random.choice([rag_entity, extract_entity]).replace("_", " ")
                                
                                else:
                                    if rag_entity is not None:
                                        # 如果 RAG 有结果，就直接用 RAG 的结果
                                        entity = rag_entity
                                    else:
                                        entity = "unknown"
                                
                                
                                    
                                    # 如果都没有那就 brek
                            print("entity: " + entity)
                                    
                            entity_list.append(entity)
                        
                        ans =  entity
                        print("Predict list: " + str(entity_list))
                        print("Last answer: " + str(last_answer))
                        print("Current answer all path: " + str(current_answer_all_path))
                        print("ans: " + str(ans))
                        log_f.write(f"Ground truth: {last_answer}\n")
                        log_f.write(f"Prediction: {entity_list}\n")
                        log_f.write(f"Current answer all path: {current_answer_all_path}\n")
                        log_f.write(f"ans: {ans}\n")
                


                        # 检查 entity_list 是否为空且第一个元素不为 None
                        if entity_list and entity_list[0] is not None and current_answer_all_path and current_answer_all_path[0] and current_answer_all_path[0][0] is not None:
                            if entity_list[0].lower() == current_answer_all_path[0][0].lower():
                                if image_flag == False:
                                    image_retriever_correct[index] += 1
                                    image_flag = True



                        if not found_ans:
                            if ans is None or ans.lower().strip() == "" or  len(ans.lower().strip())<=3  or not any(c.isalpha() for c in ans):
                                continue
                            if any(ans.lower().replace("_", " ") in k.lower().replace("_", " ") for k in last_answer) or any(ans.lower().replace("_", " ") in k.lower() for k in last_answer):
                                cor += 1
                                cor_list[index] += 1
                                found_ans = True
                        if found_ans:
                            gold_path = verify_gold_path(entity_list[0:-1], current_answer_all_path[0:-1])
                            if gold_path:
                                ver_cor += 1
                                ver_cor_list[index] += 1
                                break
                        
                        # print(f"Total: {total}, Correct: {correct},  Total: {correct / total}")
                    print(f'Acc = {cor / tot} ({cor} / {tot})')
                    print(f'Hop-Acc = {ver_cor / tot} ({ver_cor} / {tot})')
                    print(f'1-hop-tot = {tot_list[0]} 1-hop-cor = {cor_list[0]} 1-hop-acc = {ver_cor_list[0]}, 1-hop-image_retriever_correct = {image_retriever_correct[0]}')
                    print(f'2-hop-tot = {tot_list[1]} 2-hop-cor = {cor_list[1]} 2-hop-acc = {ver_cor_list[1]},2-hop-image_retriever_correct = {image_retriever_correct[1]}')
                    print(f'3-hop-tot = {tot_list[2]} 3-hop-cor = {cor_list[2]} 3-hop-acc = {ver_cor_list[2]},3-hop-image_retriever_correct = {image_retriever_correct[2]}')
                    print(f'4-hop-tot = {tot_list[3]} 4-hop-cor = {cor_list[3]} 4-hop-acc = {ver_cor_list[3]},4-hop-image_retriever_correct = {image_retriever_correct[3]}')
                    log_f.write(f"Acc = {cor / tot} ({cor} / {tot})\n")
                    log_f.write(f"Hop-Acc = {ver_cor / tot} ({ver_cor} / {tot})\n")
                    log_f.write(f"1-hop-tot = {tot_list[0]} 1-hop-cor = {cor_list[0]} 1-hop-acc = {ver_cor_list[0]}, 1-hop-image_retriever_correct = {image_retriever_correct[0]}\n")
                    log_f.write(f"2-hop-tot = {tot_list[1]} 2-hop-cor = {cor_list[1]} 2-hop-acc = {ver_cor_list[1]},2-hop-image_retriever_correct = {image_retriever_correct[1]}\n")
                    log_f.write(f"3-hop-tot = {tot_list[2]} 3-hop-cor = {cor_list[2]} 3-hop-acc = {ver_cor_list[2]},3-hop-image_retriever_correct = {image_retriever_correct[2]}\n")
                    log_f.write(f"4-hop-tot = {tot_list[3]} 4-hop-cor = {cor_list[3]} 4-hop-acc = {ver_cor_list[3]},4-hop-image_retriever_correct = {image_retriever_correct[3]}\n")

    # with open(f"question2subqs_{model_name}.json", "w", encoding="utf-8") as f:
    #     json.dump(question2subqs, f, ensure_ascii=False, indent=2)
    with open(entity_cache_path, "w", encoding="utf-8") as f:
        json.dump(entity_cache, f, ensure_ascii=False, indent=2)
    log_f.close()

if __name__ == "__main__":
    
    main()

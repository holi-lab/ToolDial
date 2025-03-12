import json,re,requests,os,random,time
import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")

from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
)
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import faiss

def relation(subgraph):
    edge = subgraph['edge'][0]
    source,target = edge['source'],edge['target']
    output_comp = edge['source_attribute']['name']
    input_param = edge['target_attribute']['name']
    return f"The output component {output_comp} of API {source} can be used as input parameter {input_param} of API {target}"

def normalize_param(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', text)
    return cleaned_text.lower()

def longest_common_substring(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    longest = 0
    lcs_end = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > longest:
                    longest = dp[i][j]
                    lcs_end = i
            else:
                dp[i][j] = 0
    lcs = str1[lcs_end - longest: lcs_end]
    return lcs

def similarity_score(str1, str2):
    str1 = normalize_param(str1.lower())
    str2 = normalize_param(str2.lower())
    lcs = longest_common_substring(str1, str2)
    lcs_length = len(lcs)
    similarity = (2 * lcs_length) / (len(str1) + len(str2)+0.01)
    return similarity

def longest_common_subsequence(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

def similarity_object(str1, str2):
    str1 = normalize_param(str1)
    str2 = normalize_param(str2)
    lcs_len = longest_common_subsequence(str1, str2)
    avg_len = (len(str1) + len(str2)) / 2
    similarity = lcs_len / avg_len
    return similarity

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def dict_sort(current_dict):
    return dict(sorted(current_dict.items(), key=lambda item: item[1], reverse=True))

def Dial(prompt,model_name="gpt-4o"):
    llm = ChatOpenAI(model_name=model_name,
                    temperature=0,
                    max_tokens=4096,
                    openai_api_key="TYPE YOUR API KEY ")
    generated = llm([HumanMessage(content=prompt)]).content
    return generated

def is_all_different(data_list):
    data1,data2 = data_list[0],data_list[0]
    cnt=0
    for param_type in ['required_parameters',"optional_parameters"]:
        for d1,d2 in zip(data1[param_type],data2[param_type]):
            if data1[param_type][d1]==data2[param_type][d2]:
                cnt+=1
    if cnt==0:
        return True
    
def graph_aug(subgraph):
    dep_idx = subgraph[0][0]
    des_idx = subgraph[0][1]
    current_edge = all_edge_list_graph[dep_idx][des_idx][0]
    chain_output,chain_param = current_edge[1][0],current_edge[3][0]
    return {"subgraph":subgraph,"chain_output":chain_output,"chain_param":chain_param,}

def freeformat(subgraph):
    return_format = {"nodes":[],
                    "edge":{"source":"",
                     "target":"",
                     "source_attribute":{"name":"","type":"","object_ad":"","object_long":""},
                     "target_attribute":{"name":"","type":"","object_ad":"","object_long":""}}
                    }
    node1,node2 = subgraph['subgraph'][0][0],subgraph['subgraph'][0][1]
    return_format['nodes']+=[normal_idx2igraph[node1],normal_idx2igraph[node2]]
    current_edge = all_edge_list_graph[node1][node2][0]
    return_format['edge']['source'] = normal_idx2igraph[node1]
    return_format['edge']['target'] = normal_idx2igraph[node2]
    return_format['edge']['source_attribute']['name'] = current_edge[1][0]
    return_format['edge']['source_attribute']['object_ad'] = current_edge[1][1]
    return_format['edge']['source_attribute']['object_long'] = current_edge[1][2]
    return_format['edge']['target_attribute']['name'] = current_edge[3][0]
    return_format['edge']['target_attribute']['object_ad'] = current_edge[3][1]
    return_format['edge']['target_attribute']['object_long'] = current_edge[3][2]
    return_format['edge'] = [return_format['edge']]
    return_format['type'] = "seq"
    for key in subgraph.keys():
        if key in ["subgraph","desc_unreal","desc_chain_output","desc_chain_param"]:
            continue
        return_format[key] = subgraph[key]
        return_format[key] = subgraph[key]
    return_format['source_req'] = [{"name":comp['name'],"description":comp['object_ad']} for comp in api_list_dict[return_format['edge'][0]['source']]['required_parameters']]
    return_format['source_opt'] = [{"name":comp['name'],"description":comp['object_ad']} for comp in api_list_dict[return_format['edge'][0]['source']]['optional_parameters']]
    return_format['target_req'] = [{"name":comp['name'],"description":comp['object_ad']} for comp in api_list_dict[return_format['edge'][0]['target']]['required_parameters']]
    return_format['target_opt'] = [{"name":comp['name'],"description":comp['object_ad']} for comp in api_list_dict[return_format['edge'][0]['target']]['optional_parameters']]
    return return_format        

with open("../after_coor_before_emb.json",'r') as f:
    api_list = json.load(f)
api_list_dict = {api['full_name']:api for api in api_list}

with open("pair_list/aug_desc.json","r") as f:
    aug_desc_list = json.load(f)
    
for api in api_list_dict:
    if api in aug_desc_list:
        api_list_dict[api]['description'] = aug_desc_list[api]
    
with open(f"pair_list/all_edge_list_graph.json",'r') as f:
    all_edge_list_graph = json.load(f)
    
with open(f"pair_list/normal_graph2idx.json",'r') as f:
    normal_graph2idx = json.load(f)
    
with open(f"pair_list/normal_idx2igraph.json",'r') as f:
    normal_idx2igraph = json.load(f)

normal_idx2igraph = {int(key):normal_idx2igraph[key] for key in normal_idx2igraph}

with open("pair_list/used_api.json",'r') as f:
    used_api = json.load(f)

def count_leaf_keys(json_data, parent_key='', key_count=None):
    if key_count is None:
        key_count = {}
        
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            full_key = f"{parent_key}|{key}" if parent_key else key
            if isinstance(value, dict) or isinstance(value, list):
                count_leaf_keys(value, full_key, key_count)
            else:
                key_count[full_key] = key_count.get(full_key, 0) + 1
    elif isinstance(json_data, list):
        for item in json_data:
            count_leaf_keys(item, parent_key, key_count)
            
    return key_count

def extract_leaf(output_components):
    leaf_list = set()
    for output in output_components:
        leaf_list.add(output['name'])
    return leaf_list

for idx,api in enumerate(api_list):
    key_counts = count_leaf_keys(api['output']['response'])
    leaf_list = extract_leaf(api['output_components'])

    multi_key = set()
    for key in key_counts:
        if key_counts[key]>1:
            multi_key.add(key)
    if len(multi_key)>0:
        api['is_list'] = multi_key
    else:
        api['is_list'] = set()
api_list_dict = {api['full_name']:api for api in api_list}        
        
def shuffle_same(list1,list2):
    return_list = []
    for c1,c2 in zip(list1,list2):
        tmp_dict = {}
        tmp_dict[c1]=c2
        return_list.append(tmp_dict)
    random.shuffle(return_list)
    list1 = [list(key.keys())[0] for key in return_list]
    list2 = [list(value.values())[0] for value in return_list]
    return list1,list2

class Retriever:
    def __init__(self,device):
        self.model = SentenceTransformer("ToolBench/ToolBench_IR_bert_based_uncased",device = device)
        self.sentence = [api_list_dict[api]['description'] for api in api_list_dict if api in used_api]
        self.name = [api_list_dict[api]['full_name'] for api in api_list_dict if api in used_api]
        print("Encoding start...",end="")
        self.embeddings = self.model.encode(self.sentence, convert_to_tensor=True)
        print("Done!")
        if "cuda" in device:
            self.embeddings = self.embeddings.to("cpu")
        self.normalized_embeddings = normalize(self.embeddings, norm='l2')
        embedding_dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dimension)
        self.index.add(self.normalized_embeddings)
    def search_api(self,query,answer,k,vague=False,suggest=False):
        query_embedding = self.model.encode(query)
        normalized_query_embedding = normalize(query_embedding.reshape(1, -1), norm='l2')
        distances, indices = self.index.search(normalized_query_embedding, k)
        retrieved_list = [self.name[idx] for idx in indices[0]]
        distances = list(distances[0])
        try:
            answer_idx = retrieved_list.index(answer)
        except:
            aaa = int(len(retrieved_list)*0.25),int(len(retrieved_list)*0.75)
            answer_idx = random.sample(list(range(aaa[0],aaa[1])),1)[0]
            retrieved_list.insert(answer_idx,answer)
            distances.insert(answer_idx,(distances[answer_idx-1]+distances[answer_idx])/2)
        if vague:
            if not suggest:
                num = random.randint(0,5)
                while distances[answer_idx-num]>0.5:
                    num+=1
                return {"list":retrieved_list[answer_idx-num:answer_idx+5-num],"score":distances[answer_idx-num:answer_idx+5-num]}
        return {"list":retrieved_list[answer_idx:answer_idx+5],"score":distances[answer_idx:answer_idx+5]}
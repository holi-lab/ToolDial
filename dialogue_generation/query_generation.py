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
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import faiss

with open("config.yml") as f:
    config = yaml.load(f,Loader=yaml.SafeLoader)
    
openai_key = config['api_key']
device = config['device']

def relation(subgraph):
    edge = subgraph['edge'][0]
    source,target = edge['source'],edge['target']
    output_comp = edge['source_attribute']['name']
    input_param = edge['target_attribute']['name']
    return f"The output component {output_comp} of API {source} can be used as input parameter {input_param} of API {target}"

## Metric

def normalize_param(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', text)
    return cleaned_text.lower()

def longest_common_substring(str1, str2): ## LCS를 찾기 전에 lower 해주고, 특수 문자를 제거해줌
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    longest = 0
    lcs_end = 0  # LCS가 끝나는 위치
    
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

def similarity_score(str1, str2): ## LCS를 찾기 전에 lower 해주고, 특수 문자를 제거해줌
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
                    openai_api_key=openai_key)
    generated = llm([HumanMessage(content=prompt)]).content
    return generated

#### mock data function

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

model = SentenceTransformer('all-mpnet-base-v2',device=device)

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
normal_graph = np.load(f"pair_list/normal_graph.npy")

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

edge_pair_for_bfs = []
for row_idx in range(normal_graph.shape[0]):
    for col_idx in range(normal_graph.shape[0]):
        if normal_graph[row_idx][col_idx]:
            edge_pair_for_bfs.append((row_idx,col_idx))
            
vertex = list(range(normal_graph.shape[0]))

sports_dict= {}
for api in normal_graph2idx:
    if "Sports" in api:
        sports_dict[api]=normal_graph[normal_graph2idx[api],:].sum()+normal_graph[:,normal_graph2idx[api]].sum()
        
sports_dict = dict(sorted(sports_dict.items(), key=lambda item: item[1],reverse=True))

top=40
ben_sports_name=[]
for idx,api in enumerate(sports_dict):
    if idx==top:
        break
    ben_sports_name.append(normal_graph2idx[api])
    
for idx,api in enumerate(normal_graph2idx):
    if "esport" in api.lower():
        ben_sports_name.append(normal_graph2idx[api])
        
################
vague_sugg_trg_prompt = """
This is the documentation of an API:
{api_docs}

Input parameter list:
{input_param_docs}

Create a user query that can only be answered using this API. The query should be vague (no input parameter information should be included in the query. Refer to the input parameter list as needed).

<Rules>
1. Only generate the query, nothing else.
2. The query should be 5-7 words long.
3. There should be no information in the query that can directly fill the API's input parameters.
4. Use words that are not found in the API description as much as possible.
"""

vague_sugg_refine_trg_prompt="""
API documentation: {api_docs}

Input parameter list:
{input_param_docs}

User Query: {query}

This user query has a similarity score of {sim_score} to the above API documentation in the dense retriever.
The score should be between 0.5 and 0.6.
- If the score is lower than 0.5, modify the query to be more similar to the API documentation. Try to generate the query using words that are found in the API description as much as possible.
- If the score is higher than 0.6, modify the query to be less similar to the API documentation. Try to generate the query using words that are not found in the API description.

If you refer to the queries and their similarity scores that you’ve refined so far, it will help you refine further.
Refinement history:
{history}

<Rules>
1. Only generate the query, nothing else.
2. The query should be 5-7 words long.
3. Do not use examples from the API documentation.
4. The query should not contain information that directly fills the API's input parameters.
"""

################
clear_trg_prompt = """
This is the documentation of an API:
{api_docs}

Input parameter list:
{input_param_docs}

Create a user query that can only be addressed using this API. The query should contain a clear intent (but no input parameter information should be included). Refer to the input parameter list as needed.

<Rules>
1. Only generate the query, nothing else.
2. The query should be 12-15 words long.
3. There should be no information in the query that can directly fill the API's input parameters.
"""

clear_refine_trg_prompt="""
API documentation: {api_docs}

User Query: {query}

Input parameter list:
{input_param_docs}

This user query has a similarity score of {sim_score} to the above API documentation in the dense retriever.
Modify the given query to make it clearer so that the retriever's score increases. Ideally, the score should be higher than 0.55. 
It would be beneficial to modify the query to resemble sentences in the API documentation.

If you refer to the queries and their similarity scores that you’ve refined so far, it will help you refine further.
Refinement history:
{history}

<Rules>
1. Only generate the query, nothing else.
2. The query should be 10-12 words long.
3. Do not use examples from the API documentation.
4. The query should not contain information that directly fills the API's input parameters.
"""

from custom_retriever import Retriever

retriever = Retriever(device)
model_ret = SentenceTransformer("ToolBench/ToolBench_IR_bert_based_uncased",device = device)

def input_param_docs(api):
    prompt = ""
    for idx,input_param in enumerate(api_list_dict[api]['required_parameters']):
        if idx == 0:
            prompt+="Required_parameters:\n"
        prompt+=f"{input_param['name']}:{input_param['object_ad']}\n"
    for idx,input_param in enumerate(api_list_dict[api]['optional_parameters']):
        if idx == 0:
            prompt+="\nOptional_parameters:\n"
        prompt+=f"{input_param['name']}:{input_param['object_ad']}\n"
    if prompt == "":
        prompt+="No input parameter for this API"
    return prompt

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def vague_sugg_query_generator_trg(api):
    api_docs = api_list_dict[api]['description']
    api_docs_emb = model_ret.encode(api_docs)
    query = Dial(vague_sugg_trg_prompt.format(api_docs = api_docs,input_param_docs=input_param_docs(api)))
    sim = cosine_similarity(api_docs_emb,model_ret.encode(query))
    query_history = []
    cnt=0
    while sim<0.5 or sim>0.6:
        cnt+=1
        if cnt>10:
            return False
        query_history.append((query,sim))
        query = Dial(vague_sugg_refine_trg_prompt.format(api_docs=api_docs,query=query,sim_score=sim,history=query_history[-5:],input_param_docs=input_param_docs(api)))
        sim = cosine_similarity(api_docs_emb,model_ret.encode(query))
    return query

def clear_query_generator_trg(api):
    api_docs = api_list_dict[api]['description']
    api_docs_emb = model_ret.encode(api_docs)
    query = Dial(clear_trg_prompt.format(api_docs = api_docs,input_param_docs=input_param_docs(api)))
    sim = cosine_similarity(api_docs_emb,model_ret.encode(query))
    query_history = []
    cnt = 0
    while sim<0.6:
        cnt+=1
        if cnt>10:
            return False
        query_history.append((query,sim))
        query = Dial(clear_refine_trg_prompt.format(api_docs=api_docs,query=query,sim_score=sim,history=query_history[-5:],input_param_docs=input_param_docs(api)))
        sim = cosine_similarity(api_docs_emb,model_ret.encode(query))
    return query
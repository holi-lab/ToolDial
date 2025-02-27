import json,re
import numpy as np
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
model_ret = SentenceTransformer("ToolBench/ToolBench_IR_bert_based_uncased",device = device)
###############################        

situation_prompt = """
A user has made a query to the system.

API documentation:
{src_api}
{src_api_docs}

{trg_api}
{trg_api_docs}

In this scenario, the query can be fulfilled by calling {trg_api}, {trg_input} is sourced from {src_output}, 
the output of {src_api} because user can't provide the {trg_input} for some reaons.

Based on this, imagine a plausible situation that the user might be in in one paragraph.
"""

def api_docs_generation(api):
    raw_docs = api_list_dict[api]
    prompt = ""
    prompt+=f"Description:{raw_docs['description']}\n\n"
    prompt+="Input Parameters:\n"
    for req_param in api_list_dict[api]['required_parameters']:
        prompt+=f"{req_param['name']}:{req_param['object_long']}\n"
    prompt+="\nOutput Components:\n"
    for output in raw_docs['output_components']:
        prompt+=f"{output['name']}:{output['object_long']}\n"
    return prompt

def situation_generator(pair):
    current_edge = freeformat(graph_aug(pair))
    source = current_edge['nodes'][0]
    target = current_edge['nodes'][1]
    src_docs = api_docs_generation(source)
    trg_docs = api_docs_generation(target)

    trg_input = current_edge['edge'][0]['target_attribute']['name']
    src_output = current_edge['edge'][0]['source_attribute']['name']
    current_prompt = situation_prompt.format(
        trg_api_docs = trg_docs,
        src_api_docs = src_docs,
        src_api = source,
        trg_api = target,
        trg_input = trg_input,
        src_output = src_output
    )
    return Dial(current_prompt,model_name="gpt-4o")

################
clear_prompt = """
This is the documentation of some two APIs:
Source API:{src_docs}

Target API:{trg_docs}

Input parameter list:
{src_input_param}

{trg_input_param}

Situation of the user:
{situation}

Create a query of a user that can only be met by using this target API. 
The query should contain clear intent. However, the input parameters from the input parameter list should not be included in the utterance.
The utterance should not contain information that can be used to fill in the input parameters of the above APIs.
The query should be generated in such a way that it has a certain degree of relevance to the content of the source API's description.

<Rules>
1. Just generate the query, not a single words.
2. There is no length limit for the query, but it must be at least 15 words long.
3. There should be no information within query that can fill the API's input parameter.
"""

clear_refine_prompt="""
This is the documentation of some two APIs:
Source API:{src_docs}

Target API:{trg_docs}

Input parameter list of target API:
{trg_input_param}

Input parameter list of source API:
{src_input_param}

User Query:{query}

Situation of the user:
{situation}

This user query has a similarity of {sim_score_trg} to the above target API documentation in the dense retriever.
And it has a similarity score of {sim_score_src} with the source API's documentation.
Modify the given query so that it has a similarity score of at least 0.5 with the source API documentation and at least 0.6 with the target API documentation.

It would be good to modify it similar to the sentence in target API documentation.
The query should contain clear intent. However, the input parameters from the input parameter list should not be included in the utterance.
The utterance should not contain information that can be used to fill in the input parameters of the above APIs.
The query should be generated in such a way that it has a certain degree of relevance to the content of the source API's description.
In other words, when creating the query, the source API's documentation should also be referenced.

If you refer to the queries and their similarity score you've been refine so far, it will help you to refine.
Refine history:
{history}

<Rules>
1. Just generate the query, not a single words.
2. There is no length limit for the query, but it must be at least 15 words long.
3. Don't use the example of the API documentation
4. There should be no information within query that can fill the API's input parameter.
"""

#############

vague_sugg_prompt = """
This is the documentation of some two APIs:
Source API:{src_docs}

Target API:{trg_docs}

Input parameter list:
{src_input_param}

{trg_input_param}

Situation of the user:
{situation}

Create a query of a user that can only be met by using these APIs. The query should be vague(No input parameter information inside the query. Refer the Input parameter list properly).
The utterance should not contain information that can be used to fill in the input parameters of the above APIs.
Query should be a query that can be solved through the results of Target API.
The query should be generated in such a way that it has a certain degree of relevance to the content of the Source API's description.

<Rules>
1. Just generate the query, not a single words.
2. There should be no information within Query that can fill the API's input parameter.
3. Try to generate query with words that are not in the API description as much as possible
"""

vague_sugg_refine_prompt="""
This is the documentation of some two APIs:
Source API:{src_docs}

Target API:{trg_docs}

Input parameter list:
{src_input_param}

{trg_input_param}

User Query:{query}

Situation of the user:
{situation}

This user query has a similarity of {sim_score_trg} to the above Target API documentation in the dense retriever.
And it has a similarity score of {sim_score_src} with the Source API's documentation.
Modify the given query so that it has a similarity score of at least 0.5 with the Source API documentation and at least 0.5~0.6 score with the Target API documentation.
The utterance should not contain information that can be used to fill in the input parameters of the above APIs.
If you refer to the queries and their similarity score you've been refine so far, it will help you to refine.
Refine history:
{history}

<Rules>
1. Just generate the query, not a single words.
2. Don't use the example of the API documentation
3. There should be no information within query that can fill the API's input parameter.
"""


def input_param_docs(api):
    prompt = f"API: {api}\n"
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

def clear_query_generator(src_api,trg_api,situation):
    src_api_docs = api_list_dict[src_api]['description']
    trg_api_docs = api_list_dict[trg_api]['description']
    src_api_docs_emb = model_ret.encode(src_api_docs)
    trg_api_docs_emb = model_ret.encode(trg_api_docs)
    
    query = Dial(clear_prompt.format(src_docs = src_api_docs,
                                     trg_docs = trg_api_docs,
                                     situation = situation,
                                     src_input_param=input_param_docs(src_api),
                                     trg_input_param=input_param_docs(trg_api)))
    sim_src = cosine_similarity(src_api_docs_emb,model_ret.encode(query))
    sim_trg = cosine_similarity(trg_api_docs_emb,model_ret.encode(query))
    query_history = []
    cnt = 0
    while sim_trg<0.6 or sim_src<0.5:
        # print(query)
        # print(sim_trg,sim_src)
        cnt+=1
        if cnt>20:
            return False
        desc_text = f"Similarity with Target API docs: {sim_trg}, Similarity with Source API docs: {sim_src}"
        also = False
        if sim_trg<0.6:
            desc_text+="It should be modified to include more words from the Target API documentation."
            also=True
        if sim_src<0.5:
            if also:
                desc_text+="Also, "
            desc_text+="It should be modified to include more words from the Source API documentation."
        query_history.append((query,desc_text))
        query = Dial(clear_refine_prompt.format(src_docs = src_api_docs,
                                                trg_docs = trg_api_docs,
                                                query=query,
                                                situation = situation,
                                                sim_score_trg=sim_trg,
                                                sim_score_src=sim_src,
                                                history=query_history[-5:],
                                                src_input_param=input_param_docs(src_api),
                                                trg_input_param=input_param_docs(trg_api)))
        sim_src = cosine_similarity(src_api_docs_emb,model_ret.encode(query))
        sim_trg = cosine_similarity(trg_api_docs_emb,model_ret.encode(query))
    # print(sim_trg,sim_src)
    return query

def vague_sugg_query_generator(src_api,trg_api,situation):
    src_api_docs = api_list_dict[src_api]['description']
    trg_api_docs = api_list_dict[trg_api]['description']
    src_api_docs_emb = model_ret.encode(src_api_docs)
    trg_api_docs_emb = model_ret.encode(trg_api_docs)
    
    query = Dial(clear_prompt.format(src_docs = src_api_docs,
                                     trg_docs = trg_api_docs,
                                     situation = situation,
                                     src_input_param=input_param_docs(src_api),
                                     trg_input_param=input_param_docs(trg_api)))
    sim_src = cosine_similarity(src_api_docs_emb,model_ret.encode(query))
    sim_trg = cosine_similarity(trg_api_docs_emb,model_ret.encode(query))
    query_history = []
    cnt = 0
    while sim_trg<0.5 or sim_trg>0.6 or sim_src<0.5:
        # print(query)
        # print(sim_trg,sim_src)
        cnt+=1
        if cnt>20:
            return False
        desc_text = f"Similarity with Target API docs: {sim_trg}, Similarity with Source API docs: {sim_src}"
        also = False
        if sim_trg>0.6:
            desc_text+="It should be modified to avoid overlapping with the words in the Target API documentation."
            also=True
        elif sim_trg<0.5:
            desc_text+="It should be modified to include more words from the Target API documentation."
            also=True
        if sim_src<0.5:
            if also:
                desc_text+="Also, "
            desc_text+="It should be modified to include more words from the Source API documentation."
        query_history.append((query,desc_text))
        query = Dial(vague_sugg_refine_prompt.format(src_docs = src_api_docs,
                                                trg_docs = trg_api_docs,
                                                query=query,
                                                situation = situation,
                                                sim_score_trg=sim_trg,
                                                sim_score_src=sim_src,
                                                history=query_history[-5:],
                                                src_input_param=input_param_docs(src_api),
                                                trg_input_param=input_param_docs(trg_api)))
        sim_src = cosine_similarity(src_api_docs_emb,model_ret.encode(query))
        sim_trg = cosine_similarity(trg_api_docs_emb,model_ret.encode(query))
    # print(sim_trg,sim_src)
    return query

vague_prompt = """
This is the documentation of some API:
{api_docs}

Input parameter list:
{input_param_docs}

Create a query of a user that can only be met by using this API. The query should be vague(No input parameter information inside the query. Refer the Input parameter list properly).

<Rules>
1. Just generate the query, not a single words.
2. The query should be 5~7 words.
3. There should be no information within Query that can fill the API's input parameter.
4. Try to generate query with words that are not in the API description as much as possible
"""

vague_refine_prompt="""
API documentation:{api_docs}

Input parameter list:
{input_param_docs}

User Query:{query}

This user query has a similarity of {sim_score} to the above application documentation in the dense retriever.
Modify the given query to make it more ambiguous so that the retriever's score comes out lower. It is good to come out lower than 0.55.

If you refer to the queries and their similarity score you've been refine so far, it will help you to refine.
Refine history:
{history}

<Rules>
1. Just generate the query, not a single words.
2. The query should be 5~7 words.
3. Don't use the example of the API documentation
4. There should be no information within Query that can fill the API's input parameter.
5. Try to generate query with words that are not in the API description as much as possible
"""

clear_add_prompt = """
This is the documentation of some API:
{api_docs}

Input parameter list:
{input_param_docs}

Create a query of a user that can only be met by using this API. The query should be clear.
Additionally, the user provides information '{value}' that can enter the input parameter {input_param} implicitly within query.
<Rules>
1. Just generate the query, not a single words.
2. The query should be 10~12 words.
3. There should be no information within Query that can fill the API's input parameter except the '{input_param}' as '{value}'.
"""

clear_add_refine_prompt = """
API documentation:{api_docs}

User Query:{query}

Input parameter list:
{input_param_docs}

This user query has a similarity of {sim_score} to the above application documentation in the dense retriever.
Modify the given query to make it more clear so that the retriever's score comes out higher. It is good to come out higher than 0.55.
In this query, user is providing the information '{value}' that can fill the input parameter {input_param}. Even if Query is modified, it should never be modified.

It would be good to modify it similar to the sentence in API documentation.

If you refer to the queries and their similarity score you've been refine so far, it will help you to refine.
Refine history:
{history}

<Rules>
1. Just generate the query, not a single words.
2. The query should be 10~12 words.
3. Don't use the example of the API documentation
4. There should be no information within Query that can fill the API's input parameter except the '{input_param}' as '{value}'.
"""

def vague_query_generator(api):
    api_docs = api_list_dict[api]['description']
    api_docs_emb = model_ret.encode(api_docs)
    query = Dial(vague_prompt.format(api_docs = api_docs,input_param_docs=input_param_docs(api)))
    sim = cosine_similarity(api_docs_emb,model_ret.encode(query))
    query_history = []
    cnt=0
    while sim>0.5:
        cnt+=1
        if cnt >10:
            return False
        query_history.append((query,sim))
        query = Dial(vague_refine_prompt.format(api_docs=api_docs,query=query,sim_score=sim,history=query_history[-5:],input_param_docs=input_param_docs(api)))
        sim = cosine_similarity(api_docs_emb,model_ret.encode(query))
    return query


def clear_add_query_generator(api,input_param,value):
    api_docs = api_list_dict[api]['description']
    api_docs_emb = model_ret.encode(api_docs)
    query = Dial(clear_add_prompt.format(api_docs = api_docs,input_param=input_param,value=value,input_param_docs=input_param_docs(api)))
    sim = cosine_similarity(api_docs_emb,model_ret.encode(query))
    query_history = []
    cnt = 0
    while sim<0.6:
        cnt+=1
        if cnt>10:
            return False
        query_history.append((query,sim))
        query = Dial(clear_add_refine_prompt.format(api_docs=api_docs,query=query,sim_score=sim,history=query_history[-5:],input_param=input_param,value=value,input_param_docs=input_param_docs))
        sim = cosine_similarity(api_docs_emb,model_ret.encode(query))
    return query


########################

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
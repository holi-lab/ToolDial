import json,re,requests,os,random,time,sys
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")

def normalize_param(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', text)
    return cleaned_text.lower()

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

def is_ben(keyword):
    ben_list = ['status',"message","messages"]
    for ben in ben_list:
        if ben in keyword:
            return False
    return True

def list2dict(api_list):
    api_list_cat = set()
    for api in api_list:
        api_list_cat.add(api['category_name'])
    api_list_copy = {cat:{} for cat in api_list_cat}
    for api in api_list:
        if api['tool_name'] not in api_list_copy[api['category_name']]:
            api_list_copy[api['category_name']][api['tool_name']] = {}
    for api in api_list:
        if api['api_name'] not in api_list_copy[api['category_name']][api['tool_name']]:
            api_list_copy[api['category_name']][api['tool_name']][api['api_name']] = api
    return api_list_copy

def dict2list(api_list):
    api_list_copy = []
    for cat in api_list:
        for tool in api_list[cat]:
            for apiname in api_list[cat][tool]:
                api_list_copy.append(api_list[cat][tool][apiname])
    return api_list_copy

def dict_sort(current_dict):
    return dict(sorted(current_dict.items(), key=lambda item: item[1], reverse=True))

def word_overlap(text1,text2):
    word_list1 = set(text1.split(" "))
    word_list2 = set(text2.split(" "))
    overlap_word = word_list1&word_list2
    avg_len = (len(word_list1)+len(word_list2))/2
    return len(overlap_word)/avg_len



with open("after_coor_before_emb.json",'r') as f:
    api_list = json.load(f)
    
print(f"Length of API: {len(api_list)}")

embedding_to_save=np.load("embedding_compressed_with_key.npz",allow_pickle = True)

for api in api_list:
    current_embedding = embedding_to_save[api['full_name']].reshape(-1)[0]
    if "req_input_emb" in current_embedding:
        api['req_input_emb'] = current_embedding['req_input_emb']
    if "opt_input_emb" in current_embedding:
        api['opt_input_emb'] = current_embedding['opt_input_emb']
    if "output_emb" in current_embedding:
        api['output_emb'] = current_embedding['output_emb']

for api in api_list:
    if "req_input_emb" in api:
        for req_idx,req_param in enumerate(api['required_parameters']):
            embedding = deepcopy(api['req_input_emb'][req_idx])
            api['required_parameters'][req_idx]['embedding'] = embedding
        del api['req_input_emb']
    if "opt_input_emb" in api:
        for opt_idx,opt_param in enumerate(api['optional_parameters']):
            embedding = deepcopy(api['opt_input_emb'][opt_idx])
            api['optional_parameters'][opt_idx]['embedding'] = embedding
        del api['opt_input_emb']
    if "output_emb" in api:
        for output_idx,output_param in enumerate(api['output_components']):
            embedding = deepcopy(api['output_emb'][output_idx])
            api['output_components'][output_idx]['embedding'] = embedding
        del api['output_emb']
        
embedding_to_save=np.load("embedding_compressed_object_long.npz",allow_pickle = True)

for api in api_list:
    current_embedding = embedding_to_save[api['full_name']].reshape(-1)[0]
    if "req_input_emb_obj" in current_embedding:
        api['req_input_emb_obj'] = current_embedding['req_input_emb_obj']
    if "opt_input_emb_obj" in current_embedding:
        api['opt_input_emb_obj'] = current_embedding['opt_input_emb_obj']
    if "output_emb_obj" in current_embedding:
        api['output_emb_obj'] = current_embedding['output_emb_obj']

for api in api_list:
    if "req_input_emb_obj" in api:
        for req_idx,req_param in enumerate(api['required_parameters']):
            embedding = deepcopy(api['req_input_emb_obj'][req_idx])
            api['required_parameters'][req_idx]['obj_embedding'] = embedding
        del api['req_input_emb_obj']
    if "opt_input_emb_obj" in api:
        for opt_idx,opt_param in enumerate(api['optional_parameters']):
            embedding = deepcopy(api['opt_input_emb_obj'][opt_idx])
            api['optional_parameters'][opt_idx]['obj_embedding'] = embedding
        del api['opt_input_emb_obj']
    if "output_emb_obj" in api:
        for output_idx,output_param in enumerate(api['output_components']):
            embedding = deepcopy(api['output_emb_obj'][output_idx])
            api['output_components'][output_idx]['obj_embedding'] = embedding
        del api['output_emb_obj']
        
api_list_dict = {api['full_name']:api for api in api_list}

input_o = {}
input_x = {}
        
for api in api_list:
    if len(api['required_parameters']) == 0:
        input_x[api['full_name']] = api
    else:
        input_o[api['full_name']] = api

graph_row_big = []
for api in api_list:
    for output_comp in api['output_components']:
        graph_row_big.append({"full_name":api['full_name'],
                            "comp_name":output_comp['name'],
                            "object_ad":output_comp['object_ad'],
                            "object_long":output_comp['object_long']})
        
graph_row_big = [output_comp for output_comp in graph_row_big if is_ben(output_comp['comp_name']) and is_ben(output_comp['object_ad']) and is_ben(output_comp['object_long'])]
graph_col_big = list()

for api in input_o:
    for param in input_o[api]['required_parameters']:
        graph_col_big.append({"full_name":api,
                              "param_name":param['name'],
                              "object_ad":param['object_ad'],
                              "object_long":param['object_long'],
                              "req_or_opt":"required_parameters"})
print("Row size:",len(graph_row_big))
print("Col size:",len(graph_col_big))

edge_pair_graph_big = [["" for _ in range(len(graph_col_big))] for __ in range(len(graph_row_big))]

emb_size = 768

print("Constructing Description similarity matrix...")
col_matrix = np.zeros((emb_size,len(graph_col_big)))
for col_idx,col_api in enumerate(tqdm(graph_col_big)):
    for param_idx,param in enumerate(api_list_dict[col_api['full_name']][col_api['req_or_opt']]):
        if param['name'] == col_api['param_name']:
            col_matrix[:,col_idx] = api_list_dict[col_api['full_name']][col_api['req_or_opt']][param_idx]['embedding']
            break

row_matrix = np.zeros((len(graph_row_big),emb_size))
for row_idx,row_api in enumerate(tqdm(graph_row_big)):
    for param_idx,param in enumerate(api_list_dict[row_api['full_name']]["output_components"]):
        if param['name'] == row_api["comp_name"]:
            row_matrix[row_idx,:] = api_list_dict[row_api['full_name']]["output_components"][param_idx]['embedding']
            break
cos_graph = row_matrix.dot(col_matrix)
norm_graph = np.linalg.norm(row_matrix,axis=1).reshape(-1,1).dot(np.linalg.norm(col_matrix,axis=0).reshape(1,-1))
cos_graph = cos_graph/norm_graph
print("Done!\n")

print("Constructing Object similarity matrix...")
col_matrix = np.zeros((emb_size,len(graph_col_big)))
for col_idx,col_api in enumerate(tqdm(graph_col_big)):
    for param_idx,param in enumerate(api_list_dict[col_api['full_name']][col_api['req_or_opt']]):
        if param['name'] == col_api['param_name']:
            col_matrix[:,col_idx] = api_list_dict[col_api['full_name']][col_api['req_or_opt']][param_idx]['obj_embedding']
            break

row_matrix = np.zeros((len(graph_row_big),emb_size))
for row_idx,row_api in enumerate(tqdm(graph_row_big)):
    for param_idx,param in enumerate(api_list_dict[row_api['full_name']]["output_components"]):
        if param['name'] == row_api["comp_name"]:
            row_matrix[row_idx,:] = api_list_dict[row_api['full_name']]["output_components"][param_idx]['obj_embedding']
            break
cos_obj_graph = row_matrix.dot(col_matrix)
norm_obj_graph = np.linalg.norm(row_matrix,axis=1).reshape(-1,1).dot(np.linalg.norm(col_matrix,axis=0).reshape(1,-1))
cos_obj_graph = cos_obj_graph/norm_obj_graph
print("Done!\n")

print("Shape of cos:",cos_graph.shape)
print("Shape of obj_cos:",cos_obj_graph.shape)

object_graph_s,cos_graph_s,cos_obj_graph_s= float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3])
object_graph = np.load("object_graph.npy") ## object_ad

print("Constructing mask matrix...")
mask_graph = np.ones((len(graph_row_big),len(graph_col_big)))
for row_idx,row_api in enumerate(tqdm(graph_row_big)):
    for col_idx,col_api in enumerate(graph_col_big):
        if col_api['full_name'] == row_api['full_name']:
            mask_graph[row_idx][col_idx] = 0

shape_set = set()
shape_set.add(object_graph.shape)
shape_set.add(cos_graph.shape)
shape_set.add(cos_obj_graph.shape)
if len(shape_set) != 1:
    print("Object_graph shape:",object_graph.shape)
    print("Cos_graph:",cos_graph.shape)
    print("Cos_obj_graph:",cos_obj_graph.shape)
    raise IndexError("Shape of all graphs does not match")
else:
    print("Shape matches!")

graph_big = np.where((object_graph>object_graph_s)&(cos_graph>cos_graph_s)&(cos_obj_graph>cos_obj_graph_s),cos_graph,0) * mask_graph

print("Constructing edge pair list graph...")
for col_idx,col_api in enumerate(tqdm(graph_col_big)):
    for row_idx,row_api in enumerate(graph_row_big):
        if graph_big[row_idx][col_idx] > 0:
            edge_pair_graph_big[row_idx][col_idx] = (row_api['full_name'],
                                                     [row_api['comp_name'],
                                                      row_api['object_ad'],
                                                      row_api['object_long']],
                                                     col_api['full_name'],
                                                     [col_api['param_name'],
                                                      col_api['object_ad'],
                                                      col_api['object_long']])
            # (row_api,[row_api_output,type,object,desc],col_api,[col_api_input,type,object,desc])

print("Removing irrelevent edge(limit, page)...",end="")
for col_idx,col_api in enumerate(tqdm(graph_col_big)):
    for row_idx,row_api in enumerate(graph_row_big):
        if graph_big[row_idx][col_idx]>0:
            if ("page" in normalize_param(col_api['param_name']) or "page" in normalize_param(row_api['comp_name'])) or ("limit" in normalize_param(col_api['param_name']) or "limit" in normalize_param(row_api['comp_name'])):
                graph_big[row_idx][col_idx]=0
                edge_pair_graph_big[row_idx][col_idx] = ""

normal_graph2idx = {node:idx for idx,node in enumerate(api_list_dict)}
normal_idx2igraph = {idx:node for idx,node in enumerate(api_list_dict)}
all_edge_list_graph = [[[] for _ in range(len(normal_graph2idx))] for __ in range(len(normal_graph2idx))]

print("Done!")

print("Normalizing Graph...",end="")
for col_idx,col_api in enumerate(tqdm(graph_col_big)):
    for row_idx,row_api in enumerate(graph_row_big):
        if edge_pair_graph_big[row_idx][col_idx]:
            all_edge_list_graph[normal_graph2idx[graph_row_big[row_idx]['full_name']]][normal_graph2idx[graph_col_big[col_idx]['full_name']]].append(edge_pair_graph_big[row_idx][col_idx])
            
normal_graph = np.zeros((len(all_edge_list_graph),len(all_edge_list_graph)))
cnt = 0
for row_idx in range(len(all_edge_list_graph)):
    for col_idx in range(len(all_edge_list_graph)):
        if all_edge_list_graph[row_idx][col_idx]:
            cnt+=1
            normal_graph[row_idx][col_idx] = 1
print("Done!")
print(f"Total edge in normal_graph:{cnt}")

ind_api = []
for idx in range(len(normal_graph)):
    if sum(normal_graph[:,idx]) == 0 and sum(normal_graph[idx,:])==0:
        ind_api.append(normal_idx2igraph[idx])
print("Independent API ratio:",len(ind_api)/len(normal_graph))

print("Removing Independent APIs...",end="")
manual = set()
survived_node = list(set(list(api_list_dict)) - set(ind_api))
survived_node.sort()

normal_graph2idx = {node:idx for idx,node in enumerate(survived_node)}
normal_idx2igraph = {idx:node for idx,node in enumerate(survived_node)}
all_edge_list_graph = [[[] for _ in range(len(normal_graph2idx))] for __ in range(len(normal_graph2idx))]

for col_idx,col_api in enumerate(tqdm(graph_col_big)):
    for row_idx,row_api in enumerate(graph_row_big):
        if edge_pair_graph_big[row_idx][col_idx]:
            all_edge_list_graph[normal_graph2idx[graph_row_big[row_idx]['full_name']]][normal_graph2idx[graph_col_big[col_idx]['full_name']]].append(edge_pair_graph_big[row_idx][col_idx])

normal_graph = np.zeros((len(all_edge_list_graph),len(all_edge_list_graph)))
cnt = 0
for row_idx in range(len(all_edge_list_graph)):
    for col_idx in range(len(all_edge_list_graph)):
        if all_edge_list_graph[row_idx][col_idx]:
            cnt+=1
            normal_graph[row_idx][col_idx] = 1
            all_edge_list_graph[row_idx][col_idx] = [all_edge_list_graph[row_idx][col_idx][0]]
print("Done!")
print(f"Edge cnt:{cnt}")

file_path = (str(object_graph_s)+"_"+str(cos_graph_s)+"_"+str(cos_obj_graph_s)).replace(".","")

print("Saving File...")
graph_eval_path = f"graph_evaluation/stable_graph_metric/edge_files/graph_for_evaluation/{file_path}"
if file_path not in os.listdir("graph_evaluation/stable_graph_metric/edge_files/graph_for_evaluation"):
    os.mkdir(graph_eval_path)

with open(f"{graph_eval_path}/used_api.json",'w') as f:
    json.dump(survived_node,f,indent=4)

with open(f"{graph_eval_path}/normal_graph2idx.json",'w') as f:
    json.dump(normal_graph2idx,f,indent=4)
    
with open(f"{graph_eval_path}/normal_idx2igraph.json",'w') as f:
    json.dump(normal_idx2igraph,f,indent=4)
    
with open(f"{graph_eval_path}/all_edge_list_graph.json",'w') as f:
    json.dump(all_edge_list_graph,f,indent=4)
    
np.save(f"{graph_eval_path}/normal_graph.npy",normal_graph)

###############
dialogue_generation_path = "dialogue_generation/pair_list"

with open(f"{dialogue_generation_path}/used_api.json",'w') as f:
    json.dump(survived_node,f,indent=4)

with open(f"{dialogue_generation_path}/normal_graph2idx.json",'w') as f:
    json.dump(normal_graph2idx,f,indent=4)
    
with open(f"{dialogue_generation_path}/normal_idx2igraph.json",'w') as f:
    json.dump(normal_idx2igraph,f,indent=4)
    
with open(f"{dialogue_generation_path}/all_edge_list_graph.json",'w') as f:
    json.dump(all_edge_list_graph,f,indent=4)
    
np.save(f"{dialogue_generation_path}/normal_graph.npy",normal_graph)

print("Done!")
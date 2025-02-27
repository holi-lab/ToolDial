import json,re,requests,os,random,time
import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
from copy import deepcopy
import yaml
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


zero_mockdata_prompt = """
You are an intelligent data generator.

API documentation:

- API name: {api_name}
- API description: {api_desc}
- Output format with example values:
{output}

This API doesn't have any input. Just generate the output of this API referring to the API documentation, especially the JSON format of the output.

Rules:
1. Errors never occur.

Return format:
{{"required_parameters": {{}}, "optional_parameters": {{}}, "output_components": <JSON FORMAT>}}

Just return the JSON. Not a single word outside of that.

"""

mockdata_prompt = """
You are an intelligent data generator.

API documentation:

- API name: {api_name}
- API description: {api_desc}
- Required parameters:
{req_params}
- Optional parameters:
{opt_params}
- Output format with example values:
{output}

Refer to the API documentation to generate the virtual input parameters and their corresponding output in the same format.

Rules:
1. Errors never occur.

Return format:
{{"required_parameters": {{<param1>: "abc", <param2>: "def"}}, "optional_parameters": {{<param3>: "fdf"}}, "output_components": <JSON FORMAT>}}
(The names of input parameters are just examples. Make sure to replace them with the actual input parameter names from the API documentation.)
(Please do not include param1, param2, or param3 in the JSON you're creating. Don't even think about adding them.)

Just return the JSON. Not a single word outside of that.
"""

mockdata_with_input_prompt = """
You are an intelligent data generator.

API documentation:

- API name: {api_name}
- API description: {api_desc}
- Required parameters:
{req_params}
- Optional parameters:
{opt_params}
- Output format with example values:
{output}

Input value:
{example_input}

Refer to the API documentation and input value, and generate the output that would be produced when you run the API with the input value.

Rules:
1. Errors never occur.

Return format:
{{"required_parameters": {{<param1>: "abc", <param2>: "def"}}, "optional_parameters": {{<param3>: "fdf"}}, "output_components": <JSON FORMAT>}}
(The names of input parameters are just examples. Make sure to replace them with the actual input parameter names from the API documentation.)
(Please do not include param1, param2, or param3 in the JSON you're creating. Don't even think about adding them.)

Just return the JSON. Not a single word outside of that.
"""

self_refine_prompt = """
This JSON string format has some issues. It can't be transformed into a JSON list using Python's `json.loads`. Fix the issues.
Just return the corrected JSON format as a dictionary. Do not return any additional words.

{json_format}
"""

def extract_outer_braced_text(text):
    pattern = r'\{.*\}'
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    return None

def post_process(response_content):
    if "<<Todo>>" in response_content:
        response_content = response_content.replace("<<Todo>>", "")
    if "```json" in response_content:
        response_content = response_content.replace("```json", "")
    if "```" in response_content:
        response_content = response_content.replace("```", "")
    response_content = response_content.replace("\n","").replace("\t","")
    return extract_outer_braced_text(response_content)

def self_refine(text):
    while True:
        print(f"text:{text}")
        text = post_process(Dial(self_refine_prompt.format(json_format = text)))
        try:
            return json.loads(text)
        except:
            try:
                return eval(text)
            except:
                print(f"{text} can't be json loads")
                continue

def api_call(api_name,input_param = None):
    api_doc = api_list_dict[api_name]
    if not input_param:
        if len(api_list_dict[api_name]['required_parameters']+api_list_dict[api_name]['optional_parameters'])>0:
            prompt = mockdata_prompt.format(
                api_name = api_doc['full_name'],
                api_desc = api_doc['description'],
                req_params = api_doc['required_parameters'],
                opt_params = api_doc['optional_parameters'],
                output = api_doc['output']
            )
        else:
            prompt = zero_mockdata_prompt.format(
                api_name = api_doc['full_name'],
                api_desc = api_doc['description'],
                output = api_doc['output']
            )
    else:
        prompt = mockdata_with_input_prompt.format(
            api_name = api_doc['full_name'],
            api_desc = api_doc['description'],
            req_params = api_doc['required_parameters'],
            opt_params = api_doc['optional_parameters'],
            example_input = input_param,
            output = api_doc['output']
        )
    result = Dial(prompt)
    result = post_process(result)    
    try:
        result = json.loads(result)
    except:
        try:
            result = eval(result)
        except:
            result = self_refine(result)
    new_dict = {"param":result['required_parameters']|result['optional_parameters'],"result":result['output_components']}
    return new_dict

def generate_api_data(
    api_name,
    openai_model='gpt-4o'
):  
    api_dict = api_list_dict
    api_doc = api_dict.get(api_name)

    def format_params(parameters:list):
        if len(parameters) == 0:
            return '{}'
        format_str = '{'
        for idx, param in enumerate(parameters):
            format_str += '"'
            format_str += param['name']
            format_str += '": <<You generate this section>>'            
            if idx + 1 != len(parameters):
                format_str += ', '
        format_str += '}'
        return format_str

    prompt = f"""
You are an intelligent and creative data generator.
I will give you an <<API documentation>> containing the following information below.
tool_name: This is name of tool. Tool is a service provider where provides numerous APIs regarding the service.
tool_description: The description of tool.
api_name: The name of API that belongs to the tool.
api_description: API description.
required_parameters: Required input parameters to call the API. It contains parameter description, type and default value(optional).
optional_parameters: Optional input parameters to call the API. It contains parameter description, type and default value(optional).

<<API documentation>>
tool_name: {api_doc['tool_name']}
tool_description: {api_doc['tool_description']}
api_name: {api_doc['full_name']}
api_description: {api_doc['description']}
required_parameters: {str(api_doc['required_parameters'])}
optional_parameters: {str(api_doc['optional_parameters'])}

<<Instruction>>
1. Refer to the above information and thoroughly understand the API documentation I've provided.
2. Generate parameters in required_parameters and optional_parameters. You can get a hint from default value in required_parameters and optional_parameters when you generate parameter values, but do not just copy the default value. Be creative!
3. However, the data that you generate should not contradict the values that have already been filled in.
For example, if the filled parameter 'max_weight' is set to '20' (interpret this string as a number), then you must generate a value for the parameter 'min_weight' that is less than 20.
**IMPORTANT**
4. You MUST NOT change or hallucinate paramter values that have been filled already in <<Todo>>.
5. ONLY <<You generate this section>> should be replaced with 'plausible' data. The meaning of 'plausible' data is that, for example, if you need to generate a URL, www.example.com is not plausible. Generate something that is likely to exist, even if it doesn't.
6. ONLY generate below <<Todo>> section.

<<Todo>>
```json
{{"required_parameters": {format_params(api_doc['required_parameters'])}, "optional_parameters": {format_params(api_doc['optional_parameters'])}}}
```
"""
    client = OpenAI(api_key=openai_key)

    response = client.chat.completions.create(
        messages=[
            {
                "role":"system",
                "content": prompt
            }
        ],
        model=openai_model,
        temperature=1
    )
    output = response.choices[0].message.content
    if "<<Todo>>" in output:
        output = output.replace("<<Todo>>", "")
    if "```json" in output:
        output = output.replace("```json", "")
    if "```" in output:
        output = output.replace("```", "")
    try:
        return json.loads(output)
    except:
        try:
            return eval(output)
        except:
            return self_refine(output)
    
def normalize_name(api: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '_', api).lower()

def input_generate(api_name):
    if len(api_list_dict[api_name]['required_parameters'])==0 and len(api_list_dict[api_name]['optional_parameters'])==0:
        return {}
    input_data = generate_api_data(api_name)
    req = input_data['required_parameters']
    opt = input_data['optional_parameters']
    req.update(opt)
    return req

from difflib import SequenceMatcher

trg_input_prompt = """
You are an intelligent data generator. Refer to the API documentation and follow the instructions.

API documentation:
- API name: {api_name}
- API description: {api_desc}
- Input parameter list:
{input_param}
- Input parameters with example values:
{input_params_ex}

If you look at the input parameters, all but one slot are blank. Refer to the API documentation to fill in the empty slots with plausible values.
Do not change the value of {chain_param}.

Return format:
{input_params_ex} (With filled empty slots.)

Rules:
1. For coordinate parameters, include both latitude and longitude in the given value of the specified input parameter.

Just return the JSON format as a dictionary. Do not include any additional words.
"""        

class MockDataGenerator:
    def __init__(self,pair):
        self.pair = pair
        self.source,self.target = self.pair['nodes'][0],self.pair['nodes'][1]
        self.edge = self.pair['edge'][0]
        self.chain_input = self.edge['target_attribute']['name']
        self.chain_value = ""
        self.src_input_output = self.generate_src()
        self.trg_input_output = self.generate_trg_output()
        
    def find_coordinates_keys(self,data):
        latitude_keys = ["lat", "latitude", "start_lat", "lat1", "lat2","end_lat"]
        longitude_keys = ["lng", "lon", "long", "longitude", "start_lon", "long1", "long2","lon1","lon2","end_lon","end_long"]
        lat_key,lon_key = None,None
        if isinstance(data, dict):
            for key in data.keys():
                if key in latitude_keys:
                    lat_key = key
                if key in longitude_keys:
                    lon_key = key
                if lat_key and lon_key:
                    break
        return lat_key, lon_key

    def coor_comb(self,data):
        if isinstance(data, dict):
            lat_key, lon_key = self.find_coordinates_keys(data)
            if lat_key and lon_key:
                data['coordinates'] = str([data.pop(lat_key), data.pop(lon_key)])
            for key, value in data.items():
                data[key] = self.coor_comb(value)
        elif isinstance(data, list):
            data = [self.coor_comb(item) for item in data]
            if len(data)==2:
                float_cnt = 0
                for d in data:
                    if type(d) == str:
                        if "." in d:
                            float_cnt+=1
                    elif type(d) == float:
                        float_cnt+=1
                if float_cnt==2:
                    data = str(data)
        return data
    
    def generate_src(self):
        result = api_call(self.source)
        result = self.coor_comb(result)
        return result
    
    def similar(self,a, b):
        return SequenceMatcher(None, a.lower().replace("|",""), b.lower().replace("|","")).ratio()
    
    def find_best_match(self, key, data):
        if isinstance(data, dict):
            for k in data.keys():
                if self.similar(k, key) >= 0.9:
                    return k
        return None

    def search_output(self, data, search_string):
        keys = search_string.split("|")
        keys = [key for key in keys if key != ""]
        current_data = data
        for i, key in enumerate(keys):
            if isinstance(current_data, dict):
                best_match_key = self.find_best_match(key, current_data)
                if best_match_key:
                    current_data = current_data[best_match_key]
                else:
                    return None  # Key not found
            elif isinstance(current_data, list):
                results = [self.search_output(item, "|".join(keys[i:])) for item in current_data]
                results = [res for res in results if res is not None]
                return results[0] if results else None  # Return the first valid result if any
            else:
                return None  # If neither dict nor list, return None

        return current_data
    
    def generate_trg_input(self):
        current_input = input_generate(self.target)
        self.chain_value = self.search_output(self.src_input_output['result']['response'],self.edge['source_attribute']['name'])
        if type(self.chain_value) == list:
            self.chain_value = random.sample(self.chain_value,1)[0]
        for inp in current_input:
            if inp == self.edge['target_attribute']['name']:
                current_input[inp] = self.chain_value
            else:
                current_input[inp] = ""
        if len(current_input) == 1:
            return current_input
        api_doc = api_list_dict[self.target]
        prompt = trg_input_prompt.format(
            api_name = api_doc['full_name'],
            api_desc = api_doc['description'],
            input_param = api_doc['required_parameters']+api_doc['optional_parameters'],
            input_params_ex = current_input,
            chain_param = self.chain_input
        )
        result = Dial(prompt)
        try:
            result = post_process(result)
            return json.loads(result)
        except:
            result = self_refine(result)
        return result
    
    def generate_trg_output(self):
        current_input = self.generate_trg_input()
        return api_call(self.target,current_input)
    
    def return_data(self):
        src_input,trg_input = self.src_input_output['param'],self.trg_input_output['param']
        src_req = [param['name'] for param in api_list_dict[self.source]['required_parameters']]
        src_opt = [param['name'] for param in api_list_dict[self.source]['optional_parameters']]
        trg_req = [param['name'] for param in api_list_dict[self.target]['required_parameters']]
        trg_opt = [param['name'] for param in api_list_dict[self.target]['optional_parameters']]
        src_return = {"api_name":self.source,
                      "api_description":api_list_dict[self.source]['description'],
                      "required_parameters":{},
                      "optional_parameters":{},
                      "output_components":self.src_input_output['result']}
        trg_return={"api_name":self.target,
                    "api_description":api_list_dict[self.target]['description'],
                    "required_parameters":{},
                    "optional_parameters":{},
                    "output_components":self.trg_input_output['result']}
        for param in src_input:
            if param in src_req:
                src_return['required_parameters'][param] = src_input[param]
            elif param in src_opt:
                src_return['optional_parameters'][param] = src_input[param]
        for param in trg_input:
            if param in trg_req:
                trg_return['required_parameters'][param] = trg_input[param]
            elif param in trg_opt:
                trg_return['optional_parameters'][param] = trg_input[param]
        return {"source":src_return,"target":trg_return}

def mock_data_generator(pair,action_seq):
    return_list = {"source":[],"target":[]}
    it=2 if "inform_intent_re" in action_seq else 1
    for i in range(it):
        mockdata_gen = MockDataGenerator(pair)
        tmp_mock = mockdata_gen.return_data()
        return_list['source'].append(tmp_mock['source'])
        return_list['target'].append(tmp_mock['target'])
    return return_list
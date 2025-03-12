import json,re,requests,os,random,time,sys
import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
from copy import deepcopy
import sys
import warnings
warnings.filterwarnings("ignore")
import traceback
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
from mockdata import mock_data_generator
from custom_retriever import Retriever
from query_generation_src import clear_add_query_generator,clear_query_generator,vague_sugg_query_generator,vague_query_generator,situation_generator, vague_sugg_query_generator_trg,clear_query_generator_trg
from scenario_prompt import backward_common_instruction,backward_prompt,aug_ment

req_prompt = "As a required parameter, the system asks {input_param} from the user."
chain_prompt = "The system needs {input_param} as a required parameter, but it can retrieve this input parameter from the previous API call's results: {output_comp}."
opt_prompt = "The system asks for the optional parameter {input_param}. The system's request must include all of the specified input parameters, as every mentioned parameter is necessary. Do not use the word 'or'; use 'and' instead."
fulled_prompt = "The system has already obtained the input parameter {input_param} as {value} from the user's utterance."
easy_prompt = "When the system requests an input parameter, it is important to refer to the description in parentheses as much as possible to clearly convey what the parameter is to the user."
ret_prompt = "The system selected {api} based on the highest retriever score to satisfy the user's query."

user_action_list_backward = { 
    "inform_intent_clear":
        {"description": "Say what one wants specifically.",
         "template": "User requests something from the system. User says '{query}'"},
    "inform_intent_clear_add":
        {"description": "Say what one wants specifically, including input parameter information.",
         "template": "User requests something from the system. User says '{query}'"},
    "affirm":
        {"description": "Agree to the system's proposition.",
         "template": "User agrees with the system's proposition."},
    "negate":
        {"description": "Deny the system's proposal.",
         "template": "User denies the system's proposal."
        },
    "inform_intent_vague":
        {"description": "Say what one wants vaguely.",
         "template": "User requests something from the system. User says '{query}'"},
    "inform":
        {"description": "Provide the requested information to the system.",
         "template": "The user provides the requested information to the system with the exact value from the mock data. In this turn, the user never responds with a short answer but with utterances of 7 words or more. Additionally, the real value in the mock data must appear in the user's utterance.",
         "template_list": "The user answers the system's request with the value of the entity {output_comp} '{value}' from the {api} call results. However, the term {output_comp} '{value}' is not included in the user's utterance."},
    "fail_inform":
        {"description": "Fail to reply to the system's request. The user tells the system they don't know the answer.",
         "template": "The system asks the user for the input parameter {input_param} to call API {api}, but the user fails to provide the input parameter for some reason. In this turn, the user says they don't know the information about the input parameter {input_param} and does not provide any information.",
         "template_tmp": "The user provided information about the input parameter {input_param} to the system (the value of the input parameter must be included in the user's utterance). However, the user could not provide information about {chain_param} for some reason."},
    "inform_intent_re":
        {"description": "Re-request the previous request to the system under different conditions (different parameters).",
         "template": "User re-requests to call {api} with different conditions, providing a different value for the input parameter {input_param} as {value}."},
    "user_bye":
        {"description": "Say thank you and goodbye to the system.",
         "template": "The user says goodbye to the system politely."}
}

system_action_list_backward = {
    "clarify":
        {"description": "If the user's query is vague, re-ask the user to clarify their intent.",
         "template": "As a result of proper API inquiry through the retriever, no API with a score higher than 0.6 could be found. Accordingly, the system requested the user to clarify their intent."},
    "request":
        {"description": "Ask the user for information.",
         "template": "Based on the retriever score, the system believes the results of API {api} can satisfy the user. To call {api}, the system asks the user for the input parameter. In this turn, the system does not provide an example input parameter to the user.",
         "template_list": "The system called API {api_s} to obtain the input parameter {input_param} for API {api_t} and received more than one result. {output_comp} can be used as {input_param} among the results, but the system cannot determine which {output_comp} to select. The system selects {other_output}, or another suitable option, and asks the user to choose one from the {other_output} values. This will be used to select the matching {output_comp}.",
         "template_chain": "Based on the retriever score, the system believes the results '{output_comp}' of API {api_s} can provide the input parameter {chain_param} for API {api_t}. To call {api_s}, the system asks the user for the input parameter. In this turn, the system does not provide an example input parameter to the user.",
         "template_tmp": "The system has gathered information about the input parameter {chain_param} but still needs to collect information about the remaining input parameter {input_param}. The system then asks the user for information about {input_param}.",
         "template_add": "Based on the user's utterance, the system obtains the input parameter {chain_param} as {value}. The system still needs to ask the user for other input parameters."},
    "response":
        {"description": "Reply to the user's request based on the result of the API call.",
         "template": "Based on the result of API {src_api}, {trg_api}, the system responds to the user. The system should aim to fully satisfy the user's request, but must not invent information that isn't in the API results. Only provide details based on the API call.",
         "template_trg":"Based on the result of API {trg_api}, the system responds to the user. The system should aim to fully satisfy the user's request, but must not invent information that isn't in the API results. Only provide details based on the API call."
         },
    "response_fail":
        {"description": "Notify the user that the system cannot execute the request due to insufficient information.",
         "template": "The user did not provide the requested information, so the system informs the user that it cannot resolve their query.",
         "template_deny": "The user did not agree with the system's proposal. The system informs the user that it cannot satisfy their request due to a lack of an appropriate API."},
    "system_bye":
        {"description": "The system says goodbye to the user politely.",
         "template": "If the user greets the system and indicates that all goals have been met, the system responds accordingly."},
    "call":
        {"description": "Call the API with the information collected from the user or elsewhere and do not respond to the user yet.",
         "template": "Based on the user's response or previous API call results, the system gathers the input parameters and calls the API {api}.",
         "template_fail": "The system asks the user for information to fill the input parameter {input_param}, but the user cannot provide it. The system decides to fill it with the output component {output_comp} from API {api}.",
         "template_add": "Based on the user's utterance, the system obtains the input parameter {input_param} as {value}.",
         "template_chain": "Based on the user's utterance, the system believes the results of API {api_s} can provide the input parameter {chain_param} for API {api_t} and calls it."},
    "suggest":
        {"description": "Make a suggestion for an unclear user intent and ask whether it satisfies the user.",
         "template": "Since the user's query is unclear, no API with a retriever score higher than 0.6 has been found. However, several APIs have scores between 0.5 and 0.6. The system asks whether it would be appropriate to run {api}, which has the highest score among them, and retrieve the result. At this time, the system does not mention the name of the API directly."},
    "retriever_call":
        {"description": "Call the retriever to find the proper API to satisfy the user's request.",
         "template": "The system, having received the user's query, calls the retriever to find an appropriate API. In this turn, the system's thought is, 'The user seems to have intent. I will call the retriever.'",
         "template_chain": "The system calls the retriever to find the proper API to retrieve the input parameter {chain_param}."
    }
}

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
                    openai_api_key=openai_key)
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



def count_leaf_keys(json_data, parent_key='', key_count=None):
    if key_count is None:
        key_count = {}
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            full_key = f"{parent_key}|{key}" if parent_key else key
            if (isinstance(value, dict) or isinstance(value, list)) and "coordi" not in key:
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


def is_error_in(pair):
    pair = pair[0]
    for node in pair:
        node_name = normal_idx2igraph[node]
        output_comps = api_list_dict[node_name]['output_components']
        for output in output_comps:
            if "error" in output['name'].lower() or "message" in output['name'].lower():
                return True
    return False

def is_ben(keyword,debug=None): 
    ben_keyword = ["id","coordinate","code","index","identifier"]
    for word in keyword.split():
        for ben in ben_keyword:
            if similarity_score(word,ben)>0.43 or word[-len(ben):].lower() == ben.lower():
                if debug:
                    print(word,ben)
                if len(word)==2:
                    if ben.lower() == word.lower():
                        return False
                    else:
                        return True
                else:
                    return False
    return True

def list2string(param_list,random_param=None):
    string = ""
    for param in param_list:
        if param['name'] != random_param:
            if "description" in param:
                string+=f"{param['name']}({param['description']}),"
            elif "desc" in param:
                string+=f"{param['name']}({param['desc']}),"
    return string[:-1]

def list2string_name(param_list):
    string = ""
    for param in param_list:
        string+=f"{param['name']},"
    return string[:-1]

def list_with_value(param_list,values):
    string = ""
    for param in param_list:
        string+=f"'{param['name']}' as value '{values[param['name']]}',"
    return "("+string[:-1] +")"


def json_search(json_data, hr):
    keys = hr.split('|')

    def recursive_search(data, key_list):
        if not key_list:
            if not isinstance(data, (list, dict)):
                return [data]
            return []

        current_key = key_list[0]
        if isinstance(data, list):
            result = []
            for item in data:
                res = recursive_search(item, key_list)
                result.extend(res)
            return result
        elif isinstance(data, dict):
            if current_key in data:
                return recursive_search(data[current_key], key_list[1:])
        return []
    result = recursive_search(json_data, keys)
    return result


def overlap_value_cnt(api,mockdata):
    json_form = mockdata['output_components']
    is_list = api_list_dict[api]['is_list']
    value_cnt = {}
    for output in is_list:
        current_output = "response|"+output
        value_cnt[output] = json_search(json_form,current_output)
    return {key for key,value in value_cnt.items() if len(value) != len(set(value))}

class PairClass:
    def __init__(self):
        self.common_instruction = backward_common_instruction
        self.prompt = backward_prompt
        self.mockdata_list_prompt= "Although the mock data for API {api} is given only one, this {api} is supposed to return multi datas. Generate at least 4 data when generating the api_response"
        self.prompt_list = []
        self.retriever = retriever
        self.query_cache = {}
        
    def p2h(self,text):
        return text.replace("|","->")
    
    def find_all_indexes(self,tup,value):
        return [index for index, elem in enumerate(tup) if elem == value]
        
    def node_param_extraction(self,subgraph):
        subgraph = subgraph[0]
        edge = all_edge_list_graph[subgraph[0]][subgraph[1]][0]
        source,target = normal_idx2igraph[subgraph[0]],normal_idx2igraph[subgraph[1]]
        source_params = api_list_dict[source]['required_parameters'] + api_list_dict[source]['optional_parameters']
        target_params = api_list_dict[target]['required_parameters'] + api_list_dict[target]['optional_parameters']
        return source,target,source_params,target_params,edge
    
    def relation(self,subgraph):
        edge = subgraph['edge'][0]
        source,target = edge['source'],edge['target']
        output_comp = edge['source_attribute']['name']
        input_param = edge['target_attribute']['name']
        return f"The output component {self.p2h(output_comp)} of API {source} can be used as input parameter {input_param} of API {target}"
    
    def mock_data_prompt(self,subgraph,mock_data,src_param,trg_param):
        mock_data_dict = deepcopy(mock_data)
        prompt = "Use this data as the actual input parameter and output components of the APIs on the dialogue.\n"
        source,target = subgraph['edge'][0]['source'],subgraph['edge'][0]['target']
        for data in mock_data_dict['source']:
            new_opt = {}
            for param in data['optional_parameters']:
                if param == src_param['name']:
                    new_opt[param] = data['optional_parameters'][param]
            data['optional_parameters'] = new_opt
            
            prompt+=f"Data for {source}:\n{data}\n"
            
        prompt+="\n"
        for data in mock_data_dict['target']:
            new_opt_trg = {}
            for param in data['optional_parameters']:
                if param == trg_param['name']:
                    new_opt_trg[param] = data['optional_parameters'][param]
            data['optional_parameters'] = new_opt_trg
            prompt+=f"Data for {target}:\n{data}\n"
        prompt+="\n"
        return prompt
            
    def is_message(self,api):
        for comp in api_list_dict[api]['output_components']:
            if similarity_score(comp['name'],"message")>0.8 or "message" in comp['name'].lower():
                return True
        return False
    

class YesYes(PairClass):
    def __init__(self,subgraph_list,mockdata_list):
        super().__init__()
        self.coor = []
        self.random_idx = False
        self.subgraph_list = subgraph_list
        self.action_seq_o = [ 
            ("inform_intent_clear","retriever_call","request","inform","call","response","user_bye","system_bye"), #
            ("inform_intent_clear_add","retriever_call","request","inform","call","response","user_bye","system_bye"), #
            ("inform_intent_vague","retriever_call","clarify","inform_intent_clear","retriever_call","request","inform","call","response","user_bye","system_bye"), #
            ("inform_intent_vague","retriever_call","suggest","affirm","request","inform","call","response","user_bye","system_bye"), #
            ("inform_intent_vague","retriever_call","suggest","negate","response_fail"), #
            ("inform_intent_vague","retriever_call","clarify","inform_intent_clear_add","retriever_call","request","inform","call","response","user_bye","system_bye") ## add에서 한 개만 말해줌 
        ]
        self.action_seq_x = [
            ("inform_intent_clear","retriever_call","request","fail_inform","retriever_call","request","inform","call","call","response","user_bye","system_bye"), ## chain 빼고는 말해줌 #
            ("inform_intent_clear","retriever_call","request","fail_inform","retriever_call","request","inform","call","request","inform","call","response","user_bye","system_bye"), ## chain을 모른다고만 함 #
            ("inform_intent_vague","retriever_call","clarify","inform_intent_clear","retriever_call","request","fail_inform","retriever_call","request","inform","call","request","inform","call","response","user_bye","system_bye"), ## chain을 모른다고만 함 #
            ("inform_intent_vague","retriever_call","clarify","inform_intent_clear","retriever_call","request","fail_inform","retriever_call","request","inform","call","call","response","user_bye","system_bye"), ## chain 빼고는 말해줌 #
            ("inform_intent_vague","retriever_call","suggest","affirm","request","fail_inform","retriever_call","request","inform","call","request","inform","call","response","user_bye","system_bye"), ## chain을 모른다고만 함 #
            ("inform_intent_vague","retriever_call","suggest","affirm","request","fail_inform","retriever_call","request","inform","call","call","response","user_bye","system_bye"), ## chain 빼고는 말해줌
            ("inform_intent_vague","retriever_call","suggest","negate","response_fail")
        ]
        self.action_seq_x_multi = [ 
            ("inform_intent_clear","retriever_call","request","fail_inform","retriever_call","request","inform","call","request","inform","call","response","user_bye","system_bye"), ## chain 빼고는 말해줌 #
            ("inform_intent_vague","retriever_call","clarify","inform_intent_clear","retriever_call","request","fail_inform","retriever_call","request","inform","call","request","inform","call","response","user_bye","system_bye"), ## chain 빼고는 말해줌 #
            ("inform_intent_vague","retriever_call","suggest","affirm","request","fail_inform","retriever_call","request","inform","call","request","inform","call","response","user_bye","system_bye"), ## chain 빼고는 말해줌 #
            ("inform_intent_vague","retriever_call","suggest","negate","response_fail"), #
        ]
        self.yes_yes_mid_o_list = []
        self.yes_yes_mid_x_list = []
        self.yes_yes_mid_x_multi_list = []
        self.freeform = []
        self.prompt_list = []
        self.scenario_list = []
        self.mockdata_list = []
        self.yes_yes_mid_o()
        self.yes_yes_mid_x()
        self.yes_yes_mid_x_multi()
        
        # with open("coor_check.json",'w') as f:
        #     json.dump(self.coor,f,indent=4)
        
        self.fill_form()
        self.counter()
        # self.sports_ratio()
        self.generate_prompt(sampling=is_sampling)
        
    def sports_ratio(self):
        sports_cnt = 0
        for freeform in self.freeform:
            subgraph = freeform['subgraph']
            cnt=0
            for node in subgraph['nodes']:
                if "Sports" in node:
                    cnt+=1
            if cnt==2:
                sports_cnt+=1
        if len(self.freeform)>0:
            print("Sports ratio",sports_cnt/len(self.freeform))
        
    def graph_aug(self,subgraph):
        dep_idx = subgraph[0][0]
        des_idx = subgraph[0][1]
        current_edge = all_edge_list_graph[dep_idx][des_idx][0]
        chain_output,chain_param = current_edge[1][0],current_edge[3][0]
        return {"subgraph":subgraph,"chain_output":chain_output,"chain_param":chain_param,}
    
    def yes_yes_mid_o(self):
        for subgraph in tqdm(self.subgraph_list):
            source,target,source_params,target_params,edge=self.node_param_extraction(subgraph)
            if len(source_params) == 0 or len(target_params) <=1:
                continue
            edge_param = edge[3][1]
            if not is_ben(edge_param): 
                continue
            self.yes_yes_mid_o_list.append(subgraph)
        print(f"Length of yesyes mid o:{len(self.yes_yes_mid_o_list)}")
        
    def is_output_in(self,chain,output_list):
        for output in output_list:
            if output == chain:
                return True
            elif "coordinate" in output and "coordinate" in chain:
                sim = similarity_object(output,chain)
                if sim >0.95:
                    return True
                else:
                    self.coor.append((output,chain,sim))
        return False
            
    def yes_yes_mid_x(self):
        for subgraph in tqdm(self.subgraph_list):
            source,target,source_params,target_params,edge=self.node_param_extraction(subgraph)
            if len(source_params) == 0 or len(target_params) <= 1:
                continue
            edge_output = {"name":edge[1][0],"object_ad":edge[1][1],"object_long":edge[1][2]}
            edge_param = {"name":edge[3][0],"object_ad":edge[3][1],"object_long":edge[3][2]}
            is_same,know_stack = False,0
            for param in source_params:
                # if similarity_object(param['object_ad'],edge_param_ad)>0.83 or similarity_object(param['object_ad'],edge_output_ad)>0.83:
                #     is_same = True
                if self.is_same(param,edge_param) or self.is_same(param,edge_output):
                    is_same=True
                    continue
                if is_ben(param['object_ad']):
                    know_stack+=1
            if is_same:
                continue
            if not is_ben(edge_param['object_ad']):
                # if edge_output['name'] not in api_list_dict[source]['is_list']:
                if not self.is_output_in(edge_output['name'],api_list_dict[source]['is_list']):
                    self.yes_yes_mid_x_list.append(subgraph)
        print(f"Length of yesyes mid x:{len(self.yes_yes_mid_x_list)}")
                
    def yes_yes_mid_x_multi(self):
        for subgraph in tqdm(self.subgraph_list):
            source,target,source_params,target_params,edge=self.node_param_extraction(subgraph)
            if len(source_params) == 0 or len(target_params) <= 1:
                continue     
            edge_output = {"name":edge[1][0],"object_ad":edge[1][1],"object_long":edge[1][2]}
            edge_param = {"name":edge[3][0],"object_ad":edge[3][1],"object_long":edge[3][2]}
            is_same,know_stack = False,0
            for param in source_params:
                # if similarity_object(param['object_ad'],edge_param_ad)>0.83 or similarity_object(param['object_ad'],edge_output_ad)>0.83:
                #     is_same = True
                if self.is_same(param,edge_param) or self.is_same(param,edge_output):
                    is_same = True
                    continue
                if is_ben(param['object_ad']):
                    know_stack+=1
            if is_same:
                continue
            if not is_ben(edge_param['object_ad']): 
                if self.is_output_in(edge_output['name'],api_list_dict[source]['is_list']):
                    self.yes_yes_mid_x_multi_list.append(subgraph)
        print(f"Length of yesyes mid x multi:{len(self.yes_yes_mid_x_multi_list)}")
        
    def is_same(self,param,edge):
        if similarity_object(param['object_ad'],edge['object_ad'])>0.83:
            if cosine_similarity(model.encode(param['object_long']),model.encode(edge['object_long']))>0.77:
                return True
        if normalize_param(param['name']) == normalize_param(edge['name']):
            return True
                
    def fill_form(self):
        for subgraph in self.yes_yes_mid_o_list:
            subgraph = freeformat(self.graph_aug(subgraph))
            subgraph['scenario'] = self.yes_yes_mid_o.__name__
            for action in self.action_seq_o:
                self.freeform.append({"subgraph":subgraph,"action":action})
        for subgraph in self.yes_yes_mid_x_list:
            subgraph = freeformat(self.graph_aug(subgraph))
            subgraph['scenario'] = self.yes_yes_mid_x.__name__
            for action in self.action_seq_x:
                self.freeform.append({"subgraph":subgraph,"action":action})
        for subgraph in self.yes_yes_mid_x_multi_list:
            subgraph = freeformat(self.graph_aug(subgraph))
            subgraph['scenario'] = self.yes_yes_mid_x_multi.__name__
            for action in self.action_seq_x_multi:
                self.freeform.append({"subgraph":subgraph,"action":action})
        self.freeform = [form for form in self.freeform if (not self.is_message(form['subgraph']['nodes'][0])) and (not self.is_message(form['subgraph']['nodes'][1]))]
        
    def ret_status(self,scenario):
        ret_list = []
        for turn in scenario.split("\n\n"):
            if "User turn\n" in turn:
                continue
            turn = turn.split("\n")
            for comp in turn:
                if "Retriever status: " in comp:
                    ret_list.append(eval(comp.replace("Retriever status: ","")))
        return ret_list
    
    def counter(self):
        cnt=0
        for idx,subgraph in enumerate(self.freeform):
            source,target = subgraph['subgraph']['nodes'][0],subgraph['subgraph']['nodes'][1]
            trg_opt_list = [{"name":param['name'],"description":param['object_long']} for param in api_list_dict[target]['optional_parameters'] if param['name'] != subgraph['subgraph']['chain_param']]
            if len(trg_opt_list)==0:
                trg_opt_list.append(1)
            src_opt_list = [{"name":param['name'],"description":param['object_long']} for param in api_list_dict[source]['optional_parameters'] if param['name'] != subgraph['subgraph']['chain_param']]
            if len(src_opt_list)==0 :
                src_opt_list.append(1)
            if subgraph['subgraph']['scenario'] == "yes_yes_mid_o":
                len_src_opt_list = 1
            else:
                len_src_opt_list = len(src_opt_list)
            cnt+=len_src_opt_list*len(trg_opt_list)
        print("-----------------------")
        print(f"Total scenario cnt: {cnt}")
        print("-----------------------")
        
    def sample_action_seq(self):
        print("Sampling start...",end="")
        total_list = []
        action_seq_type_list = [self.action_seq_o,self.action_seq_x,self.action_seq_x_multi]
        scen_type_list = [self.yes_yes_mid_o.__name__,self.yes_yes_mid_x.__name__,self.yes_yes_mid_x_multi.__name__]
        random.seed(1)
        for action_seq_type,scen_type in zip(action_seq_type_list,scen_type_list):
            for action_seq in action_seq_type:
                candidate = [subgraph for subgraph in self.freeform if subgraph['action'] == action_seq and subgraph['subgraph']['scenario'] == scen_type]
                if len(candidate)<1:
                    continue
                total_list+=random.sample(candidate,1)
        print("Done!")
        return total_list
                
    def generate_prompt(self,sampling=False):
        if sampling:
            freeform_list = self.sample_action_seq()
        else:
            freeform_list = self.freeform
        for idx,subgraph in enumerate(tqdm(freeform_list)):
            out_break = False
            source,target = subgraph['subgraph']['nodes'][0],subgraph['subgraph']['nodes'][1]
            
            if target == "Mapping|boundaries_io|Query by Radius(in miles) .for Zipcode boundaries":
                continue
            
            situation = situation_generator([(normal_graph2idx[source],normal_graph2idx[target])])
            ##optional parameters##
            trg_opt_list = [{"name":param['name'],"description":param['object_long']} for param in api_list_dict[target]['optional_parameters'] if param['name'] != subgraph['subgraph']['chain_param']]
            if len(trg_opt_list)==0:
                trg_opt_list.append(False)
            src_opt_list = [{"name":param['name'],"description":param['object_long']} for param in api_list_dict[source]['optional_parameters']]
            if len(src_opt_list)==0:
                src_opt_list.append(False)
            ##end##
            try:
                mock_data = mock_data_generator(subgraph['subgraph'],subgraph['action'])
            except:
                continue
            original_subgraph = deepcopy(subgraph)
            for src_opt in src_opt_list:
                for trg_opt in trg_opt_list:
                    subgraph = deepcopy(original_subgraph)
                    scenario = self.scenario_generation(subgraph['subgraph'],subgraph['action'],mock_data,src_opt,trg_opt,situation)
                    if scenario == False:
                        out_break = True
                        break
                    dst = self.dst_generation(subgraph['subgraph'],subgraph['action'],mock_data,src_opt,trg_opt)
                    dst_idx = 0
                    for turn_idx,turn in enumerate(scenario):
                        if "System turn\n" in turn:
                            scenario[turn_idx]+=f"\nDialogue state: {dst[dst_idx]}"
                            dst_idx+=1
                    scenario = "\n\n".join(scenario)
                    ret = self.ret_status(scenario)
                    scen = {"scenario":scenario,"dst":dst,"ret":ret,"action_seq":subgraph['action'],"subgraph":subgraph['subgraph'],"mock_data":mock_data}
                    self.scenario_list.append(scen)
                    relation = self.relation(scen['subgraph'])
                    tmp_prompt = self.prompt.format(common_instruction=self.common_instruction,relation=relation,scenario=scen['scenario'],data=self.mock_data_prompt(scen['subgraph'],scen['mock_data'],src_opt,trg_opt))
                    self.prompt_list.append(tmp_prompt)
                    self.mockdata_list.append(mock_data)
                    if sampling:
                        out_break=True
                        break
                if out_break:
                    break
                
    def mockdata2value(self,mockdata):
        data_dict = {}
        for idx,data in enumerate(mockdata):
            data_dict[f'data_{idx+1}'] = {}
            params = data['required_parameters']|data['optional_parameters']
            for key in params:
                if type(params[key])==list and "coordi" not in key:
                    self.random_idx = random.sample(list(range(0,len(params[key]))),1)[0]
                    data_dict[f'data_{idx+1}'][key] = params[key][self.random_idx]
                else:    
                    data_dict[f'data_{idx+1}'][key] = params[key]
        return data_dict
    
    def retriever_generation(self,query,api,vague=False,suggest=False):
        ret_list = self.retriever.search_api(query,api,200,vague=vague,suggest=suggest)
        return {"retriever_call":"true","retrieved_api":{name:score for name,score in zip(ret_list['list'],ret_list['score'])}}
    
    def dst_generation(self,subgraph,action_seq,mock_data,src_opt,trg_opt):
        stage,source,target = "target",subgraph['edge'][0]['source'],subgraph['edge'][0]['target']
        current_api,current_data = target,self.mockdata2value(deepcopy(mock_data['target']))
        default_dst,slot_dst = {"api_confirmed":"false","api_status":"none"},{"api_confirmed":"true","api_status":{"api_name":"","input_parameter":{},"system_action":"stand_by"}}
        dst = []
        for action_idx,action in enumerate(action_seq): 
            if action in user_action_list_backward:
                if action == "fail_inform":
                    stage,current_api,current_data = "source",source,self.mockdata2value(deepcopy(mock_data['source']))
            else:
                if action == "retriever_call":
                    tmp_dst = deepcopy(default_dst)
                    dst.append(tmp_dst)
                if action == "request":
                    current_opt = trg_opt if stage == "target" else src_opt
                    tmp_dst = deepcopy(slot_dst)
                    tmp_dst['api_status']['api_name'] = current_api
                    tmp_dst['api_status']['system_action'] = "stand_by"
                    req_name = [param['name'] for param in api_list_dict[current_api]['required_parameters']]
                    for param in current_data['data_1']:
                        if param in req_name:
                            tmp_dst['api_status']['input_parameter'][param] = ""
                        if current_opt:
                            if param == current_opt['name']:
                                tmp_dst['api_status']['input_parameter'][param] = ""
                        if stage == "target" and param == subgraph['chain_param']:
                            tmp_dst['api_status']['input_parameter'][param] = ""
                    ## list or not ##
                    if action_seq.count('request') == 3 and "multi" not in subgraph['scenario']: ## not list
                        request_idx = self.find_all_indexes(action_seq,"request")
                        if action_idx == request_idx[-1]:
                            tmp_dst['api_status']['input_parameter'][subgraph['chain_param']] = current_data['data_1'][subgraph['chain_param']]
                    elif action_seq.count('request') == 3 and "multi" in subgraph['scenario']: ## list
                        request_idx = self.find_all_indexes(action_seq,"request")
                        if action_idx == request_idx[-1]:
                            for param in current_data['data_1']:
                                if param in req_name:
                                    tmp_dst['api_status']['input_parameter'][param] = current_data['data_1'][param]
                                if trg_opt:
                                    if param == trg_opt['name']:
                                        tmp_dst['api_status']['input_parameter'][param] = current_data['data_1'][param]
                            tmp_dst['api_status']['input_parameter'][subgraph['chain_param']] = ""
                    
                    if 'inform_intent_clear_add' in action_seq and stage == "target":
                        tmp_dst['api_status']['input_parameter'][subgraph['chain_param']] = current_data['data_1'][subgraph['chain_param']]
                                
                    dst.append(tmp_dst)
                if action == "call":
                    if stage == "source":
                        tmp_dst = deepcopy(slot_dst)
                        tmp_dst['api_status']['api_name'] = current_api
                        tmp_dst['api_status']['system_action'] = "call"
                        req_name = [param['name'] for param in api_list_dict[current_api]['required_parameters']]
                        for param in current_data['data_1']:
                            if param in req_name:
                                tmp_dst['api_status']['input_parameter'][param] = current_data['data_1'][param]
                            if src_opt:
                                if param == src_opt['name']:
                                    tmp_dst['api_status']['input_parameter'][param] = current_data['data_1'][param]
                        stage,current_api,current_data = "target",target,self.mockdata2value(deepcopy(mock_data['target']))
                        dst.append(tmp_dst)
                    elif stage == "target":
                        tmp_dst = deepcopy(slot_dst)
                        tmp_dst['api_status']['api_name'] = current_api
                        tmp_dst['api_status']['system_action'] = "call"
                        req_name = [param['name'] for param in api_list_dict[current_api]['required_parameters']]
                        for param in current_data['data_1']:
                            if param in req_name:
                                tmp_dst['api_status']['input_parameter'][param] = current_data['data_1'][param]
                            if trg_opt:
                                if param == trg_opt['name']:
                                    tmp_dst['api_status']['input_parameter'][param] = current_data['data_1'][param]
                        tmp_dst['api_status']['input_parameter'][subgraph['chain_param']] = current_data['data_1'][subgraph['chain_param']]
                        dst.append(tmp_dst)
                if action == "response":
                    tmp_dst = deepcopy(slot_dst)
                    tmp_dst['api_status']['api_name'] = current_api
                    tmp_dst['api_status']['system_action'] = "respond"
                    req_name = [param['name'] for param in api_list_dict[current_api]['required_parameters']]
                    for param in current_data['data_1']:
                        if param in req_name:
                            tmp_dst['api_status']['input_parameter'][param] = current_data['data_1'][param]
                        if trg_opt:
                            if param == trg_opt['name']:
                                tmp_dst['api_status']['input_parameter'][param] = current_data['data_1'][param]
                    tmp_dst['api_status']['input_parameter'][subgraph['chain_param']] = current_data['data_1'][subgraph['chain_param']]
                    dst.append(tmp_dst)
                if action == "system_bye":
                    tmp_dst = deepcopy(default_dst)
                    dst.append(tmp_dst)
                if action == "suggest":
                    tmp_dst = deepcopy(default_dst)
                    dst.append(tmp_dst)
                if action == "clarify":
                    tmp_dst = deepcopy(default_dst)
                    dst.append(tmp_dst)
                if action == "response_fail":
                    tmp_dst = deepcopy(default_dst)
                    dst.append(tmp_dst)
        return dst
            
    def scenario_generation(self,subgraph,action_seq,mock_data,src_opt,trg_opt,situation):
        prompt=[]
        stage,source,target = "target",subgraph['edge'][0]['source'],subgraph['edge'][0]['target']
        current_api,current_data = target, self.mockdata2value(deepcopy(mock_data['target']))
        
        is_list = False
        default_ret = {'retriever_call':'false','retrieved_api':"none"}
        current_ret = False
        call_stack = 0
        for idx,action in enumerate(action_seq):
            if action in user_action_list_backward:
                if action == "inform_intent_clear":
                    if f"{source}{target}{action}" in self.query_cache:
                        query = self.query_cache[f"{source}{target}{action}"]
                    else:
                        query = clear_query_generator(source,target,situation)
                        self.query_cache[f"{source}{target}{action}"] = query
                    if query == False:
                        return False
                    tmp_situation = user_action_list_backward[action]['template'].format(query = query)
                    current_ret = self.retriever_generation(query,target)
                    if len(current_ret['retrieved_api']) == 0:
                        return False
                    
                if action == "inform_intent_clear_add":
                    value = current_data['data_1'][subgraph['chain_param']]
                    
                    if f"{source}{target}{action}" in self.query_cache:
                        query = self.query_cache[f"{source}{target}{action}"]
                    else:
                        query = clear_add_query_generator(target,subgraph['chain_param'],value)
                        self.query_cache[f"{source}{target}{action}"] = query
                    if query == False:
                        return False
                    tmp_situation = user_action_list_backward[action]['template'].format(query=query)
                    current_ret = self.retriever_generation(query,target)
                if action == "inform_intent_vague":
                    if action_seq[idx+2] == "suggest":
                        if f"{source}{target}{action}_suggest" in self.query_cache:
                            query = self.query_cache[f"{source}{target}{action}_suggest"]
                        else:
                            query = vague_sugg_query_generator(source,target,situation)
                            self.query_cache[f"{source}{target}{action}_suggest"] = query
                        if query == False:
                            return False
                        tmp_situation = user_action_list_backward[action]['template'].format(query=query)
                        current_ret = self.retriever_generation(query,target,vague=True,suggest=True)
                    else:
                        
                        if f"{source}{target}{action}" in self.query_cache:
                            query = self.query_cache[f"{source}{target}{action}"]
                        else:
                            query = vague_query_generator(target)
                            self.query_cache[f"{source}{target}{action}"] = query
                        
                        if query == False:
                            return False
                        tmp_situation = user_action_list_backward[action]['template'].format(query=query)
                        current_ret = self.retriever_generation(query,target,vague=True)
                if action == "inform":
                    if is_list:
                        value = self.mockdata2value(deepcopy(mock_data['target']))['data_1'][subgraph['chain_param']]
                        tmp_situation = user_action_list_backward[action]["template_list"].format(api=source,output_comp=self.p2h(subgraph['chain_output']),value=value)
                        is_list = False
                    else:
                        tmp_situation = user_action_list_backward[action]['template']
                        answer_values = self.mockdata2value(deepcopy(mock_data[stage]))['data_1']
                        new_answer_values = {}
                        req_list_copy = deepcopy(subgraph[stage+"_req"])
                        req_list = [param['name'] for param in req_list_copy]
                        for param in answer_values:
                            if param in req_list:
                                new_answer_values[param] = answer_values[param]
                                continue
                            if stage == "source":
                                if src_opt:
                                    if param == src_opt['name']:
                                        new_answer_values[param] = answer_values[param]
                            elif stage == "target":
                                if trg_opt:
                                    if param == trg_opt['name']:
                                        new_answer_values[param] = answer_values[param]
                        
                        tmp_situation+=f" User answers only with the values in here:{new_answer_values}(Answer only the value asked by the system)"
                if action == "fail_inform":
                    if action_seq.count("request")==3 and "multi" not in subgraph['scenario']:
                        tmp_situation = user_action_list_backward[action]['template'].format(api=current_api,input_param = subgraph['chain_param'])
                    else:
                        param_list = deepcopy(subgraph[stage+"_req"])
                        param_name = [param for param in param_list if param['name'] != subgraph['chain_param']]
                        if trg_opt:
                            param_name+=[trg_opt]
                        values = self.mockdata2value(deepcopy(mock_data['target']))['data_1']
                        tmp_situation = user_action_list_backward[action]['template_tmp'].format(input_param = list_with_value(param_name,values),chain_param = subgraph['chain_param'])
                    stage,current_api,current_data = "source",source,self.mockdata2value(deepcopy(mock_data['source']))
                    current_ret = {'retriever_call': 'true', 'retrieved_api': [(source,f"Output to procure input parameter {subgraph['chain_param']} of {target}: {self.p2h(subgraph['chain_output'])}")]}
                if action == "user_bye":
                    tmp_situation = user_action_list_backward[action]['template']
                if action == "affirm":
                    tmp_situation = user_action_list_backward[action]['template']
                if action == "negate":
                    tmp_situation = user_action_list_backward[action]['template']
                if action in aug_ment and random.sample([0,0,1,1,1],1)[0]:
                    tmp_situation+=f"User's speech should follow this format: {random.sample(aug_ment[action],1)[0]}"
                tmp_prompt = f"User turn\n-user action: {action}({user_action_list_backward[action]['description']})\n-situation:{tmp_situation}"
                prompt.append(tmp_prompt)
            else:
                if action == "request":
                    request_idx = self.find_all_indexes(action_seq,"request")
                    if action_seq[idx-1]=="call":
                        if "multi" in subgraph['scenario'] and stage == "target" and idx == request_idx[2]:
                            is_list=True
                            output_list = [output.strip("|") for output in api_list_dict[source]['is_list']]
                            for idx,output in enumerate(output_list):
                                output_list[idx] = output.replace("coordinates","coordinate")
                            chain_output = subgraph['chain_output'].strip("|")
                            if "coordinate" in chain_output:
                                chain_output = chain_output.replace("coordinates","coordinate")
                            output_list = set(output_list) - set([chain_output])
                            output_list = [output for output in output_list if output != "status" or output != "message"]
                            
                            try:
                                output_list = list(set(output_list) - overlap_value_cnt(source,mock_data['source'][0]))
                            except:
                                print(mock_data)
                                return False
                            if len(output_list)<1:
                                print("AA")
                                return False
                            
                            other = random.sample(output_list,1)[0]
                            other = self.p2h(other)
                            tmp_situation = system_action_list_backward[action]['template_list'].format(
                                api_t=target,api_s=source,output_comp=self.p2h(subgraph['chain_output']),input_param=subgraph['chain_param'],other_output = other)
                        else:
                            if idx == request_idx[2]:
                                stage,current_api,current_data = "target",target,self.mockdata2value(deepcopy(mock_data['target']))
                                param_list = deepcopy(subgraph[stage+"_req"])
                                param_name = [param for param in param_list if param['name'] != subgraph['chain_param']]
                                if trg_opt:
                                    param_name+=[trg_opt]
                                tmp_situation = system_action_list_backward[action]["template_tmp"].format(chain_param = subgraph['chain_param'],input_param=[list2string(param_name)])
                        
                    elif action_seq[idx-2] == "fail_inform":
                        tmp_situation = system_action_list_backward[action]['template_chain'].format(api_s=source,
                                                                                                     api_t=target,
                                                                                                     chain_param=subgraph['chain_param'],
                                                                                                     output_comp = self.p2h(subgraph['chain_output']))
                        req_list,opt_list = deepcopy(subgraph[stage+'_req']),deepcopy(subgraph[stage+"_opt"])
                        if len(req_list)>0:
                            tmp_situation+=req_prompt.format(input_param = list2string(req_list))
                        if src_opt:
                            ask_opt = []
                            for opt in opt_list:
                                if opt['name'] == src_opt['name']:
                                    ask_opt.append(opt)
                            tmp_situation+=opt_prompt.format(input_param = list2string(ask_opt))
                    elif "inform_intent_clear_add" in action_seq:
                        value = current_data['data_1'][subgraph['chain_param']]
                        tmp_situation = system_action_list_backward[action]['template_add'].format(chain_param = subgraph['chain_param'],value=value)
                        req_list_copy = deepcopy(subgraph[stage+"_req"])
                        req_list = [param for param in req_list_copy if param['name'] != subgraph['chain_param']]
                        if len(req_list)>0:
                            tmp_situation+=req_prompt.format(input_param = list2string(req_list))
                        if trg_opt:
                            opt_list_copy = deepcopy(subgraph[stage+"_opt"])
                            opt_list = [param for param in opt_list_copy if param['name'] == trg_opt['name']]
                            tmp_situation+=opt_prompt.format(input_param = list2string(opt_list))
                    else:
                        tmp_situation = system_action_list_backward[action]['template'].format(api=current_api)
                        tmp_situation = ret_prompt.format(api=current_api) + tmp_situation
                        req_list,opt_list = deepcopy(subgraph[stage+'_req']),deepcopy(subgraph[stage+"_opt"])
                        if len(req_list)>0:
                            tmp_situation+=req_prompt.format(input_param = list2string(req_list))
                        if trg_opt:
                            ask_opt = []
                            for opt in opt_list:
                                if opt['name'] == trg_opt['name'] or opt['name'] == subgraph['chain_param']:
                                    ask_opt.append(opt)
                            tmp_situation+=opt_prompt.format(input_param = list2string(ask_opt))
                elif action == "call":
                    if action_seq[idx-1] == "inform_intent_clear_add":
                        value = current_data['data_1'][subgraph['chain_param']]
                        tmp_situation = system_action_list_backward[action]['template_add'].format(api=current_api,input_param = subgraph['chain_param'],value=value)
                    else:
                        tmp_situation = system_action_list_backward[action]['template'].format(api=current_api)
                        if stage == "source":
                            tmp_situation+=" Input parameter source: From the user's utterence of previous turn."
                        elif "fail_inform" in action_seq:
                            tmp_situation+=f"Input parameter source: From previous API call results({subgraph['chain_param']})."
                            req_list_copy = deepcopy(subgraph[stage+"_req"])
                            req_list = [param for param in req_list_copy if param['name'] != subgraph['chain_param']]
                            opt_list_copy = deepcopy(subgraph[stage+"_opt"])
                            opt_list = [param for param in opt_list_copy]
                            if trg_opt:
                                for opt in opt_list:
                                    if opt['name'] == trg_opt['name']:
                                        req_list.append(opt)
                            if len(req_list)>0:
                                if action_seq[idx-1] == "call" or "multi" in subgraph['scenario']:
                                    tmp_situation+=f" From dialogue history({list2string_name(req_list)})."
                                else:
                                    tmp_situation+=f" From the user's utterence of previous turn({list2string_name(req_list)})."
                            tmp_situation+="(The process of referring to where the input parameter source is should be specified in text in the thought of the turn)."
                    if stage == "source":
                        stage,current_api,current_data = "target",target,self.mockdata2value(deepcopy(mock_data['target']))
                    call_stack+=1
                elif action == "response":
                    if "fail_inform" in action_seq:
                        tmp_situation = system_action_list_backward[action]['template'].format(src_api=source,trg_api=target)
                    else:
                        tmp_situation = system_action_list_backward[action]['template_trg'].format(trg_api=target)
                elif action == 'clarify':
                    tmp_situation = system_action_list_backward[action]['template']
                elif action == "response_fail":
                    if action_seq[idx-1] == "negate":
                        tmp_situation = system_action_list_backward[action]['template_deny']
                    else:
                        tmp_situation = system_action_list_backward[action]['template']
                elif action == "suggest":
                    tmp_situation = system_action_list_backward[action]['template'].format(api=current_api)
                elif action == "retriever_call":
                    if action_seq[idx-1] == "fail_inform":
                        tmp_situation = system_action_list_backward[action]['template_chain'].format(chain_param = subgraph['chain_param'])
                    else:
                        tmp_situation = system_action_list_backward[action]['template']
                if action in aug_ment and random.sample([0,0,1,1,1],1)[0]:
                    tmp_situation+=f"System's speech should follow this format: {random.sample(aug_ment[action],1)[0]}"
                tmp_prompt = f"System turn\n-system action: {action}({system_action_list_backward[action]['description']})\n-situation:{tmp_situation}"
                if current_ret:
                    tmp_prompt+=f"\nRetriever status: {current_ret}"
                    current_ret = False
                else:
                    tmp_prompt+=f"\nRetriever status: {default_ret}"
                prompt.append(tmp_prompt)
        return prompt

try:
    with open("config.yml") as f:
        config = yaml.load(f,Loader=yaml.SafeLoader)
        
    openai_key = config['api_key']
    device = config['device']
    is_sampling = config['is_sampling']
    
    pair_num = int(sys.argv[1])
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

    with open(f"pair_list/sep_pair_{pair_num}.json",'r') as f:
        pair_list = json.load(f)

    print(f"Before sports remove:{len(pair_list)}")
    pair_list = [pair for pair in pair_list if pair[0][0] not in ben_sports_name and pair[0][1] not in ben_sports_name]
    print(f"After sports remove:{len(pair_list)}")
    
    pair_list = [pair for pair in pair_list if not is_error_in(pair)]
    retriever = Retriever(device)
    model_ret = SentenceTransformer("ToolBench/ToolBench_IR_bert_based_uncased",device = device)
    
    
    with get_openai_callback() as cb:
        yesyes = YesYes(pair_list,"A")
    print(cb)
    print(len(yesyes.freeform))

    with open(f"yesyes/yesyes_scen_{pair_num}.json",'w') as f:
        json.dump(yesyes.prompt_list,f,indent=4)
    print(f"Generated length: {len(yesyes.prompt_list)}")
        
    dst_list = []
    for idx in range(len(yesyes.scenario_list)):
        dst_list.append(yesyes.scenario_list[idx]['dst'])
        
    with open(f"yesyes/yesyes_dst_{pair_num}.json",'w') as f:
        json.dump(dst_list,f,indent=4)
        
    with open(f"yesyes/yesyes_mockdata_{pair_num}.json",'w') as f:
        json.dump(yesyes.mockdata_list,f,indent=4)
except Exception as e:
    with open(f"yesyes_error_{pair_num}.txt",'w') as f:
        f.write(traceback.format_exc())
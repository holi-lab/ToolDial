import json,re,random,sys
from tqdm import tqdm
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")

with open("../../after_coor_before_emb.json",'r') as f:
    api_list = json.load(f)
api_list_dict = {api['full_name']:api for api in api_list}

with open(f"../pair_list/all_edge_list_graph.json",'r') as f:
    all_edge_list_graph = json.load(f)
    
with open(f"../pair_list/normal_graph2idx.json",'r') as f:
    normal_graph2idx = json.load(f)
    
with open(f"../pair_list/normal_idx2igraph.json",'r') as f:
    normal_idx2igraph = json.load(f)
    

def replace_bool_values(text):
    text = re.sub(r"(?<!['\"])\bfalse\b(?!['\"])", "'false'", text)
    text = re.sub(r"(?<!['\"])\btrue\b(?!['\"])", "'true'", text)
    text = re.sub(r"(?<!['\"])\bnull\b(?!['\"])", "'null'", text)
    return text


def dialogue_postprocess(dialogue):
    dialogue = replace_bool_values(dialogue)
    if "```json" in dialogue:
        dialogue = dialogue.replace("```json","")
    if "```" in dialogue:
        dialogue = dialogue.replace("```", "")
    try:
        return json.loads(dialogue)
    except:
        return eval(dialogue)
    
def dst_req_opt(dst):
    api_docs = api_list_dict[dst['api_status']['api_name']]
    api_status = {"api_name":dst['api_status']['api_name'],"required_parameters":{},"optional_parameters":{},"system_action":dst['api_status']['system_action']}
    req_name = [param['name'] for param in api_docs['required_parameters']]
    opt_name = [param['name'] for param in api_docs['optional_parameters']]
    for param in dst['api_status']['input_parameter']:
        if param in req_name:
            api_status['required_parameters'][param] = dst['api_status']['input_parameter'][param]
        elif param in opt_name:
            api_status['optional_parameters'][param] = dst['api_status']['input_parameter'][param]
    dst['api_status'] = api_status
    return dst
    
    
def dst_combination(dialogue,dst): ## 여기서 dst를 req, opt로 나누게끔 작성함
    dst_idx=0
    for turn_idx,turn in enumerate(dialogue):
        if "User" in turn:
            continue
        
        if dst[dst_idx]['api_confirmed'] == "true":
            dialogue[turn_idx]['System']['dialogue_state'] = dst_req_opt(dst[dst_idx])
        else:
            dialogue[turn_idx]['System']['dialogue_state'] = dst[dst_idx]
        dst_idx+=1
    return dialogue

def extract_relation(text):
    pattern = r"<Relation>\s*(.*?)\s*<Dialogue scenario>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None

def extract_source_target(scen_list):
    node_list = []
    for scen in scen_list:
        api = {"source":"","target":""}
        step1 = scen.split("Use this data as the actual input parameter and output components of the APIs on the dialogue.\n")[1]
        step2 = step1.split("\n\n<Dialogue format>")[0]
        source = step2.split("\n\n")[0].split("\n")[0].replace("Data for ","")[:-1]
        target = step2.split("\n\n")[1].split("\n")[0].replace("Data for ","")[:-1]
        api['source'],api['target'] = source,target
        node_list.append(api)
    return node_list

number = int(sys.argv[1])
file = sys.argv[2]

scenario_list = []
for idx in range(number):
    try:
        with open(f"{file}_scen_{idx}.json",'r') as f:
            tmp = json.load(f)
        scenario_list+=tmp
    except:
        continue
    
dst_list = []
for idx in range(number):
    try:
        with open(f"{file}_dst_{idx}.json",'r') as f:
            tmp = json.load(f)
        dst_list+=tmp
    except:
        continue
    
mockdata_list = []
for idx in range(number):
    try:
        with open(f"{file}_mockdata_{idx}.json",'r') as f:
            tmp = json.load(f)
        mockdata_list+=tmp
    except:
        continue
    
dial_list = []
for idx in range(number):
    try:
        with open(f"raw_dial_{file}_{idx}.json",'r') as f:
            tmp = json.load(f)
        dial_list+=tmp
    except:
        continue

trash = []
dial_list_dst = []
node_list = extract_source_target(scenario_list)
for idx,dial in enumerate(tqdm(dial_list)):
    try:
        tmp = {
            "subgraph_type":file,
            "dialogue":dst_combination(dialogue_postprocess(dial),dst_list[idx]),
            "mockdata":mockdata_list[idx]
        }
        tmp.update(node_list[idx])
        dial_list_dst.append(tmp)
    except:
        trash.append(dial)
        continue
    

    
with open(f"trash_dial_{file}.json",'w') as f:
    json.dump(trash,f,indent=4)

for dial_idx,dial in enumerate(dial_list_dst):
    only_dial = dial['dialogue']
    for turn_idx,turn in enumerate(only_dial):
        if "User" in turn:
            continue
        if turn['System']['action'] == "retriever_call":
            if type(turn['System']['retriever_status']['retrieved_api']) == list:
                if type(turn['System']['retriever_status']['retrieved_api'][0]) == set:
                    print("Change at",dial_idx)
                    turn['System']['retriever_status']['retrieved_api'] = list(turn['System']['retriever_status']['retrieved_api'][0])
                elif type(turn['System']['retriever_status']['retrieved_api'][0]) == tuple:
                    turn['System']['retriever_status']['retrieved_api'] = turn['System']['retriever_status']['retrieved_api'][0]
                if len(turn['System']['retriever_status']['retrieved_api']) == 1:
                    chain_prompt = "Output to procure input parameter {input_param} of {target}: {output_comp}"
                    source_idx = normal_graph2idx[dial['source']]
                    target_idx = normal_graph2idx[dial['target']]
                    edge = all_edge_list_graph[source_idx][target_idx][0]
                    turn['System']['retriever_status']['retrieved_api'].append(chain_prompt.format(input_param =edge[3][0],
                                                                                                   output_comp = edge[1][0],
                                                                                                   target = dial['target']))
        if turn['System']['action'] == "call":
            if not turn['System']['api_response'] and turn['System']['dialogue_state']['api_status']['api_name'] == dial['source']:
                turn['System']['api_response'] = {"api_name":dial['source'],"result":dial['mockdata']['source'][0]['output_components']}
            elif not turn['System']['api_response'] and turn['System']['dialogue_state']['api_status']['api_name'] == dial['target']:
                turn['System']['api_response'] = {"api_name":dial['target'],"result":dial['mockdata']['target'][0]['output_components']}

print(len(trash))
print(f"Postprocess Fail Ratio: {len(trash)/len(dial_list)}")

print(f"Total dialogue: {len(dial_list)}")
    
with open(f"../sample_dialogue_{file}.json",'w') as f:
    json.dump(dial_list_dst,f,indent=4)
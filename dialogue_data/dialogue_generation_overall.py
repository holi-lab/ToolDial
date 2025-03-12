import json,re,random
from copy import deepcopy
    
with open("../after_coor_before_emb.json",'r') as f:
    api_list = json.load(f)
api_list_dict = {api['full_name']:api for api in api_list}

def action_seq(dial):
    action_seq = []
    for turn in dial['dialogue']:
        if "User" in turn:
            action_seq.append(turn['User']['action'])
        else:
            action_seq.append(turn['System']['action'])
    return tuple(action_seq)

def post_process(text):
    if "```json" in text:
        text = text.replace("```json","")
    if "```" in text:
        text = text.replace("```", "")
    return text

def count_line(text):
    return len((text).split("\n"))

def replace_dialogue_history(replacing_text,replaced_text):
    text_line = count_line(replacing_text)
    replaced_text_list = replaced_text.split("\n")
    replaced_text_list[:text_line] = replacing_text.split("\n")
    return "\n".join(replaced_text_list)

def docs_summary(dst,real_api):
    api_name = dst['api_status']['api_name']
    if api_name in api_list_dict:
        full_docs = api_list_dict[api_name]
    else:
        full_docs = api_list_dict[real_api]
    req_name,opt_name = dst['api_status']['required_parameters'],dst['api_status']['optional_parameters']
    req_param = [{"input_parameter_name":param['name'],"description":param['object_long']} for param in full_docs['required_parameters'] if param['name'] in req_name]
    opt_param = [{"input_parameter_name":param['name'],"description":param['object_long']} for param in full_docs['optional_parameters'] if param['name'] in opt_name]
    new_docs = {"api_name":api_name,"api_description":full_docs['description'],"required_parameters":req_param,"optional_parameters":opt_param}
    return new_docs


def docs_summary_with_name(api_name):
    full_docs = api_list_dict[api_name]
    req_name,opt_name = [param['name'] for param in full_docs['required_parameters']],[param['name'] for param in full_docs['optional_parameters']]
    req_param = [{"input_parameter_name":param['name'],"description":param['object_long']} for param in full_docs['required_parameters'] if param['name'] in req_name]
    opt_param = [{"input_parameter_name":param['name'],"description":param['object_long']} for param in full_docs['optional_parameters'] if param['name'] in opt_name]
    new_docs = {"api_name":api_name,"api_description":full_docs['description'],"required_parameters":req_param,"optional_parameters":opt_param}
    return new_docs

def normalize_double(text):
    result = re.sub(r'"\s*\'\s*([^\'"]+)\s*\'\s*"', r"'\1'", text)
    return result

def total_dialogue_obs(dialogue):
    source,target = dialogue['source'],dialogue['target']
    dialogue = dialogue['dialogue']
    prompt = ""
    action_list = []
    for idx,turn in enumerate(dialogue):
        if "User" in turn:
            prompt+=f"User: {turn['User']['message']}\nSystem:\n"
            action_list.append(turn['User']['action'])
        else:
            observation = {}
            if "User" in dialogue[idx-1]:
                tmp_dst = deepcopy(turn['System']['dialogue_state'])
                if tmp_dst['api_confirmed']== "true":
                    del tmp_dst['api_status']['system_action']
                prompt+=f"- Dialogue State: {tmp_dst}\n"
            if turn['System']['action'] == "retriever_call":
                prompt+=f"- Thought: {turn['System']['thought']}\n- Action:{turn['System']['action']}\n"
                observation['retriever_status'] = turn['System']['retriever_status']
                if turn['System']['retriever_status']['retriever_call'] == "true":
                    ret_api = turn['System']['retriever_status']['retrieved_api']
                    if isinstance(ret_api,dict):
                        for api in ret_api:
                            if turn['System']['retriever_status']['retrieved_api'][api]>0.5:
                                if dialogue[idx+1]['System']['action'] == "suggest":
                                    if dialogue[idx+2]['User']['action'] == "negate":
                                        observation['api_documentation'] = docs_summary_with_name(target)
                                    else:
                                        observation['api_documentation'] = docs_summary(dialogue[idx+3]['System']['dialogue_state'],target)
                                else:
                                    observation['api_documentation'] = docs_summary(dialogue[idx+1]['System']['dialogue_state'],target)
                                break
                    else:
                        observation['api_documentation'] = docs_summary(dialogue[idx+1]['System']['dialogue_state'],source)
                    action_list.append(turn['System']['action'])
                    prompt+=f"- Observation: {observation}\n"
                continue
                
            if turn['System']['action'] == "call":
                prompt+=f"- Thought: {turn['System']['thought']}\n- Action:{turn['System']['action']}\n"
                if turn['System']['api_response']:
                    observation['call_result'] = turn['System']['api_response']
                    prompt+=f"- Observation: {observation}\n"
                else:
                    observation['call_result'] = dialogue[idx+1]['System']['api_response']
                    prompt+=f"- Observation: {observation}\n"
                action_list.append(turn['System']['action'])
                continue
            
            prompt+=f"- Thought: {turn['System']['thought']}\n- Action:{turn['System']['action']}\n"
            prompt+=f"- Observation: {observation}\n"

                    
            tmp_dst = deepcopy(turn['System']['dialogue_state'])
            if tmp_dst['api_confirmed']== "true":
                del tmp_dst['api_status']['system_action']
            action_list.append(turn['System']['action'])
            prompt+=f"- Dialogue State: {tmp_dst}\n"
                
            try:
                if turn['System']['message']:
                    current_message = turn['System']['message'].replace("\n"," ")
                    prompt+=f"- Message: {current_message}\n"
            except:
                if turn['System']['api_response']['message']:
                    current_message = turn['System']['api_response']['message'].replace("\n"," ")
                    prompt+=f"- Message: {current_message}\n"
    return {"dialogue":normalize_double(prompt),"action_list":action_list}

with open("train.json",'r') as f:
    train_dial = json.load(f)

with open("val.json",'r') as f:
    val_dial = json.load(f)

with open("test.json",'r') as f:
    test_dial = json.load(f)

train_dial_list = [total_dialogue_obs(dial) for dial in train_dial]
val_dial_list = [total_dialogue_obs(dial) for dial in val_dial]    
test_dial_list = [total_dialogue_obs(dial) for dial in test_dial]

with open("train_dialogue_overall_obs.json",'w') as f:
    json.dump(train_dial_list,f,indent=4)
    
with open("val_dialogue_overall_obs.json",'w') as f:
    json.dump(val_dial_list,f,indent=4)
    
with open("test_dialogue_overall_obs.json",'w') as f:
    json.dump(test_dial_list,f,indent=4)
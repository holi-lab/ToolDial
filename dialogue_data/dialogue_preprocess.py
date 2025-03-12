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

def total_dialogue(dialogue):
    source,target = dialogue['source'],dialogue['target']
    dialogue = dialogue['dialogue']
    prompt = ""
    action_list = []
    for idx,turn in enumerate(dialogue):
        if "User" in turn:
            prompt+=f"User: {turn['User']['message']}\nSystem:\n"
            action_list.append(turn['User']['action'])
        else:
            if "User" in dialogue[idx-1]:
                if dialogue[idx-1]['User']['action'] == "inform":
                    tmp_dst = deepcopy(turn['System']['dialogue_state'])
                    if tmp_dst['api_confirmed']== "true":
                        del tmp_dst['api_status']['system_action']
                    prompt+=f"- Dialogue State: {tmp_dst}\n"
                    prompt+=f"- Thought: {turn['System']['thought']}\n- Action:{turn['System']['action']}\n"
                    prompt+=f"- Retriever status: {turn['System']['retriever_status']}\n"
                    if turn['System']['api_response']:
                        prompt+=f"- Call result: {turn['System']['api_response']}\n"
                    else:
                        prompt+=f"- Call result: {dialogue[idx+1]['System']['api_response']}\n"
                    action_list.append(turn['System']['action'])
                    continue
            
            if turn['System']['action'] == "retriever_call":
                prompt+=f"- Thought: {turn['System']['thought']}\n- Action:{turn['System']['action']}\n"
                prompt+=f"- Retriever status: {turn['System']['retriever_status']}\n"
                if turn['System']['retriever_status']['retriever_call'] == "true":
                    ret_api = turn['System']['retriever_status']['retrieved_api']
                    if isinstance(ret_api,dict):
                        for api in ret_api:
                            if turn['System']['retriever_status']['retrieved_api'][api]>0.5:
                                if dialogue[idx+1]['System']['action'] == "suggest":
                                    if dialogue[idx+2]['User']['action'] == "negate":
                                        prompt+=f"- API documentations: {docs_summary_with_name(target)}\n" ## 어차피 이 상황에선 docs를 안쓰니까 ㄱㅊ
                                    else:
                                        prompt+=f"- API documentations: {docs_summary(dialogue[idx+3]['System']['dialogue_state'],target)}\n"
                                else:
                                    prompt+=f"- API documentations: {docs_summary(dialogue[idx+1]['System']['dialogue_state'],target)}\n"
                                break
                    else:
                        prompt+=f"- API documentations: {docs_summary(dialogue[idx+1]['System']['dialogue_state'],source)}\n"
                    action_list.append(turn['System']['action'])
                continue
                
            if turn['System']['action'] == "call":
                prompt+=f"- Thought: {turn['System']['thought']}\n- Action:{turn['System']['action']}\n"
                prompt+=f"- Retriever status: {turn['System']['retriever_status']}\n"
                if turn['System']['api_response']:
                    prompt+=f"- Call result: {turn['System']['api_response']}\n"
                else:
                    prompt+=f"- Call result: {dialogue[idx+1]['System']['api_response']}\n"
                action_list.append(turn['System']['action'])
                continue
            
            prompt+=f"- Thought: {turn['System']['thought']}\n- Action:{turn['System']['action']}\n"
            prompt+=f"- Retriever status: {turn['System']['retriever_status']}\n"
            try:
                if turn['System']['message']:
                    prompt+=f"- Message: {turn['System']['message']}\n"
            except:
                if turn['System']['api_response']['message']:
                    prompt+=f"- Message: {turn['System']['api_response']['message']}\n"
                    
            tmp_dst = deepcopy(turn['System']['dialogue_state'])
            if tmp_dst['api_confirmed']== "true":
                del tmp_dst['api_status']['system_action']
            action_list.append(turn['System']['action'])
            prompt+=f"- Dialogue State: {tmp_dst}\n"
    return {"dialogue":normalize_double(prompt),"action_list":action_list}

with open("train.json",'r') as f:
    train_dial = json.load(f)

with open("val.json",'r') as f:
    val_dial = json.load(f)

with open("test.json",'r') as f:
    test_dial = json.load(f)

train_dial_list = [total_dialogue(dial) for dial in train_dial]
val_dial_list = [total_dialogue(dial) for dial in val_dial]    
test_dial_list = [total_dialogue(dial) for dial in test_dial]

with open("train_dialogue.json",'w') as f:
    json.dump(train_dial_list,f,indent=4)
    
with open("val_dialogue.json",'w') as f:
    json.dump(val_dial_list,f,indent=4)
    
with open("test_dialogue.json",'w') as f:
    json.dump(test_dial_list,f,indent=4)
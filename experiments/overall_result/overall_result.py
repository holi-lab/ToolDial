import json,random,re,os,yaml
from transformers import AutoTokenizer

def normalize_action(text):
    lower_text = text.lower()
    normalized_text = re.sub(r'[^a-z]', '', lower_text)
    return normalized_text.replace("action","").replace("api","")

def extract_observation(label):
    lines = label.split("\n")
    return_label = {}
    for idx,sentence in enumerate(lines):
        if "- Action:" in sentence:
            current_obs = lines[idx+1].replace("- Observation: ","")
            current_action = normalize_action(sentence.replace("- Action:","").strip())
            if current_action in return_label:
                return_label[current_action].insert(0,current_obs)
            else:
                return_label[current_action] = [current_obs]
    return return_label

def overall_form(full_dialogue,tokenizer_inf,is_val=False):
    label = []
    prompt = ""
    lines = full_dialogue['dialogue'].split("\n")
    for idx,sentence in enumerate(lines):
        prompt+=sentence+"\n"
        if "User: " in sentence:
            current_dial = {"dial":prompt,"label":""}
            tmp_idx = 1
            while "User: " not in lines[idx+tmp_idx]:
                current_dial["label"]+=lines[idx+tmp_idx]+"\n"
                tmp_idx+=1
                if idx+tmp_idx==len(lines):
                    break
            current_dial['dial']+="System:"
            current_dial['label'] = current_dial['label'].replace("System:\n","")
            if is_val: ## test
                chat_eval = [{"role": "user", "content":  prompt}] #
                template_chat = tokenizer_inf.apply_chat_template(chat_eval, tokenize=False, add_generation_prompt=True)
                label.append({"dial":template_chat,
                              "label":current_dial['label'],
                              "obs":extract_observation(current_dial['label'])})
            else: ## train
                history_chat = [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}]
                chat = [{"role": "user", "content": prompt}, {"role": "assistant", "content": current_dial['label']}]
                template_history = tokenizer_inf.apply_chat_template(history_chat, tokenize=False, add_generation_prompt=False)
                template_chat = tokenizer_inf.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                
                label.append({
                    "history": template_history.replace("\n<|eot_id|>", "\n"),
                    "dial": template_chat,
                    "label": current_dial['label']})
    return label

def normalize_json(data):
    def traverse_and_normalize(value):
        if isinstance(value, dict):
            return {traverse_and_normalize(k): traverse_and_normalize(v) for k, v in value.items()}
        elif isinstance(value, list):
            value = str(value).lower()
            return re.sub(r'[^a-z0-9]', '', value.lower())
        elif isinstance(value, str):
            return re.sub(r'[^a-z0-9]', '', value.lower())
        elif isinstance(value,int) or isinstance(value,float):
            value = str(value)
            return re.sub(r'[^a-z0-9]', '', value.lower())
        else:
            return value
    return traverse_and_normalize(data)

def extract_and_convert_dict(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        dict_str = match.group(0)  # Extract the matched string (the dictionary)
        dict_str = dict_str.replace("'", '"')
        try:
            extracted_dict = json.loads(dict_str)
            return extracted_dict
        except json.JSONDecodeError as e:
            try:
                extracted_dict=eval(dict_str)
                return extracted_dict
            except:
                return {"foo":"bar"}
    else:
        print("No dictionary found in the text.")
        return {"foo":"bar"}
    
def normalize_action(text):
    lower_text = text.lower()
    normalized_text = re.sub(r'[^a-z]', '', lower_text)
    return normalized_text.replace("action","").replace("api","")
    
def action_preprocess(text,is_thought,is_normal): ## 경우의 수 네 가지
    text = text.lower()
    if not is_normal:
        if is_thought:
            text = text.split("<end_of_system_thought>")[1]
            text = text.replace("<end_of_system_action>","").replace("<start_of_system_action>","").replace("<|eot_id|>","").strip().replace("\n","")
        else:
            text = text.replace("<end_of_system_action>","").replace("<start_of_system_action>", "").strip().replace("\n","")
    else:
        if is_thought:
            text = text.split("- action:")[1]
        else:
            text = text.replace("- action:","")
    return text

def reasoning_step_validate(label,predict):
    label_lines = label.split("\n")
    predict_lines = predict.split("\n")
    label_dst= [normalize_json(extract_and_convert_dict(line)) for line in label_lines if "- Dialogue State: " in line]
    predict_dst = [normalize_json(extract_and_convert_dict(line)) for line in predict_lines if "- Dialogue State: " in line]
    label_action = [normalize_action(action_preprocess(line,False,True)) for line in label_lines if "- Action:" in line]
    predict_action = [normalize_action(action_preprocess(line,False,True)) for line in predict_lines if "- Action:" in line]
    
    if label_action != predict_action or label_dst != predict_dst:
        return False
    return True

def evaluate_dialogue(dialogue):
    correct = 0
    for idx in range(len(dialogue)):
        if dialogue[idx]['predict'] == None:
            continue
        if reasoning_step_validate(dialogue[idx]['label'],dialogue[idx]['predict']):
            correct+=1
    if correct == len(dialogue):
        return 1
    return 0


with open("overall_result.yml") as f:
    config = yaml.load(f,Loader=yaml.SafeLoader)
model_name = config['model_name']
file_list = os.listdir(f"{model_name}")
file_list.sort()
results=[]
for file in file_list:
    if ".json" not in file or "test_dialogue.json" in file:
        continue
    with open(f"{model_name}/{file}",'r') as f:
        results+=json.load(f)
    
print(len(results))

with open("../../dialogue_data/test_dialogue.json",'r') as f:
    test_dialogue = json.load(f)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({"pad_token": "<pad>","unk_token":"<unk>"})

dial_length = []
n_turn = 0
for dial in test_dialogue:
    n = len(overall_form(dial,tokenizer,is_val=True))
    dial_length.append(n)
    n_turn+=n


correct = 0
error = []
for idx in range(len(results)):
    if results[idx]['predict'] == None:
        error.append(results[idx])
        continue
    if reasoning_step_validate(results[idx]['label'],results[idx]['predict']):
        correct+=1
    else:
        error.append(results[idx])
    
print(f"{model_name} overall performance:",correct/len(results))

current_idx = 0
all_dials = []
for length in dial_length:
    current_dial = results[current_idx:current_idx+length]
    current_idx+=length
    all_dials.append(current_dial)
    

correct=0
error=[]
for dial in all_dials:
    if evaluate_dialogue(dial):
        correct+=1
    else:
       error.append(dial) 

print(f"{model_name} dialogue success rate:",correct/len(all_dials))
import re,json
from prompts import evaluation_dst_prompt,add_tokens
from prompts import action_evaluation_prompt_thought_normal,action_evaluation_prompt_thought_token,action_evaluation_prompt_action
from prompts import hall_evaluation_prompt

timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat = re.compile("\d{1,3}[.]\d{1,2}")

with open('mapping.pair') as fin:
    replacements = []
    for line in fin.readlines():
        tok_from, tok_to = line.replace('\n', '').split('\t')
        replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))
########## Custom ############
def bs_normalize(current_bel):
    pattern = r"'([^']*)'"
    current_bel = str(current_bel).replace(":"," :").replace("{","{ ").replace("}"," }")
    current_bel = re.sub(pattern, r"' \1 '", current_bel)
    current_bel = re.sub(r",", " ,", current_bel)
    current_bel = current_bel.replace("_"," _ ")
    return current_bel

def decode2dict(string):
    tmp_dict = eval(string)
    new_dict = {}
    for slot in tmp_dict:
        new_dict[slot.strip().replace(" _ ","_")] = tmp_dict[slot].strip()
    return new_dict
#############################33


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text


def normalize(text):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # normalize phone number
    ms = re.findall(r'\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))

    # normalize postcode
    ms = re.findall(r'([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub(r'[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub(r'$/', '', text)
    text = text.replace('/', ' and ')

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, doesn't, you'd ... etc
    # 따옴표를 제거하는 부분을 모두 주석 처리
    # text = re.sub('^\'', '', text)
    # text = re.sub('\'$', '', text)
    # text = re.sub(r'\'\s', ' ', text)
    # text = re.sub(r'\s\'', ' ', text)

    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(r' +', ' ', text)

    # concatenate numbers
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(r'^\d+$', tokens[i]) and re.match(r'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text

def replace_bool_values(text):
    text = re.sub(r"(?<!['\"])\bfalse\b(?!['\"])", "'false'", text)
    text = re.sub(r"(?<!['\"])\btrue\b(?!['\"])", "'true'", text)
    text = re.sub(r"(?<!['\"])\bnull\b(?!['\"])", "'null'", text)
    return text

def finetune_form(dial):
    lines = dial.split("\n")
    new_lines = []
    for sentence in lines:
        if "User: " in sentence:
            new_sentence = sentence.replace("User:","").strip()
            new_sentence = "<start_of_user_message> " + new_sentence + " <end_of_user_message>"
            new_lines.append(normalize(new_sentence))
        elif "- Thought:" in sentence:
            new_sentence = sentence.replace("- Thought:","").strip()
            new_sentence = "<start_of_system_thought> " + new_sentence + " <end_of_system_thought>"
            new_lines.append(normalize(new_sentence))
        elif "System:" in sentence:
            continue
        elif "- Action:" in sentence:
            new_sentence = sentence.replace("- Action:","").strip()
            new_sentence = "<start_of_system_action> " + new_sentence + " <end_of_system_action>"
            new_lines.append(normalize(new_sentence))
        elif "- Retriever status:" in sentence:
            new_sentence = sentence.replace("- Retriever status:","").strip()
            new_sentence = "<start_of_retriever_status> " + new_sentence + " <end_of_retriever_status>"
            new_lines.append(normalize(new_sentence))
        elif "- API documentations:" in sentence:
            new_sentence = sentence.replace("- API documentations:","").strip()
            new_sentence = "<start_of_api_documentation> " + new_sentence + " <end_of_api_documentation>"
            new_lines.append(normalize(new_sentence))
        elif "- Message:" in sentence:
            new_sentence = sentence.replace("- Message:","").strip()
            new_sentence = "<start_of_system_message> " + new_sentence + " <end_of_system_message>"
            new_lines.append(normalize(new_sentence))
        elif "- Dialogue State:" in sentence:
            new_sentence = sentence.replace("- Dialogue State:","").strip()
            new_sentence = "<start_of_dialogue_state> " + new_sentence + " <end_of_dialogue_state>"
            new_lines.append(normalize(new_sentence))
        elif "- Call result:" in sentence:
            new_sentence = sentence.replace("- Call result:","").strip()
            new_sentence = "<start_of_call_result> " + new_sentence + " <end_of_call_result>"
            new_lines.append(normalize(new_sentence))
        else:
            if sentence != "":
                new_lines.append(normalize(sentence))
    text2return = "\n".join(new_lines)
    return replace_bool_values(text2return)


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

#-------------------------------------

def dst_form_token(full_dialogue,tokenizer,is_give_label=True,is_val=False):
    label = []
    prompt = ""
    action_list = full_dialogue['action_list']
    for sentence in full_dialogue['dialogue'].split("\n"):
        if "<start_of_dialogue_state>" in sentence:
            current_label = eval(sentence.replace("<start_of_dialogue_state>","").replace("<end_of_dialogue_state>","").strip())
            if is_val:
                chat_eval = [{"role" : "user", "content" : evaluation_dst_prompt + prompt}]
                template_chat = tokenizer.apply_chat_template(chat_eval, tokenize=False, add_generation_prompt=True) ## eval에선 이렇게
                label.append({"dial":template_chat + "<start_of_dialogue_state> ","label":"<start_of_dialogue_state> "+str(current_label)+" <end_of_dialogue_state>"})
            else:
                chat = [{"role" : "user", "content" : evaluation_dst_prompt + prompt}, {"role" : "assistant", "content" : "<start_of_dialogue_state> "+str(current_label)+" <end_of_dialogue_state>"}]## form of train
                history_chat = [{"role" : "user", "content" : evaluation_dst_prompt + prompt}, {"role" : "assistant", "content" : ""}]## form of train
                template_history = tokenizer.apply_chat_template(history_chat, tokenize=False, add_generation_prompt=False)
                template_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) ## train에선 이렇게
                label.append({"history":template_history.replace("\n<|eot_id|>","\n"),"dial":template_chat,"label":"<start_of_dialogue_state> "+str(current_label)+" <end_of_dialogue_state>"})
            if is_give_label:
                prompt+=sentence+"\n"
        else:
            prompt+=sentence+"\n"
    return label

def dst_form_normal(full_dialogue,tokenizer,is_give_label=True,is_val=False):
    label = []
    prompt = ""
    for sentence in full_dialogue['dialogue'].split("\n"):
        if "- Dialogue State: " in sentence:
            current_label = eval(sentence.replace("- Dialogue State: ","").strip())
            if is_val:
                chat_eval = [{"role" : "user", "content" : evaluation_dst_prompt + prompt}]
                template_chat = tokenizer.apply_chat_template(chat_eval, tokenize=False, add_generation_prompt=True) ## eval에선 이렇게
                label.append({"dial":template_chat + "- Dialogue State: ","label":str(current_label)})
            else:
                chat = [{"role" : "user", "content" : evaluation_dst_prompt + prompt}, {"role" : "assistant", "content" : "- Dialogue State: "+str(current_label)}]## form of train
                history_chat = [{"role" : "user", "content" : evaluation_dst_prompt + prompt}, {"role" : "assistant", "content" : ""}]## form of train
                template_history = tokenizer.apply_chat_template(history_chat, tokenize=False, add_generation_prompt=False)
                template_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) ## train에선 이렇게
                label.append({"history":template_history.replace("\n<|eot_id|>","\n"),"dial":template_chat,"label":str(current_label)})
            if is_give_label:
                prompt+=sentence+"\n"
        else:
            prompt+=sentence+"\n"
    return label


#-------------------------------------

def action_form_action_token(full_dialogue, tokenizer,is_give_label, is_val=False): ## y -> <sos>
    label = []
    prompt = ""
    action_list = full_dialogue['action_list']
    for sentence in full_dialogue['dialogue'].split("\n"):
        if "<start_of_system_action>" in sentence:
            current_label = sentence.replace("<start_of_system_action>", "").replace("<end_of_system_action>", "").strip()
            if is_val:
                chat_eval = [{"role": "user", "content": action_evaluation_prompt_action + prompt}]
                template_chat = tokenizer.apply_chat_template(chat_eval, tokenize=False, add_generation_prompt=True)
                label.append({"dial": template_chat + "<start_of_system_action> ", "label": str(current_label)})
            else:
                chat = [{"role": "user", "content": action_evaluation_prompt_action + prompt}, {"role": "assistant", "content": "<start_of_system_action> " + str(current_label) + " <end_of_system_action>"}]
                history_chat = [{"role": "user", "content": action_evaluation_prompt_action + prompt}, {"role": "assistant", "content": ""}]
                template_history = tokenizer.apply_chat_template(history_chat, tokenize=False, add_generation_prompt=False)
                template_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                label.append({"history": template_history.replace("\n<|eot_id|>", "\n"), "dial": template_chat, "label": "<start_of_system_action> " + str(current_label) + " <end_of_system_action>"})
            if is_give_label:
                prompt += sentence + "\n"
        elif "<start_of_retriever_status>" in sentence and len(action_list) > 1:
            if action_list[-2] == "retriever_call" and "{'retriever_call': 'false', 'retrieved_api': 'none'}" in sentence:
                continue
            else:
                prompt += sentence + "\n"
        elif "<start_of_system_thought>" in sentence:
            continue
        else:
            prompt += sentence + "\n"
    return label

def action_form_action_normal(full_dialogue, tokenizer,is_give_label, is_val=False):
    label = []
    prompt = ""
    action_list = full_dialogue['action_list']
    for sentence in full_dialogue['dialogue'].split("\n"):
        if "- Action:" in sentence:
            current_label = sentence.replace("- Action:", "").strip()
            if is_val:
                chat_eval = [{"role": "user", "content": action_evaluation_prompt_action + prompt}]
                template_chat = tokenizer.apply_chat_template(chat_eval, tokenize=False, add_generation_prompt=True)
                label.append({"dial": template_chat + "- Action:", "label":str(current_label)})
            else:
                chat = [{"role": "user", "content": action_evaluation_prompt_action + prompt}, {"role": "assistant", "content": "- Action:"+str(current_label)}]
                history_chat = [{"role": "user", "content": action_evaluation_prompt_action + prompt}, {"role": "assistant", "content": ""}]
                template_history = tokenizer.apply_chat_template(history_chat, tokenize=False, add_generation_prompt=False)
                template_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                label.append({"history": template_history.replace("\n<|eot_id|>", "\n"), "dial": template_chat, "label":str(current_label)})
            if is_give_label:
                prompt += sentence + "\n"
        elif "- Retriever status:" in sentence and len(action_list) > 1:
            if action_list[-2] == "retriever_call" and "{'retriever_call': 'false', 'retrieved_api': 'none'}" in sentence:
                continue
            else:
                prompt += sentence + "\n"
        elif "- Thought:" in sentence:
            continue
        else:
            prompt += sentence + "\n"
    return label


def action_form_thought_normal(full_dialogue, tokenizer,is_give_label=True, is_val=False):
    label = []
    prompt = ""
    lines = full_dialogue['dialogue'].split("\n")
    for sentence_idx,sentence in enumerate(lines):
        if "- Thought: " in sentence:
            current_label = lines[sentence_idx+1].replace("- Action:", "").strip()
            if is_val:
                chat_eval = [{"role": "user", "content": action_evaluation_prompt_thought_normal + prompt}]
                template_chat = tokenizer.apply_chat_template(chat_eval, tokenize=False, add_generation_prompt=True)
                label.append({"dial": template_chat, "label": str(current_label)})
                prompt += sentence + "\n"
            else:
                history_chat = [{"role": "user", "content": action_evaluation_prompt_thought_normal + prompt}, {"role": "assistant", "content": ""}]
                template_history = tokenizer.apply_chat_template(history_chat, tokenize=False, add_generation_prompt=False)
                chat = [{"role": "user", "content": action_evaluation_prompt_thought_normal + prompt},{
                        "role":"assistant",
                        "content":sentence+"\n"+"- Action:"+str(current_label)
                    }]
                template_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                label.append({"history": template_history.replace("\n<|eot_id|>", "\n"), "dial": template_chat, "label": str(current_label)})
                prompt += sentence + "\n"
        else:
            prompt += sentence + "\n"
    return label


def action_form_thought_token(full_dialogue, tokenizer,is_give_label=True, is_val=False):
    label = []
    prompt = ""
    lines = full_dialogue['dialogue'].split("\n")
    for sentence_idx, sentence in enumerate(lines):
        if "<start_of_system_thought>" in sentence:
            current_label = lines[sentence_idx+1].replace("<start_of_system_action>", "").replace("<end_of_system_action>", "").strip()
            if is_val:
                chat_eval = [{"role": "user", "content": action_evaluation_prompt_thought_token + prompt}]
                template_chat = tokenizer.apply_chat_template(chat_eval, tokenize=False, add_generation_prompt=True)
                label.append({"dial": template_chat, "label": str(current_label)})
                prompt += sentence + "\n"
            else:
                history_chat = [{"role": "user", "content": action_evaluation_prompt_thought_token + prompt}, {"role": "assistant", "content": ""}]
                template_history = tokenizer.apply_chat_template(history_chat, tokenize=False, add_generation_prompt=False)
                chat = [
                    {"role": "user", "content": action_evaluation_prompt_thought_token + prompt}, 
                    {"role": "assistant", "content": sentence+"\n"+"<start_of_system_action> "+str(current_label) + " <end_of_system_action>"}
                    ]
                template_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                label.append({"history": template_history.replace("\n<|eot_id|>", "\n"), "dial": template_chat, "label": str(current_label)})
                prompt += sentence + "\n"
        else:
            prompt += sentence + "\n"
    return label

#-------------------------------------

def hall_form_token(dialogue, tokenizer_inf,is_val=False):
    lines = dialogue['dialogue'].split("\n")
    label = []
    prompt = ""
    for s_idx, sentence in enumerate(lines):
        if "<start_of_system_action> response" in sentence:
            if not is_val: ## train
                idx = 1
                while "<start_of_system_message>" not in lines[s_idx + idx]:
                    idx += 1
                message_idx = 1
                while "<start_of_dialogue_state>" not in lines[s_idx+idx+message_idx]:
                    message_idx+=1
                prompt += sentence + "\n"
                history_chat = [{"role": "user", "content": hall_evaluation_prompt + prompt}, {"role": "assistant", "content": ""}]
                chat = [{"role": "user", "content": hall_evaluation_prompt + prompt}, {"role": "assistant", "content": "\n".join(lines[s_idx + idx:s_idx+idx+message_idx])}]
                template_history = tokenizer_inf.apply_chat_template(history_chat, tokenize=False, add_generation_prompt=False)
                template_chat = tokenizer_inf.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                label.append({"history": template_history.replace("\n<|eot_id|>", "\n"), "dial": template_chat, "label": "\n".join(lines[s_idx + idx:s_idx+idx+message_idx])})
            else: ## val
                idx = 1
                while "<start_of_system_message>" not in lines[s_idx + idx]:
                    idx += 1
                message_idx = 1
                while "<start_of_dialogue_state>" not in lines[s_idx+idx+message_idx]:
                    message_idx+=1
                prompt += sentence + "\n"
                chat_eval = [{"role": "user", "content": hall_evaluation_prompt + prompt}] #
                template_chat = tokenizer_inf.apply_chat_template(chat_eval, tokenize=False, add_generation_prompt=True)
                label.append({"dial": template_chat,"label": "\n".join(lines[s_idx + idx:s_idx+idx+message_idx])})
        else:
            prompt += sentence + "\n"
    return label


def hall_form_normal(dialogue, tokenizer_inf,is_val=False):
    lines = dialogue['dialogue'].split("\n")
    label = []
    prompt = ""
    for s_idx, sentence in enumerate(lines):
        if "- Action:response" in sentence:
            if not is_val: ## train
                idx = 1
                while "- Message: " not in lines[s_idx + idx]:
                    idx += 1
                message_idx = 1
                while "- Dialogue State: " not in lines[s_idx+idx+message_idx]:
                    message_idx+=1
                prompt += sentence + "\n"
                history_chat = [{"role": "user", "content": hall_evaluation_prompt + prompt}, {"role": "assistant", "content": ""}]
                chat = [{"role": "user", "content": hall_evaluation_prompt + prompt}, {"role": "assistant", "content": "\n".join(lines[s_idx + idx:s_idx+idx+message_idx])}]
                template_history = tokenizer_inf.apply_chat_template(history_chat, tokenize=False, add_generation_prompt=False)
                template_chat = tokenizer_inf.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                label.append({"history": template_history.replace("\n<|eot_id|>", "\n"), "dial": template_chat, "label": "\n".join(lines[s_idx + idx:s_idx+idx+message_idx])})
            else: ## val
                idx = 1
                while "- Message: " not in lines[s_idx + idx]:
                    idx += 1
                message_idx = 1
                while "- Dialogue State: " not in lines[s_idx+idx+message_idx]:
                    message_idx+=1
                prompt += sentence + "\n"
                chat_eval = [{"role": "user", "content": hall_evaluation_prompt + prompt}] #
                template_chat = tokenizer_inf.apply_chat_template(chat_eval, tokenize=False, add_generation_prompt=True)
                label.append({"dial": template_chat,"label": "\n".join(lines[s_idx + idx:s_idx+idx+message_idx])})
        else:
            prompt += sentence + "\n"
    return label
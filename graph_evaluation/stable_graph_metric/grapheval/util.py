import openai 
import re 
import requests
import json
import os
import yaml



config_file="./config.yml"
CONFIG = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

OPENAI_API_KEY=CONFIG['api_key']


def valid_edge_idx(all_edge_list_graph):
    ''' 
    input : 리스트 형태의 그래프 파일
    output : 모든 edge의 index
    리스트 형태의 그래프 파일을 입력받아 모든 edge의 index를 출력하는 함수
    '''
    new_graph = all_edge_list_graph
    index_combinations = []
    for i in range(len(new_graph)):
        for j in range(len(new_graph[i])):
            for k in range(len(new_graph[i][j])):
                if len(new_graph[i][j][k]) > 0:  # 길이가 1 이상인 경우
                    index_combinations.append((i, j, k))
    
    # print("number of valid edges : ", len(index_combinations))
    return index_combinations



def convert_list_to_graph(original_graph):
    ''' 
    grpah의 format을 변환해주는 함수
    '''
    new_graph = {
        "nodes": [],
        "edge": []
    }
    
    source = original_graph[0]  
    target = original_graph[2]  
    new_graph["nodes"].append(source)
    new_graph["nodes"].append(target)
    
    # source_attribute 
    source_attribute = {
        "name": original_graph[1][0],
        "description": original_graph[1][2] 
    }    
    # target_attribute
    target_attribute = {
        "name": original_graph[3][0],
        "description": original_graph[3][2]  
    }
    edge = {
        "source": source,
        "target": target,
        "source_attribute": source_attribute,
        "target_attribute": target_attribute
    }
    new_graph["edge"].append(edge)
    
    return new_graph



def get_message_content_from_chatgpt(prompt:str="", temperature:float=0.5):
    '''
    input : prompt
    output : response 
    '''
    openai.api_key = OPENAI_API_KEY
    
    response = openai.chat.completions.create(
        model=CONFIG['model'],
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],
        temperature=temperature
    )
    return response.choices[0].message.content



def get_api_doc_by_api_name(apis, api_name:str) -> dict:
    ''' 
    input : full_api_name
    output : api document
    '''
    keys_to_keep = ['tool_name', 'tool_description', 'full_name', 'description', 'required_parameters', 'output_components']
    api_dict = {api_item["full_name"]: api_item for api_item in apis}  
    api_doc = api_dict.get(api_name)
    
    filtered_api_doc = {key: api_doc[key] for key in keys_to_keep if key in api_doc}
    filtered_api_doc['api_name'] = filtered_api_doc.pop('full_name')
    filtered_api_doc['api_description'] = filtered_api_doc.pop('description')
    return filtered_api_doc



def __replace_quotes(match):
    value = match.group(1).replace('"', '\\"')
    return f': "{value}"{match.group(2)}'
    
def preprocessed_response(response_content):
    
    if "<<Todo>>" in response_content:
        response_content = response_content.replace("<<Todo>>", "")
    if "```json" in response_content:
        response_content = response_content.replace("```json", "")
    if "```" in response_content:
        response_content = response_content.replace("```", "")
    if "\n" in response_content:
        response_content = response_content.replace("\n", "")
    if ": False" in response_content:
        response_content = response_content.replace(': False', ': "False"')  
    if ": True" in response_content:
        response_content = response_content.replace(': True', ': "True"')  
    response_content = re.sub(r"(?<!\\)'([^']*?)':", r'"\1":', response_content)
    response_content = re.sub(r": '([^']*?)'(,|\s*[\]}])", __replace_quotes, response_content)
    response_content = response_content.replace("'", '"')
    response_content = response_content.replace('"s', "'s")
    response_content = response_content.replace(", 's",', "s')
    response_content = response_content.replace("{'s",'{"s')
    response_content = response_content.replace(": 's",': "s')
    if "<<You generate this section>>" in response_content:
        response_content = response_content.replace("<<You generate this section>>", "")
    return response_content
    
    

def is_valid_json(result):
    """
    Checks if the given string is valid JSON.

    Args:
    data: The string to be checked.

    Returns:
    True if the string is valid JSON, False otherwise.
    """
    # check json format
    try:
        result = json.loads(result)
        # result_output = result['required_parameters']
        return True
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Cannot parse result into JSON or missing key: {result}") 
        return False
    


def main_server_call(full_name, tool_input, source_api_doc, graph):
    
    category, tool_name, api_name =  full_name.split('|', 2)
    toolbench_key = 'XyRuwbS0CVNQrIkEdeZaD1mU395427qxvGniTJMzLcPpOYhAHg'
    
    url = 'http://0.0.0.0:8080/virtual'
    data = {
        "category": category,
        "tool_name": tool_name,
        "api_name": api_name,
        "tool_input": tool_input,
        "strip": "filter", # truncate(default), filter, random
        "toolbench_key": toolbench_key,
        "source_api_doc" : source_api_doc,
        "graph" : graph
        }
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }
    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.text


def additional_info(filtered_api_docs):
    ''' 
    tool_description, api_info, api_description 제공
    '''
    api_name = list(filtered_api_docs.keys())[0]
    api_description = filtered_api_docs[api_name]['api_description']
    tool_description = filtered_api_docs[api_name]['tool_description']

    api_doc = {
        'tool_description': "",
        'api_info': "",
    }
    api_info = []
    api_info.append({
        'name': api_name,
        'description': api_description
    })
    api_doc = {
        'tool_description': tool_description,
        'api_info': api_info
    }
    
    return api_doc


def split_list_and_save_json(data_list, n, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    
    split_size = len(data_list) // n
    remainder = len(data_list) % n
    
    splits = []
    start_idx = 0
    
    for i in range(n):
        end_idx = start_idx + split_size + (1 if i < remainder else 0)
        split = data_list[start_idx:end_idx]
        splits.append(split)
        start_idx = end_idx
        with open(f'{output_dir}/split_{i}.json', 'w') as f:
            json.dump(split, f, ensure_ascii=False, indent=4)
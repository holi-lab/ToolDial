from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.requests import Request
import uvicorn
import time
import json
import os, yaml
import requests
from typing import Union
from utils import standardize, change_name

from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from tenacity import retry, wait_random_exponential, stop_after_attempt

config_file='config.yml'
CONFIG = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
print(CONFIG)
CACHE_FOLDER = CONFIG['cache_folder']

# OpenAI API
# from openai import OpenAI
import openai 

if 'api_base' in CONFIG:
    OPENAI_API_BASE=CONFIG['api_base']
else:
    OPENAI_API_BASE="https://api.openai.com/v1"
OPENAI_API_KEY=CONFIG['api_key']

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

from typing import List, Dict, Union

class Attribute(BaseModel):
    name: str
    description: str

class Edge(BaseModel):
    source: str
    target: str
    source_attribute: Attribute
    target_attribute: Attribute

class Graph(BaseModel):
    nodes: List[str]
    edge: List[Edge]


class ApiInfo(BaseModel):
    name: str
    description: str

class SourceApiDoc(BaseModel):
    tool_description: str
    api_info: List[ApiInfo]


class Info(BaseModel):
    category: str
    tool_name: str
    api_name: str
    tool_input: Union[str, dict]
    strip: str
    toolbench_key: str
    source_api_doc: SourceApiDoc
    graph : Graph

def prepare_tool_name_and_url(info):
    category = info.category
    standard_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in standard_category or "," in standard_category:
        standard_category = standard_category.replace(" ", "_").replace(",", "_")
    standard_category = standard_category.replace("__", "_")
    
    tool_name = info.tool_name
    api_name = change_name(standardize(info.api_name)).split("_for_")[0]
    if not tool_name.endswith(f"_for_{standard_category}"):
        tool_name = standardize(info.tool_name)
        code_string = f"""from my_tools.{standard_category}.{tool_name}.api import {api_name}"""
        tool_name += f"_for_{standard_category}"
    else:
        tmp_tool_name = standardize(tool_name.replace(f"_for_{standard_category}", ""))
        code_string = f"""from my_tools.{standard_category}.{tmp_tool_name}.api import {api_name}"""
    return tool_name, standard_category, api_name, code_string

# @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(1))
@app.post('/virtual')
def get_virtual_response(request: Request, info: Info):
    user_key = info.toolbench_key
    source_api_doc = info.source_api_doc
    graph = info.graph 
    
    tool_name, standard_category, api_name, code_string = prepare_tool_name_and_url(info)
    tool_input = info.tool_input
    tool_name_original = info.tool_name

    if api_name == "chat_with_user":
        return {"error": "", "response": "Chat with user."}
    
    try:
        tool_input = json.loads(tool_input)
    except Exception as e:
        if tool_input == "":
            tool_input = {}
        elif isinstance(tool_input, dict):
            tool_input = tool_input
        else:
            print(f"Can not parse tool input into json: {tool_input}")
            print(type(tool_input))
            print(tool_input)
            response_dict = {"error": f"Tool input parse error...\n", "response": ""}
            return response_dict
    if not os.path.exists(CACHE_FOLDER):
        os.mkdir(CACHE_FOLDER)

    # load from cache
    cache = {}
    # prerequisite: to read files correctly, "my_tools_cache" folder and "toolenv/tools/" folder should be available
    try:
        if os.path.exists(os.path.join(CACHE_FOLDER, standard_category)):
            if os.path.exists(os.path.join(CACHE_FOLDER, standard_category, tool_name)):
                if os.path.exists(os.path.join(CACHE_FOLDER, standard_category, tool_name, api_name+".json")):
                    tools_cache_record = json.load(open(os.path.join(CACHE_FOLDER, standard_category, tool_name, api_name+".json"), "r"))
                    cache.update(tools_cache_record)
                    if str(tool_input) in cache:
                        print("using cached real response")
                        response_dict = cache[str(tool_input)]
                        return response_dict
    except Exception as e:
        print(f"Loading cache error: {e}")
    
    if str(tool_input) not in cache or not cache:
        print("cached response is nothing")
        
    """
    Call the real api before generating fake response
    """
    
    # headers = {
    # 'accept': 'application/json',
    # 'Content-Type': 'application/json',
    # 'toolbench_key': user_key
    # }
    # os.environ['HTTP_PROXY']= ''
    # if "_for_" in tool_name_original:
    #     tool_name_real = tool_name_original.split("_for_")[0]
    # else:
    #     tool_name_real = tool_name_original
    # data = {
    #     "category": standard_category,
    #     "tool_name": tool_name_real,
    #     "api_name": api_name,
    #     "tool_input": tool_input,
    #     "strip": "",
    #     "toolbench_key": user_key
    # }
    
    # real_response = requests.post(CONFIG['toolbench_url'], headers=headers, data=json.dumps(data))

    # # Check if the request was successful
    # if real_response.status_code == 200:
    #     real_response = real_response.json() 
    #     if check_result(real_response):
    #         print("returning real_response")
    #         if CONFIG['is_save']:
    #             save_cache(cache, tool_input, real_response, standard_category, tool_name, api_name)
    #         return real_response
    # else : 
    #     print("real_response is nothing")

    """
    Fake response function here. Use the cached history response for in-context examples.
    result = fake_response_function(api_doc, api_name, api_parameters, *kwargs)
    """

    # parse api_doc
    tool_name_original = standardize(tool_name_original)
    api_name = standardize(api_name)
    api_doc = {
        'tool_description': "",
        'api_info': "",
    }
    
    try:
        if os.path.exists(os.path.join(CONFIG['tools_folder'], standard_category)):
            if os.path.exists(os.path.join(CONFIG['tools_folder'], standard_category, tool_name_original.split("_for_")[0]+".json")):
                # read json
                api_intro = json.load(open(os.path.join(CONFIG['tools_folder'], standard_category, tool_name_original.split("_for_")[0]+".json"), "r"))
                # get tool_dexcription and api_info
                tool_description = api_intro['tool_description']
                api_info = []
                for api in api_intro['api_list']:
                    if api_name == standardize(api['name']):
                        api_info.append({
                            'name': api['name'],
                            'description': api['description']
                        })
                # check invalid api name
                if len(api_info) == 0:
                    print("cant match api name")
                api_doc = {
                    'tool_description': tool_description,
                    'api_info': api_info
                }
            else:
                print(f"cant get {tool_name_original}")
    except Exception as e:
        print(f"Loading api_doc error: {e}")

    # get several examples from cache
    example_num = 5
    # get top example_num examples
    api_example = list(cache.items())[:example_num]
    while len(str(api_example)) > 2048 and example_num > 1:
        example_num -= 1
        api_example = list(cache.items())[:example_num]

    print(f"api example: {api_example},,, tool_input: {tool_input},,, api_doc: {api_doc},,, source_api_doc: {source_api_doc},,,graph: {graph}")
        
    result = fake_response_function_chat(api_example,tool_input,api_doc, source_api_doc, graph)
    print(f"fake result: {result}")

    # if CONFIG['is_save']:
    #     save_cache(cache, tool_input, result, standard_category, tool_name, api_name)

    if not isinstance(result, dict):
        return json.loads(result)
    else:
        return result
    
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
        return True
    except Exception as e:
        print(f"Can not parse result into json: {result}")
        return False

def check_result(processes_value: dict):
    if 'error' not in processes_value or processes_value['error'] != '':
        return False
    if 'response' not in processes_value:
        return False
    response = str(processes_value['response'])
    if 'http' in response.lower() or 'connection' in response.lower() or 'rate limit' in response.lower() or 'time out' in response.lower() or 'timed out' in response.lower() or 'does not exist' in response.lower() or '404' in response.lower() or '504' in response.lower() or '500' in response.lower() or 'internal error' in response.lower() or 'API doesn\'t exists' in response.lower() or "API doesn\'t exists" in response.lower() or response == '{\'message\': "API doesn\'t exists"}' or 'Service Not Found' in response:
        return False
    elif 'authoriz' in response.lower() or 'authenticat' in response.lower() or 'unauthorized' in response.lower() or 'blocked user' in response.lower() or 'unsubscribe' in response.lower() or 'blocked' in response.lower() or '401' in response.lower() or '403' in response.lower() or 'credential' in response.lower() or 'unauthenticated' in response.lower() or 'disabled for your subscription' in response.lower() or 'ACCESS_DENIED' in response:
        return False
    elif 'parameter' in response.lower() or 'parse' in response.lower() or 'is not defined' in response.lower():
        return False
    elif len(response) == 0:
        return False
    elif "status_code=50" in response or "status_code=429" in response:
        return False
    return True

def save_cache(cache, tool_input, result, standard_category, tool_name, api_name, save_folder=CACHE_FOLDER):
    # save cache
    try:
        if isinstance(result, dict):
            cache[str(tool_input)] = result
        elif isinstance(result, str):
            try:
                result_dict = json.loads(result)
                cache[str(tool_input)] = result_dict
            except Exception as e:
                print(f"Load result failed: {e}")
                return

        if not os.path.exists(os.path.join(save_folder, standard_category)):
            os.mkdir(os.path.join(save_folder, standard_category))
        if not os.path.exists(os.path.join(save_folder, standard_category, tool_name)):
            os.mkdir(os.path.join(save_folder, standard_category, tool_name))    
        json.dump(cache, open(os.path.join(save_folder, standard_category, tool_name, api_name+".json"), "w"), indent=4)
    except Exception as e:
        print(f"Save cache failed: {e}")

def fake_response_function_chat(api_example, tool_input, api_doc, source_api_doc, graph):
    '''
    api_example: list of tuple, [(input, output), ...]
    tool_input: dict, input of the tool
    api_doc: dict, api document
    '''
    system_prompt = '''
1. Your task is to determine whether the source_attribute in the response from the source API is compatible with the api_input of the target API. Then, craft a JSON formatted response that aligns with the expected output of the API, guided by the provided examples.
2. For your judgment, we will provide descriptions of tool_description, API Documentation, source_attribute and target_attribute of both APIs.
3. The judgment is a two-step process. In the first step, determine whether the two attributes are compatible based on a deep understanding of the source_attribute and target_attribute. Determine whether the source_attribute and target_attribute are compatible through attribute descriptions.
4. The second step is to determine whether the input of the target API is compatible with the intent of the target API.
5. If both steps are considered compatible, follow the <Output format for True> to output the result. If not, follow the <Output format for False> to output the result.\n
Your responses must adhere to a specific JSON structure, which is as follows:\n
<Output format for True>
{
"error": "",
"response": "<Your_Response>"
}\n

<Your_Response> must be in json format.
Use all names of output_component in API Documentation as keys of <Your_Response>.
Create appropriate values for the corresponding keys, considering the API Input.

<Output format for False>
{
"error": “graph validation error”,
"response": "<Your_Response>"
}\n

The response field should contain the content you formulate based on the API's functionality and the input provided. Ensure that your responses are meaningful, directly addressing the API's intended functionality. If the provided examples are mostly error messages or lack substantial content, use your judgment to create relevant and accurate responses. The key is to maintain the JSON format's integrity while ensuring that your response is an accurate reflection of the API's intended output within the tool.\n
Please note that your answer should not contain anything other than a json format object, which should be parsable directly to json.
Note that:

- your response should be around 100 to 200 words, containing rich information given the api input parameters. Keep Your answer short and simple.
- your response must be effective and have practical content.
- if the api response example if null or ineffective, ignore the example and give your independent response.  
'''
    system_prompt = {"role": "system", "content": system_prompt}
    # user prompt, truncated to 2048 characters if too long
    user_prompt ="source API Documentation:"+str(source_api_doc)+"\n"+"target API Documentation:"+str(api_doc)+"\n"+"API Examples:"+str(api_example)[:2048]+"\n"+"API Input:"+str(tool_input)+"\n" + "attribute descriptions"+str(graph.edge)+"\n"
    user_prompt = {"role": "user", "content": user_prompt}

    # client = OpenAI(
    #     api_key = OPENAI_API_KEY,
    #     base_url = OPENAI_API_BASE,
    # )
    openai.api_key = OPENAI_API_KEY
    openai.api_base = OPENAI_API_BASE
    
    max_retries = 3 
    flag = False
    for attempt in range(max_retries):
        # response = client.chat.completions.create(
        response = openai.chat.completions.create(
            model = CONFIG['model'],
            messages=[system_prompt, user_prompt],
            max_tokens = 1024,
            temperature=CONFIG['temperature'],
            response_format={"type": "json_object"},
        )
        print("Response:",response)
        result = response.choices[0].message.content
        
        # if len(result)>3000:
        #     fake_error = {
        #         "error": "Failed to generate fake response",
        #         "response": "",
        #     }
        #     return json.dumps(fake_error)
        
        if "```json" in result:
            result = result.replace("```json", "").replace("```", "").strip()
        if is_valid_json(result):
            flag = True
            break
        print(f"Invalid JSON response on attempt {attempt + 1}. Retrying...")
        # break
        time.sleep(1)  # Optional delay between retries

    if flag:
        return result
    else:
        fake_error = {
            "error": "Failed to generate fake response",
            "response": "",
        }
        return json.dumps(fake_error)

if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=CONFIG['port'])
    
    
    
    
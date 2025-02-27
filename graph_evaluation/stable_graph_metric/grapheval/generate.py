import time 
import json
from util import get_message_content_from_chatgpt, get_api_doc_by_api_name, preprocessed_response, is_valid_json


def form_structure_for_api_data(filtered_api_doc): 
    ''' 
    make output format 
    ex) {key : ''}
    '''
    keys = ['required_parameters', 'output_components']
    structure = {'api_name':filtered_api_doc['api_name'], 'api_description':filtered_api_doc['api_description']}
    for key in keys:
        if len(filtered_api_doc[key]) != 0:
            structure[key] = {item['name']: '' for item in filtered_api_doc[key]}
        else:
            structure[key] = {}
    return structure



def fill_common_value_for_source_target(edge, final_data_structure, filtered_api_docs):
    ''' 
    input : graph, structure, document
    output : 
    common value -> 두 API에서 공유하는 real value
    final_data_structure -> 두 API의 required_parameters와 output_format이 ""이면서 commmon value만 채워져 있음
    '''
    source_api_name = edge['source']
    target_api_name = edge['target']
    source_param_name = edge['source_attribute']['name']
    target_param_name = edge['target_attribute']['name']
    
    attributes = {}
    attributes[source_api_name] = []
    attributes[target_api_name] = []
    attributes[source_api_name].append(edge['source_attribute'])
    attributes[target_api_name].append(edge['target_attribute'])

    for component in filtered_api_docs[source_api_name]['output_components']:
        if component['name'] == source_param_name:
            attributes[source_api_name].append(component)
            break

    for component in filtered_api_docs[target_api_name]['required_parameters']:
        if component['name'] == target_param_name:
            attributes[target_api_name].append(component)
            break
    
    prompt = f"""
<Instruction>
1. You are an intelligent and creative data generator.
2. In <List> section below, you will be given two parameter descriptions.
3. two parameter description refer to the same parameter.
4. Creatively generate mock data for the parameter that encompasses all descriptions and data types.
5. Only generate the mock value for the parameter and absolutely no other text, such as descriptions or other strings.
6. If the names of the parameters in <List> are very different, create them by referring only to the description of the former. Only one value should be created. Do not create them in json format.
<List>
{str(attributes)}
"""
    common_value = get_message_content_from_chatgpt(prompt=prompt)
    print('common_value', common_value)
    # if edge['source'] == api_name:
    final_data_structure[source_api_name]['output_components'][source_param_name] = common_value
        # if target_param_name in final_data_structure[target_api_name]['required_parameters']:
    final_data_structure[target_api_name]['required_parameters'][target_param_name] = common_value
    return final_data_structure, common_value




def prompt_from_api_doc(filtered_api_doc:dict, final_data_structure:dict,) -> str:
    '''
    target API의 input을 생성하는 프롬프트 
    blank 상태의 final_data_structure['required_parameters']를 제공하고 값을 채워넣으라고 지시
    '''
    
    final_data_structure_for_prompt = str(final_data_structure['required_parameters']).replace("''","<<You generate this section>>")
    
    system_prompt = f"""
You are an intelligent and creative data generator.
I will give you an <<API documentation>> containing the following information below.
tool_name: This is name of tool. Tool is a service provider where provides numerous APIs regarding the service.
tool_description: The description of tool.
api_name: The name of API that belongs to the tool.
api_description: API description.
required_parameters: Required input parameters to call the API. It contains parameter description, type and default value(optional).

<<API documentation>>
tool_name: {filtered_api_doc['tool_name']}
tool_description: {filtered_api_doc['tool_description']}
api_name: {filtered_api_doc['api_name']}
api_description: {filtered_api_doc['api_description']}
required_parameters: {str(filtered_api_doc['required_parameters'])}

<<Instruction>>
1. Refer to the above information and thoroughly understand the API documentation I've provided.
2. If <<Todo>>'s value is blank, generate parameter value. You can get a hint from default value in required_parameters when you generate parameter values, but do not just copy the default value. Be creative!
3. However, the data that you generate should not contradict the values that have already been filled in.
For example, if the filled parameter 'max_weight' is set to '20' (interpret this string as a number), then you must generate a value for the parameter 'min_weight' that is less than 20.
**IMPORTANT**
4. You MUST NOT change or hallucinate parameter values that have been filled already in <<Todo>>.
5. ONLY <<You generate this section>> should be replaced with 'plausible' data. The meaning of 'plausible' data is that, for example, if you need to generate a URL, www.example.com is not plausible. Generate something that is likely to exist, even if it doesn't.
6. ONLY generate below <<Todo>> section.
7. Parameter values cannot be None and required_parameter cannot be blank.

<<Todo>>
```json
{final_data_structure_for_prompt}
```
"""
    return system_prompt



def generate_data(graph, after_coor_before_emb) -> dict:
    ''' 
    generated_data : target API의 input이 gpt에 의해 생성된 결과
    filtered_api_docs : API documents
    final_data_structure : blank인 API data
    '''
    apis = after_coor_before_emb
    filtered_api_docs = {}
    final_data_structure = {}
    for api_name in graph['nodes']:
        filtered_api_doc = get_api_doc_by_api_name(apis, api_name)
        filtered_api_docs[api_name] = filtered_api_doc
        final_data_structure[api_name] = form_structure_for_api_data(filtered_api_doc) 
    final_data_structure, common_value = fill_common_value_for_source_target(graph['edge'][0], final_data_structure, filtered_api_docs)
    
    max_retries = 3 
    target_api_name = graph['nodes'][1]
    for attempt in range(max_retries):
        generated_data = []

        prompt = prompt_from_api_doc(filtered_api_docs[target_api_name], final_data_structure[target_api_name])
        response_content = get_message_content_from_chatgpt(prompt=prompt)
        response_content = preprocessed_response(response_content)
        
        if is_valid_json(response_content): 
            response_in_json = json.loads(response_content)
            generated_data.append(response_in_json)
            return generated_data, final_data_structure, filtered_api_docs, common_value
        else: 
            print(f"Invalid JSON response on attempt {attempt + 1}. Retrying... : ", response_content)
            time.sleep(1)  # Optional delay between retries
        if attempt == max_retries-1:
            generated_data = False


    return generated_data, final_data_structure, filtered_api_docs, common_value



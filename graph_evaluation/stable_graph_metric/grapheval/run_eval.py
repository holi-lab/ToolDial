import json,sys,yaml,openai,os
from tqdm import tqdm

from util import valid_edge_idx, additional_info, convert_list_to_graph, main_server_call,split_list_and_save_json
from generate import generate_data

config_file='./config.yml'
CONFIG = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
print(CONFIG)

TOOL_INFO_PATH = CONFIG['tool_info_path']
current_path = sys.argv[2]
SAVED_PATH = CONFIG['saved_path']

with open(TOOL_INFO_PATH,'r') as f:
    after_coor_before_emb = json.load(f)
    
with open(f"stable_graph_metric/edge_files/graph_for_evaluation/{current_path}/all_edge_list_graph.json",'r') as f:
    all_edge_list_graph = json.load(f)

if __name__ == "__main__":
    
    with open(f"stable_graph_metric/edge_files/graph_for_evaluation/{current_path}/split_file/split_{sys.argv[1]}.json",'r') as f:
        index_combinations = json.load(f)
    
    results_save = {"graphs": []}
    graph_valid = []
    graph_not_valid = []
    stable_tool_bench_json_error = []
    graph_evaluation = []

    for i in tqdm(range(0, len(index_combinations))):
        a, b, c = index_combinations[i]
        if a in [210,211,212,213,214,215]:
            continue
        test_graph = convert_list_to_graph(all_edge_list_graph[a][b][c])

        # target input generation      
        generated_data, final_data_structure, filtered_api_docs, common_value = generate_data(test_graph, after_coor_before_emb)
        
        if generated_data == False:
            stable_tool_bench_json_error.append(index_combinations[i])
            graph_evaluation.append({
            "ids" : index_combinations[i],
            "graph" : test_graph,
            "common_value" : common_value,
            "required_parameters" : None,
            "graph_valid" : False,
            "graph_not_valid" : False,
            "stable_tool_bench_json_error" : True,
            "response" : output
            })
            continue
        
        source_info = additional_info(filtered_api_docs)
        # target API call
        # print(test_graph['nodes'][1])
        output = main_server_call(test_graph['nodes'][1], generated_data[0], source_info, test_graph)
        
        try:
            output = json.loads(output)
        except json.JSONDecodeError :
            # print("stable_tool_bench_json_error")
            stable_tool_bench_json_error.append(index_combinations[i])
            graph_evaluation.append({
            "ids" : index_combinations[i],
            "graph" : test_graph,
            "common_value" : common_value,
            "required_parameters" : generated_data[0],
            "graph_valid" : False,
            "graph_not_valid" : False,
            "stable_tool_bench_json_error" : True,
            "response" : output
            })
            continue
            
        if output['error'] != "":
            # print('graph_not_valid')
            graph_not_valid.append(index_combinations[i])
            graph_evaluation.append({
            "ids" : index_combinations[i],
            "graph" : test_graph,
            "common_value" : common_value,
            "required_parameters" : generated_data[0],
            "graph_valid" : False,
            "graph_not_valid" : True,
            "stable_tool_bench_json_error" : False,
            "response" : output
            })
        else:
            # print('graph_valid')
            graph_valid.append(index_combinations[i])
            graph_evaluation.append({
            "ids" : index_combinations[i],
            "graph" : test_graph,
            "common_value" : common_value,
            "required_parameters" : generated_data[0],
            "graph_valid" : True,
            "graph_not_valid" : False,
            "stable_tool_bench_json_error" : False,
            "response" : output
            })
            
    # all results saved
    results_save = {"graphs": []}
    results_save['graphs'].append(graph_evaluation) 
    
    file_name = f"stable_graph_metric/edge_files/graph_for_evaluation/{current_path}/result_file/result_{sys.argv[1]}.json"
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(results_save, f, ensure_ascii=False, indent=4)

    print('graph_valid : ', len(graph_valid))
    print('graph_not_valid : ',len(graph_not_valid))
    print('stable_tool_bench_json_error : ',len(stable_tool_bench_json_error))

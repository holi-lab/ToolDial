import json,sys,os

current_path = sys.argv[1]

with open(f"stable_graph_metric/edge_files/graph_for_evaluation/{current_path}/eval_result_edge.json",'r') as f:
    edges = json.load(f)

pair_list = []
for edge in edges:
    pair = [(edge['ids'][0],edge['ids'][1])]
    pair_list.append(pair)


print(f"Length of pair: {len(pair_list)}")
# with open(f"stable_graph_metric/saved_results/{current_path}/pair_list.json",'w') as f:
#     json.dump(pair_list,f,indent=4)
    
with open(f"../dialogue_generation/pair_list/pair_list.json",'w') as f:
    json.dump(pair_list,f,indent=4)

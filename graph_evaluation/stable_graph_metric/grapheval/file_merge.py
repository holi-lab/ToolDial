import json,sys,yaml,openai,os

result = {"graphs":[]}
current_path = sys.argv[2]
for idx in range(int(sys.argv[1])):
    try:
        with open(f"stable_graph_metric/edge_files/graph_for_evaluation/{current_path}/result_file/result_{idx}.json",'r') as f:
            tmp = json.load(f)
    except:
        continue
    for edge in tmp['graphs'][0]:
        result['graphs'].append(edge)
        
graph_size = len(result['graphs'])
valid_edge = []
for i in range(graph_size):
    if result['graphs'][i]['graph_valid'] == True:
         valid_edge.append(result['graphs'][i])

print(graph_size)

print(f"Accuracy on {current_path}:{len(valid_edge)/graph_size}")
print(f"Length of Valid edge:{len(valid_edge)}")
            
with open(f"stable_graph_metric/edge_files/graph_for_evaluation/{current_path}/eval_result_edge.json",'w') as f:
    json.dump(valid_edge,f,indent=4)
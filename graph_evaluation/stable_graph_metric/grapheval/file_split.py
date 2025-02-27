import yaml,sys,json,random,os
import openai
from tqdm import tqdm
from util import split_list_and_save_json,valid_edge_idx

file_path = sys.argv[2]

with open(f"stable_graph_metric/edge_files/graph_for_evaluation/{file_path}/all_edge_list_graph.json",'r') as f:
    all_edge_list_graph = json.load(f)

index_combinations = valid_edge_idx(all_edge_list_graph)

print(len(index_combinations))
split_list_and_save_json(index_combinations,int(sys.argv[1]),output_dir=f"stable_graph_metric/edge_files/graph_for_evaluation/{file_path}/split_file")

if "result_file" not in os.listdir(f"stable_graph_metric/edge_files/graph_for_evaluation/{file_path}"):
    os.mkdir(f"stable_graph_metric/edge_files/graph_for_evaluation/{file_path}/result_file")
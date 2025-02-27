import json,re,random,sys
from tqdm import tqdm
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")

from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
)

import yaml
with open("../config.yml") as f:
    config = yaml.load(f,Loader=yaml.SafeLoader)
openai_key = config['api_key']
device = config['device']

with open("../../after_coor_before_emb.json",'r') as f:
    api_list = json.load(f)
api_list_dict = {api['full_name']:api for api in api_list}

with open(f"../pair_list/all_edge_list_graph.json",'r') as f:
    all_edge_list_graph = json.load(f)
    
with open(f"../pair_list/normal_graph2idx.json",'r') as f:
    normal_graph2idx = json.load(f)
    
with open(f"../pair_list/normal_idx2igraph.json",'r') as f:
    normal_idx2igraph = json.load(f)

def Dial(prompt):
    model_name = "gpt-4o"
    llm = ChatOpenAI(model_name=model_name,
                    temperature=0,
                    max_tokens=4096,
                    openai_api_key=openai_key)
    generated = llm([HumanMessage(content=prompt)]).content
    return generated

def dialogue_geneator(prompt_list):
    dial_list = []
    for prompt in tqdm(prompt_list):
        dial_list.append(Dial(prompt))
    return dial_list

number = int(sys.argv[1])
file = sys.argv[2]
with get_openai_callback() as cb:

    with open(f"{file}_scen_{number}.json",'r') as f:
        scenario_list = json.load(f)
        
    file_length = len(scenario_list)
    dial_list = dialogue_geneator(scenario_list)
    
    with open(f"raw_dial_{file}_{number}.json",'w') as f:
        json.dump(dial_list,f,indent=4)
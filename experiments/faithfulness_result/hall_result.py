import json,re,os,yaml

with open("hall_result.yml") as f:
    config = yaml.load(f,Loader=yaml.SafeLoader)
model_name = config['model_name']

file_list = os.listdir(f"{model_name}")
print(f"{model_name}")

results=[]
for file in file_list:
    if ".json" not in file:
        continue
    with open(f"{model_name}/{file}",'r') as f:
        results+=json.load(f)

score = 0
cnt = 0
for res in results:
    if "- Action:response_fail" not in res['input']:
        score+=res['correct']
        cnt+=1
    
print(f"{model_name} Faithfulness score:",score/cnt)
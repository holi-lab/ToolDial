import json,sys

with open("pair_list/pair_list_original.json",'r') as f:
    pair_list = json.load(f)

split = int(sys.argv[1])
check_points = [int((len(pair_list)/split)*i) for i in range(split+1)]

for i in range(split):
    tmp_list = []
    print(check_points[i],check_points[i+1])
    for idx,inst in enumerate(pair_list[check_points[i]:check_points[i+1]]):
        tmp_list.append(inst)
    with open(f"pair_list/sep_pair_{i}.json",'w') as f:
        json.dump(tmp_list,f,indent=4)
import json

dial_list = []
category = ['yesyes','yeszero','zeroyes','zerozero']

for cat in category:
    with open(f"sample_dialogue_{cat}.json",'r') as f:
        tmp = json.load(f)
    dial_list+=tmp

print("Dialogue Data Complete")
print(f"Cnt: {len(dial_list)}")

with open("total_dialogue.json",'w') as f:
    json.dump(dial_list,f,indent=4)
import json,re
import os,yaml
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

def normalize_action(text):
    lower_text = text.lower()
    normalized_text = re.sub(r'[^a-z]', '', lower_text)
    return normalized_text.replace("action","").replace("api","")


with open("action_error_analysis.yml") as f:
    config = yaml.load(f,Loader=yaml.SafeLoader)
model_name = config['model_name']
is_give_label = config['is_give_label']
label_dir = "withgt" if is_give_label else "wogt"

file_list = os.listdir(f"{model_name}/{label_dir}")
print(f"{model_name}/{label_dir}")

results=[]
for file in file_list:
    if ".json" not in file:
        continue
    with open(f"{model_name}/{label_dir}/{file}",'r') as f:
        results+=json.load(f)
print(len(results))

correct = 0
cnt=0
error = []
label = []
predict = []
for ent in results:
    label.append(ent['label'])
    predict.append(ent['process'])
    if ent['correct']:
        correct+=1
    else:
        
        error.append(ent)
    cnt+=1

label = [normalize_action(act) for act in label]
predict = [normalize_action(act) for act in predict]

print(f"Action accuracy: {correct/cnt}")
print(f"Length of error:{cnt-correct}")
print(metrics.classification_report(label, predict,digits=3,labels = list(set(label))))
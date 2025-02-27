from sklearn import metrics
import pandas as pd
import json, re, yaml
import warnings
warnings.filterwarnings("ignore")

def normalize_action(text):
    lower_text = text.lower()
    normalized_text = re.sub(r'[^a-z]', '', lower_text)
    return normalized_text.replace("action","").replace("api","")


with open("action_gpt_error_analysis.yml") as f:
    config = yaml.load(f,Loader=yaml.SafeLoader)
model = config['model_name']
is_label = int(config['is_give_label'] )
label_dir = "withgt" if is_label else "wogt"

model_score = {"wgt":"","wogt":""}
model_accuracy = {"wgt":"","wogt":""}
        
with open(f"{model}/{label_dir}/model:{model},is_give_label:{str(is_label)},accumulation:0_label.json",'r') as f:
    label = json.load(f)
label = [normalize_action(l['label']) for l in label]
category = list(set(label))
labels = ["response","responsefail","request","retrievercall","clarify","systembye","suggest","call"]
with open(f"{model}/{label_dir}/model:{model},is_give_label:{str(is_label)},accumulation:0_pred.json",'r') as f:
    predict = json.load(f)

current_label = "wgt" if is_label else "wogt"
corr = 0
for l,p in zip(label,predict):
    if l == p:
        corr+=1
model_accuracy[current_label] = corr/len(label)

report = metrics.classification_report(label, predict, output_dict=True)   
df = pd.DataFrame(report).transpose()

score = {}
for l in category:
    score[l] = df['f1-score'][l]
model_score[current_label] = score

if is_label:
    print("F1 score for each actions:")
    for action in model_score['wgt']:
        print(f"{action}:",model_score['wgt'][action])
    print("---------------------")
    print(f"Action Prediction Accuracy of {model} in {label_dir} setting:",model_accuracy['wgt'])
else:
    print("F1 score for each actions:")
    for action in model_score['wogt']:
        print(f"{action}:",model_score['wogt'][action])
    print("---------------------")
    print(f"Action Prediction Accuracy of {model} in {label_dir} setting:",model_accuracy['wogt'])
import json,re,sys,yaml

class DSTErrorAnalysis:
    def __init__(self,model_name,is_label):
        self.is_give_label = 1 if is_label else 0
        self.model_name = model_name
        label_dir = "withgt" if self.is_give_label else "wogt"
        with open(f"{model_name}/{label_dir}/model:{model_name},is_give_label:{self.is_give_label},accumulation:0_label.json",'r') as f:
            label = json.load(f)
        with open(f"{model_name}/{label_dir}/model:{model_name},is_give_label:{self.is_give_label},accumulation:0_pred.json",'r') as f:
            pred_label = json.load(f)
        self.pred_label = [{"dial":label[idx]['dial'],"label":label[idx]['label'],"predict":pred_label[idx]} for idx in range(len(label))]
        self.error_length = 0
        self.error = []
        self.error_type = {
            "generation_error":[],
            "predict_false_label_true":[],
            "label_false_predict_true":[],
            "format":[],
            "input_parameter_slot_error":[],
            "value_error":[]
        }
        self.report_error()
        
    def pop_idx(self,lst,indices):
        indices = sorted(indices, reverse=True)
        for index in indices:
            if 0 <= index < len(lst):
                lst.pop(index)
                
    def normalize_value(self,value):
        return re.sub(r'[^a-z0-9]', '', value.lower())
    
    def normalize_json(self,data):
        def traverse_and_normalize(value):
            if isinstance(value, dict):
                return {traverse_and_normalize(k): traverse_and_normalize(v) for k, v in value.items()}
            elif isinstance(value, list):
                value = str(value).lower()
                return re.sub(r'[^a-z0-9]', '', value.lower())
            elif isinstance(value, str):
                return re.sub(r'[^a-z0-9]', '', value.lower())
            elif isinstance(value,int) or isinstance(value,float):
                value = str(value)
                return re.sub(r'[^a-z0-9]', '', value.lower())
            else:
                return value
        return traverse_and_normalize(data)
    
    def error_classification(self):
        corr=0
        total = 0
        for idx,result in enumerate(self.pred_label):
            if self.normalize_json(result['label']) == self.normalize_json(result['predict']):
                corr +=1
            else:
                self.error.append(result)
            total+=1
        print(f"Model name: {self.model_name}",end = " -- ")
        if self.is_give_label:
            print("With GT")
        else:
            print("W/O GT")
        
        print(f"DST accuracy: {corr/total}")
        print(f"Length of Error: {len(self.error)}")
        print("*******************")
        self.error_length = len(self.error)
        
    def generation_error(self):
        tmp_idx = []
        for idx,err in enumerate(self.error):
            if not isinstance(err['predict'],dict):
                self.error_type['generation_error'].append(err)
                tmp_idx.append(idx)
        self.pop_idx(self.error,tmp_idx)
                
    def predict_false_label_true(self):
        tmp_idx = []
        for idx,err in enumerate(self.error):
            if err['predict'] == {'api_confirmed': 'false', 'api_status': 'none'}:
                if err['label'] != {'api_confirmed': 'false', 'api_status': 'none'}:
                    self.error_type['predict_false_label_true'].append(err)
                    tmp_idx.append(idx)
        self.pop_idx(self.error,tmp_idx)
                
    def label_false_predict_true(self):
        tmp_idx = []
        for idx,err in enumerate(self.error):
            if err['label'] == {'api_confirmed': 'false', 'api_status': 'none'}:
                if err['predict'] != {'api_confirmed': 'false', 'api_status': 'none'}:
                    self.error_type['label_false_predict_true'].append(idx)
                    tmp_idx.append(idx)
        self.pop_idx(self.error,tmp_idx)
    
    def format(self):
        tmp_idx = []
        for idx,err in enumerate(self.error):
            if "api_confirmed" in err['predict']:
                if err['predict']['api_confirmed'] == 'true':
                    if "api_name" in err['predict']['api_status'] and "required_parameters" in err['predict']['api_status'] and "optional_parameters" in err['predict']['api_status']:
                        continue
                    else:
                        self.error_type['format'].append(err)
                        tmp_idx.append(idx)
                else:
                    if "api_status" in err['predict']:
                        continue
                    else:
                        self.error_type['format'].append(err)
                        tmp_idx.append(idx)
            else:
                self.error_type['format'].append(err)
                tmp_idx.append(idx)
        self.pop_idx(self.error,tmp_idx)
                
    def input_parameter_slot_error(self):
        for idx,err in enumerate(self.error):
            if err['predict']['api_confirmed'] == "true": ## 여기서는 slot error
                if list(err['predict']['api_status']['required_parameters']) == list(err['label']['api_status']['required_parameters']) and list(err['predict']['api_status']['optional_parameters']) != list(err['label']['api_status']['optional_parameters']):
                    self.error_type['input_parameter_slot_error'].append(err)
                elif list(err['predict']['api_status']['required_parameters']) != list(err['label']['api_status']['required_parameters']) and list(err['predict']['api_status']['optional_parameters']) == list(err['label']['api_status']['optional_parameters']):
                    self.error_type['input_parameter_slot_error'].append(err)
                elif list(err['predict']['api_status']['required_parameters']) != list(err['label']['api_status']['required_parameters']) and list(err['predict']['api_status']['optional_parameters']) != list(err['label']['api_status']['optional_parameters']):
                    self.error_type['input_parameter_slot_error'].append(err)
                label_dict = err['label']['api_status']['required_parameters']|err['label']['api_status']['optional_parameters']
                try:
                    predict_dict = err['predict']['api_status']['required_parameters']|err['predict']['api_status']['optional_parameters']
                except:
                    self.error_type['value_error'].append(err)
                
                # print(label_dict)
                
                cnt = 0
                for slot in label_dict:
                    if slot in predict_dict: ## 일단 label의 slot이 predict에 있어야 함
                        if isinstance(label_dict[slot],dict) or isinstance(label_dict[slot],list):
                            compare_label_slot = self.normalize_json(label_dict[slot])
                        elif isinstance(label_dict[slot],str):
                            compare_label_slot = self.normalize_value(label_dict[slot])
                        else:
                            compare_label_slot = label_dict[slot]
                            
                        if isinstance(predict_dict[slot],dict) or isinstance(predict_dict[slot],list):
                            compare_predict_slot = self.normalize_json(predict_dict[slot])
                        elif isinstance(predict_dict[slot],str):
                            compare_predict_slot = self.normalize_value(predict_dict[slot])
                        else:
                            compare_predict_slot = predict_dict[slot]
                        
                        if compare_label_slot == compare_predict_slot:
                            cnt+=1
                if cnt != len(label_dict):
                    self.error_type['value_error'].append(err)
        
    def report_error(self):
        self.error_classification()
        self.generation_error()
        self.predict_false_label_true()
        self.label_false_predict_true()
        self.format()
        self.input_parameter_slot_error()
        total_sum = 0
        for error in self.error_type:
            print(f"{error}: {len(self.error_type[error])} - {round(len(self.error_type[error])/self.error_length,3)}")
            total_sum+=len(self.error_type[error])
        print("*******************")
        print(f"Final Classified Error: {total_sum}")
        
def normalize_json(data):
    def traverse_and_normalize(value):
        if isinstance(value, dict):
            return {traverse_and_normalize(k): traverse_and_normalize(v) for k, v in value.items()}
        elif isinstance(value, list):
            value = str(value).lower()
            return re.sub(r'[^a-z0-9]', '', value.lower())
        elif isinstance(value, str):
            return re.sub(r'[^a-z0-9]', '', value.lower())
        elif isinstance(value,int) or isinstance(value,float):
            value = str(value)
            return re.sub(r'[^a-z0-9]', '', value.lower())
        else:
            return value
    return traverse_and_normalize(data)
with open("dst_gpt_error_analysis.yml") as f:
    config = yaml.load(f,Loader=yaml.SafeLoader)
model = config['model_name']
is_label = int(config['is_give_label'] )

DSTErrorAnalysis(model,is_label)
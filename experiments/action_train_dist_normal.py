import os
import json,random,re,yaml
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import bitsandbytes
import datetime
from nlp import finetune_form, action_form_action_token, action_form_action_normal, action_form_thought_normal, action_form_thought_token
from prompts import add_tokens
import traceback
from torch.optim.lr_scheduler import StepLR
from flash_attn.modules.mha import FlashSelfAttention

def set_seed(seed=1):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def replace_with_flash_attention(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):  # Check for standard attention layers
            setattr(
                model, name, 
                FlashSelfAttention(causal=True)  # Causal=True for autoregressive models
            )


def normalize_action(text):
    lower_text = text.lower()
    normalized_text = re.sub(r'[^a-z]', '', lower_text)
    return normalized_text.replace("action","").replace("api","")
    
def action_preprocess(text,is_thought,is_normal):
    text = text.lower()
    if not is_normal:
        if is_thought:
            text = text.split("<end_of_system_thought>")[1]
            text = text.replace("<end_of_system_action>","").replace("<start_of_system_action>","").replace("<|eot_id|>","").strip().replace("\n","")
        else:
            text = text.replace("<end_of_system_action>","").replace("<start_of_system_action>", "").strip().replace("\n","")
    else:
        if is_thought:
            text = text.split("- action:")[1]
        else:
            text = text.replace("- action:","")
    return text
    
def collate_function(batches,tokenizer):
    history_length = []
    for item in batches:
        history_length.append(len(tokenizer.encode(item['history'])))
    x_texts = [item['X'] for item in batches]
    x_encoded = tokenizer(x_texts, return_tensors='pt', padding='longest')
    return {"history":torch.tensor(history_length),"input_ids":x_encoded['input_ids'],"attention_mask":x_encoded['attention_mask']}

def action_collate_fn(batches):
    x_texts,y = [],[]
    for item in batches:
        x_texts.append(item['X'])
        y.append(item['y'].strip())
    return {"input":x_texts,"output":y}

def action_infer_single_mistral(input_text, model, rank, tokenizer_inf):
    split_text = "[/INST] "
    input_encoding = tokenizer_inf(input_text, return_tensors="pt", padding=True).to(rank)
    input_encoding['max_new_tokens'] = 300
    input_encoding['num_return_sequences'] = 1
    output = model.module.generate(**input_encoding)
    output = tokenizer_inf.decode(output[0])
    output = output.split(split_text)[1].replace("</s>","")
    return output

def action_infer_single_qwen(input_text, model, rank, tokenizer_inf):
    split_text = "<|im_start|>assistant\n"
    input_encoding = tokenizer_inf(input_text, return_tensors="pt", padding=True).to(rank)
    input_encoding['max_new_tokens'] = 300
    input_encoding['num_return_sequences'] = 1
    output = model.module.generate(**input_encoding)
    output = tokenizer_inf.decode(output[0])
    output = output.split(split_text)[1].replace("<|im_end|>","")
    return output

def action_infer_single_code(input_text, model, rank, tokenizer_inf):
    split_text = "[/INST]"
    input_encoding = tokenizer_inf(input_text, return_tensors="pt", padding=True).to(rank)
    input_encoding['max_new_tokens'] = 300
    input_encoding['num_return_sequences'] = 1
    input_encoding['pad_token_id'] = tokenizer_inf.encode("<pad>")[1]
    input_encoding['eos_token_id'] = tokenizer_inf.encode("</s>")[1]
    
    output = model.module.generate(**input_encoding)
    output = tokenizer_inf.decode(output[0])
    output = output.split(split_text)[1].strip().replace("</s>", "")
    return output

def action_infer_single(input_text, model, rank, tokenizer_inf):
    split_text = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    input_encoding = tokenizer_inf(input_text, return_tensors="pt", padding=True).to(rank)
    input_encoding['max_new_tokens'] = 300
    input_encoding['num_return_sequences'] = 1
    input_encoding['pad_token_id'] = tokenizer_inf.encode("<pad>")[1]
    input_encoding['eos_token_id'] = tokenizer_inf.encode("<|eot_id|>")[1]
    
    output = model.module.generate(**input_encoding)
    output = tokenizer_inf.decode(output[0])
    output = output.split(split_text)[1].replace("<|eot_id|>", "")
    return output

##########################
# Setup function for DDP
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size,timeout=datetime.timedelta(seconds=36000))
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


##########################
# Check GPU availability
def check_gpu_availibility():
    if torch.cuda.is_available():
        num_cuda_devices = torch.cuda.device_count()
        print(f'Number of CUDA Device available: {num_cuda_devices}')
        for i in range(num_cuda_devices):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"Is available: {torch.cuda.is_available()}")
    else:
        print("No cuda devices available")


##########################
# Load data only on rank 0 and broadcast to other ranks
def load_data(train_file,test_file,rank, world_size,tokenizer,is_thought,is_normal,is_give_label):
    if rank == 0:
        random.seed(1)
        print("Train data loading....")
        with open(train_file, 'r') as f:
            train_dialogue_list = json.load(f)
            # train_dialogue_list = random.sample(train_dialogue_list,10)
        print(f"Length of train: {len(train_dialogue_list)}")
        print("Test data loading....")
        with open(test_file, 'r') as f:
            val_dialogue_list = json.load(f)
            # val_dialogue_list = random.sample(val_dialogue_list,10)
        print(f"Length of Test: {len(val_dialogue_list)}")
        
        if not is_normal:
            train_sft_form_list = [{"dialogue": finetune_form(dial['dialogue']), "action_list": dial['action_list']} for dial in tqdm(train_dialogue_list)]
            if is_thought: ## token_thought
                train_dst = []
                for dial in train_sft_form_list:
                    train_dst+=action_form_thought_token(dial,tokenizer,is_give_label=is_give_label)
                val_sft_form_list = [{"dialogue": finetune_form(dial['dialogue']), "action_list": dial['action_list']} for dial in tqdm(val_dialogue_list)]
                val_dst = []
                for dial in val_sft_form_list:
                    val_dst+=action_form_thought_token(dial,tokenizer,is_give_label=is_give_label,is_val=True)
            else: ## token_action
                train_dst = []
                for dial in train_sft_form_list:
                    train_dst+=action_form_action_token(dial,tokenizer,is_give_label=is_give_label)
                val_sft_form_list = [{"dialogue": finetune_form(dial['dialogue']), "action_list": dial['action_list']} for dial in tqdm(val_dialogue_list)]
                val_dst = []
                for dial in val_sft_form_list:
                    val_dst+=action_form_action_token(dial,tokenizer,is_give_label=is_give_label,is_val=True)
        else:
            if is_thought: ## normal_thought
                train_dst = []
                for dial in train_dialogue_list:
                    train_dst+=action_form_thought_normal(dial,tokenizer,is_give_label=is_give_label)
                    
                val_dst = []
                for dial in val_dialogue_list:
                    val_dst+=action_form_thought_normal(dial,tokenizer,is_give_label=is_give_label,is_val=True)
            else: ## normal_action
                train_dst = []
                for dial in train_dialogue_list:
                    train_dst+=action_form_action_normal(dial,tokenizer,is_give_label=is_give_label)
                val_dst = []
                for dial in val_dialogue_list:
                    val_dst+=action_form_action_normal(dial,tokenizer,is_give_label=is_give_label,is_val=True)
    else:
        train_dst = None
        val_dst = None
    # Broadcast the data from rank 0 to all other ranks
    train_dst = [train_dst]
    val_dst = [val_dst]
    dist.broadcast_object_list(train_dst, src=0)
    dist.broadcast_object_list(val_dst, src=0)

    # Unpack the broadcasted data on all ranks
    train_dst = train_dst[0]
    val_dst = val_dst[0]

    return train_dst, val_dst



def validate(rank, 
            model, 
            val_dataset,
            world_size, 
            tokenizer_inf,
            is_normal,
            is_thought,
            file_name,
            infer_function,
            result_save_path):
    model.eval()
    local_results = []
    
    total_samples = len(val_dataset)
    samples_per_gpu = total_samples // world_size
    remainder = total_samples % world_size
    
    # 각 GPU에 할당할 데이터의 시작 및 끝 인덱스 계산
    start_index = rank * samples_per_gpu + min(rank, remainder)
    end_index = start_index + samples_per_gpu + (1 if rank < remainder else 0)

    # 각 GPU에 할당된 데이터 부분을 Subset으로 만듭니다.
    val_subset = Subset(val_dataset, list(range(start_index, end_index)))
    val_loader = DataLoader(val_subset, batch_size=1, collate_fn=action_collate_fn)

    correct, total = 0,0
    print(f"Rank {rank} processing samples {start_index} to {end_index}")
    
    with torch.no_grad():
        for batch in tqdm(val_loader,dynamic_ncols=False, ascii=True):
            label = batch['output'][0].strip()  # batch_size=1이므로 첫 번째 항목만 가져옴
            y_hat = infer_function(batch['input'][0], model, rank, tokenizer_inf)
            predict = action_preprocess(y_hat,is_thought,is_normal)
            if normalize_action(label) in normalize_action(predict):
                correct+=1
            total+=1
            local_results.append({
                'input': batch['input'][0],
                'generated': y_hat,
                'process': predict,
                'label': label,
                'correct': normalize_action(label) in normalize_action(predict)
            })
            # print(batch['input'][0])
            # print("----------------")
            # print(label)
    print(f"Rank {rank} local results: {len(local_results)}, accuracy:{correct/total}")
    
    correct_tensor = torch.tensor(correct).to(rank)
    total_tensor = torch.tensor(total).to(rank)
    
    dist.barrier()
    
    dist.reduce(correct_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_tensor, dst=0, op=dist.ReduceOp.SUM)

    with open(f"{result_save_path}/action_prediction_result_{rank}.json", 'w') as result_file:
        json.dump(local_results, result_file, indent=4)

##########################
# Dataset Class
class CustomDataset(Dataset):
    def __init__(self, full_data):
        self.data = []
        self.length = []
        for entity in tqdm(full_data):
            if len(entity['dial']) > 15000:
                continue
            new_dial = {"X": entity['dial'], "history": entity['history']}
            self.data.append(new_dial)
            self.length.append(len(new_dial['X']))
        random.shuffle(self.data)
        print(f"Avg length: {sum(self.length)/len(self.length)}")
        print(f"MAx len: {max(self.length)}")
        print("Total used ratio:", len(self.data)/len(full_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ActionDataset(Dataset):
    def __init__(self, full_data):
        self.data = []
        for entity in tqdm(full_data):
            new_dial = {"X": entity['dial'], "y": entity['label']}
            self.data.append(new_dial)
        print(f"Length of testdata: {len(self.data)}")
        # random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


##########################
# Data Loading and Distributed Training Setup
def train(rank, world_size, device="cuda"):
    setup(rank, world_size)
    
    generate_function = {
        "codellama/CodeLlama-7b-Instruct-hf": action_infer_single_code,
        "meta-llama/Meta-Llama-3-8B-Instruct":action_infer_single,
        "Qwen/Qwen2.5-Coder-7B-Instruct": action_infer_single_qwen,
    }
    
    with open("action_train.yaml") as f:
        config = yaml.load(f,Loader=yaml.SafeLoader)
        
    train_file = config['train_file']
    test_file = config['test_file']
    
    is_give_label = config['is_give_label']
    is_normal = config['is_normal']
    is_thought = config['is_thought']
    is_van = config['is_van']

    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
    model_id = config['model_id']
    model_sum = model_id.split("/")[1]
    
    if not is_van:
        if "td_llama" not in os.listdir("action_prediction_result"):
            os.mkdir("action_prediction_result/td_llama")
        if is_give_label:
            if "withgt" not in os.listdir("action_prediction_result/td_llama"):
                os.mkdir("action_prediction_result/td_llama/withgt")
            result_save_path = "action_prediction_result/td_llama/withgt"
        else:
            if "wogt" not in os.listdir("action_prediction_result/td_llama"):
                os.mkdir("action_prediction_result/td_llama/wogt")
            result_save_path = "action_prediction_result/td_llama/wogt"
    else:
        if model_sum not in os.listdir("action_prediction_result"):
            os.mkdir(f"action_prediction_result/{model_sum}")
        if is_give_label:
            if "withgt" not in os.listdir(f"action_prediction_result/{model_sum}"):
                os.mkdir(f"action_prediction_result/{model_sum}/withgt")
            result_save_path = f"action_prediction_result/{model_sum}/withgt"
        else:
            if "wogt" not in os.listdir(f"action_prediction_result/{model_sum}"):
                os.mkdir(f"action_prediction_result/{model_sum}/wogt")
            result_save_path = f"action_prediction_result/{model_sum}/wogt"
    
    special_tokens_dict = {
        "additional_special_tokens":add_tokens
    }

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens({"pad_token": "<pad>","unk_token":"<unk>"})

    tokenizer_inf = AutoTokenizer.from_pretrained(model_id,padding_side='left')
    tokenizer_inf.add_special_tokens({"pad_token": "<pad>","unk_token":"<unk>"})
    
    if not is_normal:
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        num_added_toks = tokenizer_inf.add_special_tokens(special_tokens_dict)

    # Load the dataset after process group is initialized
    train_sft_form_list, val_sft_form_list = load_data(train_file,test_file,rank,world_size,tokenizer,is_thought,is_normal,is_give_label)

    # Convert loaded data to CustomDataset
    dst_train_dataset = CustomDataset(train_sft_form_list)
    dst_metric_dataset = ActionDataset(val_sft_form_list)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config)
    model.config.use_cache = False
    replace_with_flash_attention(model) ############
    model = get_peft_model(model, lora_config)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # Move the model to the correct device and wrap it with DDP
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Optimizer
    optim = bitsandbytes.optim.adam.Adam8bit(model.parameters(), lr=0.00001, betas=(0.9, 0.995))
    scheduler = StepLR(optim, step_size=1, gamma=0.1)  # Reduce LR by 10x every epoch


    # Use DistributedSampler for both train and validation datasets
    train_sampler = DistributedSampler(dst_train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(dst_metric_dataset, num_replicas=world_size, rank=rank)

    # Create DataLoaders using DistributedSampler
    n_batch = len(devices)
    train_loader = DataLoader(dst_train_dataset, batch_size=n_batch // world_size, sampler=train_sampler, collate_fn=lambda batches: collate_function(batches, tokenizer))
    # val_loader = DataLoader(dst_metric_dataset, batch_size=n_batch // world_size, sampler=val_sampler, collate_fn=action_collate_fn)

    epochs = 1

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        print(f"Training on GPU {rank}... Epoch {epoch + 1}/{epochs}")

        # Set the epoch for the sampler to shuffle the data differently at each epoch
        train_sampler.set_epoch(epoch)
        if not is_van:
            for entity in tqdm(train_loader,dynamic_ncols=False, ascii=True):
                X = entity['input_ids'].to(rank)
                a = entity['attention_mask'].to(rank)
                history_length = entity['history']
                optim.zero_grad()
                # print("Hist:",tokenizer.decode(entity['input_ids'].tolist()[0][:entity['history'].item()]))
                # print("--------------------")
                # print("Dial:",tokenizer.decode(entity['input_ids'].tolist()[0]))
                X_masked = X.clone()
                rows = torch.arange(X.size(1)).unsqueeze(0)
                mask = rows < history_length.unsqueeze(1)
                X_masked[mask] = -100
                outputs = model(X, attention_mask=a, labels=X_masked)
                loss = outputs.loss
                # print(f"Current loss:{loss.detach().item()}")
                train_loss += loss.detach().item()
                loss.backward()
                optim.step()

            # Reduce train_loss across all GPUs
            train_loss_tensor = torch.tensor(train_loss).to(rank)
            dist.reduce(train_loss_tensor, dst=0, op=dist.ReduceOp.SUM)

            # Only rank 0 prints the average train loss
            if rank == 0:
                avg_train_loss = train_loss_tensor.item() / len(train_loader) / world_size
                print(f"Train Loss (Average) on all GPUs: {avg_train_loss}")

            scheduler.step()
        
        file_name = f"action_is_give_label_{str(is_give_label)}_normal_{str(is_normal)}_thought_{is_thought}_result_{test_file}_{model_sum}"
        if is_van:
            file_name+="_van"
        file_name+=".json"

        if rank == 0:
            model_name = f"model_epoch_action_is_givelabel_{str(is_give_label)}_normal_{str(is_normal)}"
            if is_van:
                model_name+="_van"
            if not is_van:
                model_name+=".pt"
                torch.save(model.module.state_dict(), model_name)
        infer_function = generate_function[model_id]
        validate(rank, model, dst_metric_dataset, world_size, tokenizer_inf,is_normal,is_thought,file_name,infer_function,result_save_path)
    cleanup()


##########################
def main():
    world_size = torch.cuda.device_count()  # Automatically detect the number of GPUs
    try:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        error_message = traceback.format_exc()
        with open("error_train_action.txt", 'w') as error_file:
            error_file.write(f"An error occurred during training:\n{error_message}")


if __name__ == "__main__":
    check_gpu_availibility()
    main()

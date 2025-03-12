import json,random,re,yaml,os,datetime
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
from nlp import finetune_form,dst_form_token,dst_form_normal
from prompts import add_tokens
import traceback
from torch.optim.lr_scheduler import StepLR

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
            print("Self attn")
            setattr(model, name, FlashSelfAttention(causal=True))  # Causal=True for autoregressive models
            
def dst_infer_single_mistral(input_text, model, rank, tokenizer_inf):
    split_text = "[/INST] "
    input_encoding = tokenizer_inf(input_text, return_tensors="pt", padding=True).to(rank)
    input_encoding['max_new_tokens'] = 300
    input_encoding['num_return_sequences'] = 1
    output = model.module.generate(**input_encoding)
    output = tokenizer_inf.decode(output[0])
    output = output.split(split_text)[1].replace("</s>","")
    return output

def dst_infer_single_qwen(input_text, model, rank, tokenizer_inf):
    split_text = "<|im_start|>assistant\n"
    input_encoding = tokenizer_inf(input_text, return_tensors="pt", padding=True).to(rank)
    input_encoding['max_new_tokens'] = 300
    input_encoding['num_return_sequences'] = 1
    output = model.module.generate(**input_encoding)
    output = tokenizer_inf.decode(output[0])
    output = output.split(split_text)[1].replace("<|im_end|>","")
    return output


def dst_infer_single_code(input_text, model, rank, tokenizer_inf):
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

def dst_infer_single(input_text, model, rank, tokenizer_inf):
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
    
def collate_function(batches,tokenizer):
    history_length = []
    for item in batches:
        history_length.append(len(tokenizer.encode(item['history'])))

    x_texts = [item['X'] for item in batches]
    x_encoded = tokenizer(x_texts, return_tensors='pt', padding='longest')
    return {"history":torch.tensor(history_length),"input_ids":x_encoded['input_ids'],"attention_mask":x_encoded['attention_mask']}

def dst_collate_fn(batches):
    x_texts,y = [],[]
    for item in batches:
        x_texts.append(item['X'])
        y.append(eval(item['y'].replace("- Dialogue State: ","").strip()))
    return {"input":x_texts,"output":y}

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

def extract_and_convert_dict(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        dict_str = match.group(0)  # Extract the matched string (the dictionary)
        dict_str = dict_str.replace("'", '"')
        try:
            extracted_dict = json.loads(dict_str)
            return extracted_dict
        except json.JSONDecodeError as e:
            try:
                extracted_dict=eval(dict_str)
                return extracted_dict
            except:
                return {"foo":"bar"}
                # raise ValueError
    else:
        print("No dictionary found in the text.")
        return None


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

def split_data_among_gpus(data, world_size):
    chunk_size = len(data) // world_size
    chunks = [data[i * chunk_size: (i + 1) * chunk_size] for i in range(world_size - 1)]
    chunks.append(data[(world_size - 1) * chunk_size:])  # Handle the last chunk
    return chunks

def convert_sets_to_lists(obj):
    if isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(elem) for elem in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj

##########################
# Load data only on rank 0 and broadcast to other ranks
def load_data(rank,
              world_size,
              tokenizer,
              train_file,
              test_file,
              is_give_label,
              is_normal):
    if rank == 0:
        print("Train data loading....")
        with open(train_file, 'r') as f:
            train_dialogue_list = json.load(f)
            # train_dialogue_list = random.sample(train_dialogue_list,10)
        
        print("Test data loading....")
        with open(test_file, 'r') as f:
            val_dialogue_list = json.load(f)
            # val_dialogue_list = random.sample(val_dialogue_list,10)
        
        if is_normal:
            train_dst = []
            for dial in train_dialogue_list:
                train_dst+=dst_form_normal(dial,tokenizer,is_give_label=is_give_label)
                
            val_dst = []
            for dial in val_dialogue_list:
                val_dst+=dst_form_normal(dial,tokenizer,is_give_label=is_give_label,is_val=True)
            # val_dst = random.sample(val_dst,2000)
        else:
            train_sft_form_list = [{"dialogue": finetune_form(dial['dialogue']), "action_list": dial['action_list']} for dial in tqdm(train_dialogue_list)]
            train_dst = []
            for dial in train_sft_form_list:
                train_dst+=dst_form_token(dial,tokenizer,is_give_label=is_give_label)
            
            val_sft_form_list = [{"dialogue": finetune_form(dial['dialogue']), "action_list": dial['action_list']} for dial in tqdm(val_dialogue_list)]
            val_dst = []
            for dial in val_sft_form_list:
                val_dst+=dst_form_token(dial,tokenizer,is_give_label=is_give_label)
        print(f"Length of train: {len(train_dst)}")
        print(f"Length of test: {len(val_dst)}")
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
             file_name,
             infer_function,
             result_save_path):
    model.eval()
    local_results = []

    total_samples = len(val_dataset)
    samples_per_gpu = total_samples // world_size
    remainder = total_samples % world_size

    start_index = rank * samples_per_gpu + min(rank, remainder)
    end_index = start_index + samples_per_gpu + (1 if rank < remainder else 0)

    val_subset = Subset(val_dataset, list(range(start_index, end_index)))
    val_loader = DataLoader(val_subset, batch_size=1, collate_fn=dst_collate_fn)

    correct = 0
    total = 0

    print(f"Rank {rank} processing samples {start_index} to {end_index}")

    with torch.no_grad():
        for batch in tqdm(val_loader,dynamic_ncols=False, ascii=True):
            label = batch['output'][0]
            y_hat = infer_function(batch['input'][0], model, rank, tokenizer_inf)

            predict = extract_and_convert_dict(y_hat)

            if normalize_json(predict) == normalize_json(label):
                correct += 1
            total += 1
            # print("Input----")
            # print(batch['input'][0])
            local_results.append({
                'input': batch['input'][0],
                'generated': y_hat,
                'process': convert_sets_to_lists(predict),
                'label': label,
                'correct': normalize_json(predict) == normalize_json(label)
            })

    # print(f"Rank {rank} local results: {len(local_results)}, accuracy:{correct/total}")
    
    correct_tensor = torch.tensor(correct).to(rank)
    total_tensor = torch.tensor(total).to(rank)
    
    dist.barrier()
    
    dist.reduce(correct_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_tensor, dst=0, op=dist.ReduceOp.SUM)

    with open(f"{result_save_path}/dst_result_{rank}.json", 'w') as result_file:
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
        # random.shuffle(self.data)
        print(f"Avg length: {sum(self.length)/len(self.length)}")
        print(f"MAx len: {max(self.length)}")
        print("Total used ratio:", len(self.data)/len(full_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DSTDataset(Dataset):
    def __init__(self, full_data):
        self.data = []
        for entity in tqdm(full_data):
            new_dial = {"X": entity['dial'], "y": entity['label']}
            self.data.append(new_dial)
        print(f"Length of testdata: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


##########################
# Data Loading and Distributed Training Setup
def train(rank, world_size, device="cuda"):
    setup(rank, world_size)
    generate_function = {
        "codellama/CodeLlama-7b-Instruct-hf":dst_infer_single_code,
        "meta-llama/Meta-Llama-3-8B-Instruct":dst_infer_single,
        "Qwen/Qwen2.5-Coder-7B-Instruct":dst_infer_single_qwen,
    }
    
    with open("dst_train.yaml") as f:
        config = yaml.load(f,Loader=yaml.SafeLoader)
        
    is_give_label = config['is_give_label']
    train_file,test_file = config['train_file'],config['test_file']
    is_normal = config['is_normal']
    is_van = config['is_van']
    if "ood" in train_file:
        is_ood = True
    else:
        is_ood = False
    
    model_id = config['model_id']
    model_sum = model_id.split("/")[1]
    
    ####
    if not is_van:
        if "td_llama" not in os.listdir("dst_result"):
            os.mkdir("dst_result/td_llama")
        if is_give_label:
            if "withgt" not in os.listdir("dst_result/td_llama"):
                os.mkdir("dst_result/td_llama/withgt")
            result_save_path = "dst_result/td_llama/withgt"
        else:
            if "wogt" not in os.listdir("dst_result/td_llama"):
                os.mkdir("dst_result/td_llama/wogt")
            result_save_path = "dst_result/td_llama/wogt"
    else:
        if model_sum not in os.listdir("dst_result"):
            os.mkdir(f"dst_result/{model_sum}")
        if is_give_label:
            if "withgt" not in os.listdir(f"dst_result/{model_sum}"):
                os.mkdir(f"dst_result/{model_sum}/withgt")
            result_save_path = f"dst_result/{model_sum}/withgt"
        else:
            if "wogt" not in os.listdir(f"dst_result/{model_sum}"):
                os.mkdir(f"dst_result/{model_sum}/wogt")
            result_save_path = f"dst_result/{model_sum}/wogt"

    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
    
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
    
    train_sft_form_list, val_sft_form_list = load_data(rank,world_size,tokenizer,train_file,test_file,is_give_label,is_normal)

    # Convert loaded data to CustomDataset
    dst_train_dataset = CustomDataset(train_sft_form_list)
    dst_metric_dataset = DSTDataset(val_sft_form_list)

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
    replace_with_flash_attention(model)
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

    # Create DataLoaders using DistributedSampler
    n_batch = len(devices)
    train_loader = DataLoader(dst_train_dataset, batch_size=n_batch // world_size, sampler=train_sampler, collate_fn=lambda batches: collate_function(batches, tokenizer))

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
                # print("Hist:",tokenizer.decode(entity['input_ids'].tolist()[0][:entity['history'].item()]))
                # print("--------------------")
                # print("Dial:",tokenizer.decode(entity['input_ids'].tolist()[0]))
                optim.zero_grad()
                X_masked = X.clone()
                rows = torch.arange(X.size(1)).unsqueeze(0)  # (1, seq_length)
                mask = rows < history_length.unsqueeze(1)  # (batch_size, seq_length)
                X_masked[mask] = -100  # Apply -100 to positions to be ignored in loss calculation
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
        
        file_name = f"dst_is_give_label_{str(is_give_label)}_normal_{str(is_normal)}_result_{test_file}_{model_sum}"
        
        if is_ood:
            file_name+="_ood"
        if is_van:
            file_name+="_van"
        file_name+=".json"

        if rank == 0:
            model_name = f"model_epoch_dst_is_givelabel_{str(is_give_label)}_normal_{str(is_normal)}"
            if is_ood:
                model_name+="_ood"
            if is_van:
                model_name+="_van"
            model_name+=".pt"
            if not is_van:
                print(model_name)
                torch.save(model.module.state_dict(), model_name)
        infer_function = generate_function[model_id]
        print(infer_function.__name__)
        validate(rank, model, dst_metric_dataset, world_size, tokenizer_inf,file_name,infer_function,result_save_path)
    cleanup()


##########################
def main():
    world_size = torch.cuda.device_count()  # Automatically detect the number of GPUs
    try:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        error_message = traceback.format_exc()
        with open("error_train_dst.txt", 'w') as error_file:
            error_file.write(f"An error occurred during training:\n{error_message}")


if __name__ == "__main__":
    check_gpu_availibility()
    main()

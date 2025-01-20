import subprocess

def run_commands(commands, log_flags):
    all_output = ""  # 用于保存所有命令的输出
    
    # 遍历命令列表和标志列表
    for command, log in zip(commands, log_flags):
        # 使用列表格式执行每个命令
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        
        # 输出每个命令的标准输出和标准错误
        print(f"Running command: {command}")
        print(result.stdout)
        
        # 如果命令执行过程中有错误，输出错误
        if result.stderr:
            print(result.stderr)

        # 如果标志为True，则记录命令输出
        if log:
            all_output += f"Command: {' '.join(command)}\n"
            all_output += "Standard Output:\n"
            all_output += result.stdout + "\n"
            if result.stderr:
                all_output += "Standard Error:\n"
                all_output += result.stderr + "\n"
            all_output += "-" * 40 + "\n"  # 分隔符
    
    # 将所有输出保存到一个.txt文件
    with open("output_second.txt", "w") as file:
        file.write(all_output)
    print("All outputs have been saved to output_second.txt.")

# 示例命令和对应的记录标志
base_dir = "/home/xly/OPERA"
local_path_to_utils_file = f"{base_dir}/Youlong/"
path_to_utils_file = f"{base_dir}/transformers-4.29.2/src/transformers/generation/"
path_to_opera = f"{base_dir}/"
path_to_coco_val = f"{base_dir}/data/val2014/"
path_to_coco_annotations  = f"{base_dir}/data/annotations"
path_to_generated_captions = f"{base_dir}/log/llava-1.5/"
# file_name_of_generated_captions = "ours-second.jsonl" # 这个是默认的
save_path = f"{base_dir}/ourresults/"

def Encapsulation(file_name: str = 'utils_1.py', comment: str = '') -> list:
    command_1 = f"cp {local_path_to_utils_file + file_name} {path_to_utils_file + 'utils.py'}"
    # 记住在这里要把gpu-id也改一下（如果需要的话）
    command_2 = f"python chair_eval.py --model llava-1.5  --data_path {path_to_coco_val} --beam 5 --scale_factor 50 --threshold 15 --num_attn_candidates 5 --penalty_weights 1 \
                --gpu-id 1 \
                --output_path new_{file_name}.jsonl"
    # command_3 = f"python chair.py   --cap_file {path_to_generated_captions + file_name_of_generated_captions}  --image_id_key image_id --caption_key caption --coco_path {path_to_coco_annotations}  --save_path {save_path + str(idx)}.jsonl"
    # command_4 = f"mv {path_to_generated_captions + file_name_of_generated_captions} {path_to_generated_captions + str(idx)}.jsonl"
    command_5 = f"rm -f {path_to_utils_file + 'utils.py'}"
    return [command_1, command_2, command_5]

commands = [
    # *Encapsulation('utils_1.py','d0 = 5'),
    # *Encapsulation('utils_2.py','alpha = 2.0'),
    # *Encapsulation('utils_3.py','c = log 0.2'),
    # *Encapsulation('utils_4.py','reward = log 30'),
    # *Encapsulation('utils_5.py','reward = log 7'),
    # *Encapsulation('utils_6.py','with candidate rewards'),
    # *Encapsulation('utils_7.py','c = log 0.05'),
    # *Encapsulation('utils_8.py'),
    # *Encapsulation('utils_9.py'),
    # *Encapsulation('utils_10.py'),
    # *Encapsulation('utils_11.py'),
    *Encapsulation('utils_12.py'),
    *Encapsulation('utils_13.py'),
    *Encapsulation('utils_14.py'),
]
print(commands)
log_flags = ["python chair.py" in x for x in commands] 
print(log_flags)
run_commands(commands, log_flags)

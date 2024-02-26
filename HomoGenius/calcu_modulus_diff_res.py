# 将calculate_resized_elastic_matrix_FNOres128_18_24_diff_res.py重新运行3次，将train_res=128分别改成64和32。

import subprocess

def modify_and_run(file_path, original, replacements):
    # 读取原始文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()

    for replacement in replacements:
        # 创建修改后的文件内容
        modified_content = [line.replace(original, f"train_res = {replacement}") if "train_res = " in line else line for line in content]

        # 写入到临时文件
        temp_file_path = 'temp_script.py'
        with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
            temp_file.writelines(modified_content)

        # 运行修改后的文件
        print(f"Running script with train_res={replacement}")
        subprocess.run(['python', temp_file_path])

# 使用示例
file_path = 'calculate_resized_elastic_matrix_FNOres128_18_24_diff_res.py'  # 替换为您的 Python 文件路径
modify_and_run(file_path, "train_res = 128", [128, 64, 32])

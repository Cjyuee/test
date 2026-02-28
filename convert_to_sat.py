# 处理MathV360K为和SAT数据集一样的格式用于训练

import json
import os

input_file = "/root/autodl-tmp/datasets/MathV360K/train_samples_all_tuning.json"
output_file = "/root/autodl-tmp/datasets/MathV360K/mathv360k_train.json"



IMAGE_PREFIX = "data_images"  # 要添加的前缀路径

def convert_sample(sample):
    messages = []

    for conv in sample.get("conversations", []):
        if conv.get("from") == "human":
            messages.append({
                "role": "user",
                "content": conv.get("value", "").strip()
            })

        elif conv.get("from") == "gpt":
            messages.append({
                "role": "assistant",
                "content": conv.get("value", "").strip()
            })

    # 拼接新路径
    original_image_path = sample.get("image", "")
    new_image_path = os.path.join(IMAGE_PREFIX, original_image_path)

    new_sample = {
        "messages": messages,
        "images": [new_image_path]
    }

    return new_sample


# ===== 读取 JSON 数组 =====
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# ===== 转换 =====
converted_data = [convert_sample(sample) for sample in data]

# ===== 保存 =====
with open(output_file, "w", encoding="utf-8") as f_out:
    json.dump(converted_data, f_out, ensure_ascii=False, indent=4)

print("转换完成，输出文件：", output_file)
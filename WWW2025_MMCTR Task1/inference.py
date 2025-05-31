import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd

# 指定本地模型路径
model_path = "/root/Qwen2.5-VL-7B-Instruct"  # 修改为您的本地模型路径

# 加载本地模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)

# 如果您想启用flash_attention_2以获得更好的加速和内存节省
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# 加载本地处理器
processor = AutoProcessor.from_pretrained(model_path)


def get_embeddings(messages, pooling_method="mean"):
    """
    获取多模态输入的嵌入向量
    
    参数:
    messages - 包含文本和图像的消息列表
    pooling_method - 池化方法，可选 'mean'(平均池化)、'last'(最后一个token)或'cls'(第一个token)
    
    返回:
    嵌入向量
    """
    # 准备输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 处理本地图像路径
    for i, img in enumerate(image_inputs):
        if isinstance(img, str) and os.path.exists(img):
            from PIL import Image
            image_inputs[i] = Image.open(img).convert('RGB')
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # 使用模型获取嵌入向量(不生成文本)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # print(outputs)
        # 获取倒数第二层的隐藏状态
        last_hidden_states = outputs.hidden_states[-2]
        if pooling_method == "last":
            # 使用最后一个token的隐藏状态作为整个序列的表示
            embedding = last_hidden_states[0, -1, :]
            # 进行L2归一化
            embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1).squeeze(0).to(torch.float32).cpu().numpy()
        elif pooling_method == "cls":
            # 使用第一个token(CLS token)的隐藏状态
            embedding = last_hidden_states[0, 0, :]
            # 进行L2归一化
            embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1).squeeze(0).to(torch.float32).cpu().numpy()
        else:  # 默认使用mean
            # 使用所有token的平均隐藏状态
            embedding = torch.mean(last_hidden_states, dim=1).squeeze()
            # 进行L2归一化
            embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1).squeeze(0).to(torch.float32).cpu().numpy()
    
    return embedding

def process_items_and_generate_embeddings(parquet_file, images_folder, output_file, batch_size=100, save_per=1000):
    """
    处理parquet文件，为每个item生成embedding并保存结果
    
    参数:
    - parquet_file: 输入的parquet文件路径
    - images_folder: 存储图片的文件夹路径
    - output_file: 输出的parquet文件路径
    - batch_size: 处理的批次大小，用于中间保存
    """
    
    # 读取parquet文件
    print(f"正在读取parquet文件: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    
    # 检查列名并确保数据格式正确
    # 假设第一列是item_id, 第二列是文本内容
    cols = df.columns.tolist()
    id_col = cols[0]
    text_col = cols[1]
    
    # 创建新列用于存储embeddings
    df['embedding'] = None
    
    # 检查是否存在中间结果文件，如果存在则加载
    temp_file = f"{output_file}.temp.parquet"
    if os.path.exists(temp_file):
        print(f"发现中间结果文件，加载已处理的数据...")
        temp_df = pd.read_parquet(temp_file)
        
        # 获取已处理的item_ids
        processed_ids = set(temp_df[id_col].tolist())
        print(f"已处理的item数量: {len(processed_ids)}")
        
        # 更新主DataFrame中已处理的embeddings
        df = pd.concat([df[~df[id_col].isin(processed_ids)], temp_df])
    else:
        processed_ids = set()
    
    # 获取待处理的items
    items_to_process = df[~df[id_col].isin(processed_ids)].copy()
    
    if len(items_to_process) == 0:
        print("所有数据已处理完毕，无需再次处理")
        df.to_parquet(output_file)
        return
    
    print(f"待处理的item数量: {len(items_to_process)}")
    
    # 初始化进度条
    pbar = tqdm(total=len(items_to_process))
    
    # 批次处理
    batch_count = 0
    for idx, row in items_to_process.iterrows():
        item_id = row[id_col]
        text_content = row[text_col]
        image_path = os.path.join(images_folder, f"{item_id}.jpg")
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"警告: 图片不存在 - {image_path}")
            df.loc[idx, 'embedding'] = np.zeros(1024).tolist()  # 使用零向量作为占位符，转换为Python列表
            continue
        
        try:

            # 构建消息
            messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an AI expert in multimodal feature extraction. Identify and represent the core attributes and concepts conveyed by both the image and its associated text."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path
                    },
                    {
                        "type": "text",
                        "text": f"""Content Title/Keywords: {text_content}
                        Considering both the image and the title/keywords, what are the most salient visual features (objects, scenes, colors, style) and textual concepts? 
                        How do they relate to form a cohesive understanding of the content?"""
                    }
                ]
            }
        ]
            
            # 获取嵌入向量
            embedding = get_embeddings(messages)
            
            # 将numpy数组转换为Python列表
            df.at[idx, 'embedding'] = embedding.tolist()
            
            # 更新已处理集合
            processed_ids.add(item_id)
            
            # 更新进度条
            pbar.update(batch_size)
            
            # 批次计数
            batch_count += 1
            
            # 每处理batch_size个item保存一次中间结果
            if batch_count % save_per == 0:
                # 保存中间结果
                temp_save_df = df[df[id_col].isin(processed_ids)].copy()
                temp_save_df.to_parquet(temp_file)
                print(f"\n已保存中间结果，当前处理了 {len(processed_ids)} 个items")
                
        except Exception as e:
            print(f"\n处理item {item_id}时出错: {str(e)}")
            # 继续处理下一个item
            continue
    
    # 关闭进度条
    pbar.close()
    
    # 保存最终结果
    print("处理完毕，保存最终结果...")
    
    # 将列表格式的embedding转换回numpy数组用于验证
    sample_idx = df.index[0]
    sample_embedding_list = df.at[sample_idx, 'embedding']
    sample_embedding = np.array(sample_embedding_list)
    print(f"样本embedding维度: {sample_embedding.shape}")
    
    # 保存结果
    df.to_parquet(output_file)
    
    # 删除临时文件
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    # 参数
    parquet_file = "/root/item_feature.parquet"  # 修改为您的输入文件路径
    images_folder = "/root/item_images"  # 修改为您的图片文件夹路径
    output_file = "/root/qwen_7B_vl_mean_L2_2_48.parquet"  # 输出文件路径
            
        
    process_items_and_generate_embeddings(
        parquet_file=parquet_file,
        images_folder=images_folder,
        output_file=output_file,
        batch_size=2
    )
    
    # 测试读取结果
    test_df = pd.read_parquet(output_file)
    sample_embedding_list = test_df.iloc[0]['embedding']
    sample_embedding = np.array(sample_embedding_list)
    print(f"最终结果中的embedding维度: {sample_embedding.shape}")
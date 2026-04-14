import sys  # 添加 sys 模块以便退出程序
sys.path.append("/home/NCUT/teacher/xmy/llava-yl")

import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
import json
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"     # 使用的GPU位置

# from .PAI_attention import llama_modify
# from .CFG import CFGLogits

import shutil
from io import BytesIO
import plotly.io as pio
import plotly.express as px
from transformers import TextStreamer
from collections import defaultdict
from transformers.generation.logits_process import LogitsProcessorList

# flask
from flask import Flask,request,jsonify
from types import SimpleNamespace
import base64
from io import BytesIO

# 全局变量
tokenizer = None
model = None
image_processor = None
context_len = None

model_path = "/home/NCUT/teacher/xmy/llava-yl/llava-v1.5-7b"   # 模型地址
model_name = get_model_name_from_path(model_path)              # 模型名字

max_new_tokens = 100   # 最大的token数字

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_llava_model_tokenizer(half=True):
    global tokenizer,model,image_processor,context_len
    if tokenizer == None or model == None:    # 判定是否是None
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name = model_name,         # 通常根据 model_path 自动生成
            load_8bit=False,
            load_4bit=False,
            device="cuda"
        )
    if(half == True):
        model.to(dtype=torch.float16)              # 半进度模式
    model.eval()                  # 评估模式

def answer(image,question):
    if image == None or question == None:
        return None
    # 如果图片 或者 问题 是None的话，那么就返回None
    
    global tokenizer,model,image_processor,context_len    # 全局变量的使用
    global max_new_tokens      # 最大token

    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    conv = conv_templates["llava_v0"].copy()

    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    # question = "what is it in the picture?"

    # 构造 multimodal prompt（注意这里加了 <image>）
    if image is not None:
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        else:
            inp = DEFAULT_IMAGE_TOKEN + "\n" + question
        # image = None

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output,image_position = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            output_hidden_states=True,
            output_attentions=True,
            return_dict_in_generate=True,
            max_new_tokens=max_new_tokens,
            use_cache=True)

    output_ids = output[0][0] 
    outputs = tokenizer.decode(output_ids).strip()
    return outputs

# 获取llava模型中的回答
def extract_response(output_text: str) -> str:
    """
    从 LLaVA / LLaMA 类模型的输出中提取 <s> ... </s> 之间的内容。
    如果 </s> 缺失，也能处理。
    """
    if not output_text:
        return ""

    # 查找 <s> 的位置（一般在最前面）
    start = output_text.find("<s>")
    if start == -1:
        start = 0
    else:
        start += 3  # 跳过 <s>

    # 查找 </s> 的位置（可能不存在）
    end = output_text.find("</s>")
    if end == -1:
        end = len(output_text)

    return output_text[start:end].strip()


app = Flask(__name__)

@app.route("/llava/load_model",methods = ["POST"])
def init_load_model():
    data = request.get_json()
    half_get = data.get("half")             # 是否半精度
    load_llava_model_tokenizer(half_get)    # 载入llava的模型
    return jsonify({"status": "模型加载完成"})  # 返回状态


@app.route("/llava/answer",methods = ["POST"])
def model_answer():
    # 返回回答
    global tokenizer,model,image_processor,context_len    # 全局变量的使用
    if tokenizer == None or model == None:
        return jsonify({"error": "模型尚未加载"}), 400
    
    # 获得数据
    data = request.get_json()
    image_path = data.get("image_path")     # 图片地址
    question = data.get("question")         # 问题

    # 数据获取判断
    if image_path is None or question is None:
        return jsonify({"error": "缺少参数"}), 400
    
    image = load_image(image_path)   # 载入图片

    raw_answer = answer(image,question)

    if raw_answer == None:
        # 如果没有生成回答
        return jsonify({"error": "生成回答错误"}), 400

    answer_final = extract_response(raw_answer)  # 提取

    return jsonify({"answer": answer_final})    # 返回模型回答


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5008 , debug=True)   # 支持修改bug，保存文件之后重启
    # 运行之后，直接触发窗口信号


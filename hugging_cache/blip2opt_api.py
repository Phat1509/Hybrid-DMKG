# blip2opt model
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

# flask
from flask import Flask,request,jsonify
from types import SimpleNamespace
import base64
from io import BytesIO
import os
import argparse

# 全局变量
processor = None
model = None

# device default
device_id = 0
device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

# # 改变设备
# def Change_Device(device_id_):
#     global device_id,device
#     device_id = device_id_
#     device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
#     if(model is not None):
#         model.to(device)
#         # 如果model不是None的话，就放置到device中

# 提取回答中的answer之后的部分
def extract_answer(text: str, keyword: str = "Answer:") -> str:
    if keyword in text:
        return text.split(keyword, 1)[1].strip()
    else:
        return text.strip()

# 用于问题的处理
def question_process(question):
    question = "Question:" + question + "? Answer:"
    return question         
    # 因为blip2opt模型对输入的格式有要求，因此需要对输入的prompt进行处理


# 向模型提问并且获取回答
def answer(image,question):
    # 预处理输入
    inputs = processor(images=image, text=question, return_tensors="pt")
    # 设置 tensor 到设备，并保持类型
    inputs["pixel_values"] = inputs["pixel_values"].to(model.device, dtype=torch.float16)
    inputs["input_ids"] = inputs["input_ids"].to(model.device)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].to(model.device)

    # 生成回答
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        num_beams=5,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=20,
        top_p=0.8,
        temperature=0.2,
    )

    # 只解码新生成的部分
    new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    text_decode = processor.decode(new_tokens[0], skip_special_tokens=True)

    # 回答提取
    answer_text = extract_answer(text_decode)
    return answer_text
    
app = Flask(__name__)

# 加载模型
@app.route("/blip2opt/load_model",methods = ["POST"])
def init_load_model():
    data = request.get_json()
    half_get = data.get("half")   # 是否半精度
    half_get = True
    global processor,model
    if processor is None or model is None:
        # 判断，避免重复载入模型
        processor = Blip2Processor.from_pretrained("/home/NCUT/teacher/xmy/MRAG/PMI/hugging_cache/BLIP2-OPT")
        # processor = Blip2Processor.from_pretrained("/home/lyuan/MQA/PMI/hugging_cache/BLIP2-OPT")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "/home/NCUT/teacher/xmy/MRAG/PMI/hugging_cache/BLIP2-OPT",
            torch_dtype=torch.float16,
            device_map={"": device}
        )
        if(half_get is True):
            model.half()       # 半精度调整
        model.eval()           # 推理模型
    return jsonify({"status": "模型加载完成"})  # 返回状态



# blip2opt模型 回答接口
@app.route("/blip2opt/answer",methods = ["POST"])
def model_answer():
    global processor, model
    # 判断全局变量是否已经加载完成
    if processor is None or model is None:
        return jsonify({"error": "模型尚未加载"}), 400

    # 获得数据
    data = request.get_json()
    image_path = data.get("image_path")
    question = data.get("question")

    # 数据获取判断
    if image_path is None or question is None:
        return jsonify({"error": "缺少参数"}), 400

    image = Image.open(image_path)     # 打开图片数据
    question = question_process(question)   # 进行处理

    answer_final = answer(image,question) # 获得模型回答
    return jsonify({"answer": answer_final})    # 返回模型回答

# # 还未测试
# # 更换模型所在的gpu
# @app.rout("/blip2opt/device",mothods = ["POST"])
# def device_change():
#     data = request.get_json()
#     device_id_ = data.get("device_id")
#     Change_Device(device_id_)
#     return jsonify({"device_change":f"模型已经转移到GPU{device_id_}"})  # 返回修改状态



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=None)
    args = parser.parse_args()
    # 优先命令行参数，其次环境变量，最后默认5005
    port = args.port or int(os.environ.get("PORT", 5006))
    app.run(host='0.0.0.0', port=port , debug=True)   # 支持修改bug，保存文件之后重启
    # 运行之后，直接触发窗口信号



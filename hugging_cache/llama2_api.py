# 使用环境：llama2_7b

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# flask
from flask import Flask,request,jsonify
from types import SimpleNamespace
import base64
from io import BytesIO


# 模型路径：改成你本地的路径
model_path = "/home/NCUT/teacher/xmy/MRAG/PMI/hugging_cache/Llama-2-7b-chat-hf"

# 全局变量
tokenizer = None
model = None

# device default
device_id = 3
device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

# # 改变设备
# def Change_Device(device_id_):
#     global device_id,device
#     device_id = device_id_
#     device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
#     if(model is not None):
#         model.to(device)
#         # 如果model不是None的话，就放置到device中

# 载入模型
def load_llama_model_tokenizer(half=True):
    global tokenizer, model
    half =False
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if half:
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)
    model.eval()
    model.to(device)
    return tokenizer, model       # 返回tokenizer和model

# 询问llama模型
def ask_llama(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]

    # 生成文本
    print("模型输入：", prompt)  # 打印模型输入
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,  # 缩短最大生成长度
        do_sample=True,
        top_k=20,
        top_p=0.8,
        temperature=0.2,
        eos_token_id=tokenizer.eos_token_id,  # 添加终止符
    )


    #只解码新生成的部分
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    # 可选：遇到终止符提前截断
    if tokenizer.eos_token in response:
        response = response.split(tokenizer.eos_token)[0]
    print("模型回答：", response)  # 打印模型回答
    print("----------------------")  # 分隔线

    return response

####################################
# 没有找到可以用的prompt，这里先设置，以后可以修改或者换chat版本的llama2模型
# prompt 设计
prompt = ""

# 从模型回答中提取回答
before_answer = ""  # 在模型回答之前的关键字，下面的代码是用于提取后面的内容
def extract_answer(text: str, keyword: str = before_answer) -> str:
    if keyword in text:
        return text.split(keyword, 1)[1].strip()
    else:
        return text.strip()
######################################

# 载入模型
app = Flask(__name__)

# 初始化模型，flask接口
@app.route("/llama2/load_model",methods = ["POST"])
def init_load_model():    
    data = request.get_json()
    half_get = data.get("half")   # 是否半精度
    global tokenizer,model
    if tokenizer is None or model is None:
        load_llama_model_tokenizer(half_get)   # 载入模型
    return jsonify({"status": "模型加载完成"})  # 返回状态

# 模型回答，flask接口
@app.route("/llama2/answer",methods = ["POST"])
def model_answer():
    global tokenizer, model
    # 判断全局变量是否已经加载完成
    if tokenizer is None or model is None:
        return jsonify({"error": "模型尚未加载"}), 400

    # 获得数据
    data = request.get_json()
    question = data.get("question")

    generated_text = ask_llama(question)

    return jsonify({"answer":generated_text})  # 返回回答

# # 还未测试
# # 更换模型所在的gpu
# @app.rout("/llama2/device",mothods = ["POST"])
# def device_change():
#     data = request.get_json()
#     device_id_ = data.get("device_id")
#     Change_Device(device_id_)
#     return jsonify({"device_change":f"模型已经转移到GPU{device_id_}"})  # 返回修改状态


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007 , debug=True)   # 支持修改bug，保存文件之后重启
    # 运行之后，直接触发窗口信号
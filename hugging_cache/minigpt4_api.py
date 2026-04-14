import sys
# minigpt4加载使用到的包
# sys.path.append('/home/NCUT/teacher/xmy/MRAG/PMI/hugging_cache/MiniGPT-4')
sys.path.append('/home/lyuan/MQA/PMI/hugging_cache/MiniGPT-4')


from flask import Flask,request,jsonify
from types import SimpleNamespace
import os
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub
from transformers import StoppingCriteriaList
import torch
from PIL import Image
import base64
from io import BytesIO
import argparse





# 两个全局变量
minigpt4_model = None          # 先定义一个模型（空），当实现的
conv_template = None                  # 也定义一个空的prompt

# device default
device_id = 0
device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

# 改变设备
# def Change_Device(device_id_):
#     global device_id,device
#     device_id = device_id_
#     device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
#     if(model is not None):
#         model.to(device)
#         # 如果model不是None的话，就放置到device中

# 载入minigpt4模型
def load_minigpt4_model(cfg_path='/home/lyuan/MQA/PMI/hugging_cache/MiniGPT-4/eval_configs/minigpt4_eval.yaml',half = True):

# def load_minigpt4_model(cfg_path='/home/NCUT/teacher/xmy/MRAG/PMI/hugging_cache/MiniGPT-4/eval_configs/minigpt4_eval.yaml',half = True):
    args = SimpleNamespace(
        cfg_path=cfg_path,
        options=[],
        gpu_id=device_id
    )
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    
    if(half is True):
        # 转为半精度
        model = model.half()

    # 设置为推理模式
    model.eval()

    conv_dict = {
        'pretrain_vicuna0': CONV_VISION_Vicuna0,
        'pretrain_llama2': CONV_VISION_LLama2
    }
    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(model.device) for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, device=model.device, stopping_criteria=stopping_criteria)

    return chat, CONV_VISION

# 使用minigpt4模型进行回答
def minigpt4_answer(chat, CONV_VISION, image: Image.Image, question: str) -> str:
    chat_state = CONV_VISION.copy()
    img_list = []
    chat.upload_img(image, chat_state, img_list)
    chat.encode_img(img_list)
    chat.ask(question, chat_state)

    try:
        answer = chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=1,
            temperature=1.0,
            max_new_tokens=10,
        )[0]
        # print(f"Answer: {answer        gunicorn -w 4 -b 0.0.0.0:5005 minigpt4_api:app}")
    except Exception as e:
        print(f"Error during model inference: {e}")
        answer = "unkonw"

    return answer



app = Flask(__name__)

# 使用Flask运行，并与调用本模块的函数进行通信
@app.route("/minigpt4/load_model",methods = ["POST"])
def init_load_model():
    # 已经定义好了模型的路径和使用的GPU
    data = request.get_json()
    half_get = data.get("half")   # 是否半精度
    half_get =True
    global minigpt4_model,conv_template

    if minigpt4_model is None or conv_template is None:
        # 避免重复载入模型
        minigpt4_model,conv_template = load_minigpt4_model(half=half_get) # 载入模型
    return jsonify({"status": "模型加载完成"})

# minigpt4模型 回答接口
@app.route("/minigpt4/answer",methods = ["POST"])
def model_answer():
    # 直接调用模型
    
    global minigpt4_model, conv_template
    if minigpt4_model is None or conv_template is None:
        return jsonify({"error": "模型尚未加载"}), 400

    data = request.get_json()
    image_path = data.get("image_path")    # 图片地址
    question = data.get("question")        # 问题

    # 数据获取判断
    if image_path is None or question is None:
        return jsonify({"error": "缺少参数"}), 400

    image = Image.open(image_path)         # 打开图片

    answer = minigpt4_answer(minigpt4_model,conv_template,image,question)
    return jsonify({"answer": answer})    # 返回模型回答

# # 还未测试
# # 更换模型所在的gpu
# @app.rout("/minigpt4/device",mothods = ["POST"])
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
    port = args.port or int(os.environ.get("PORT", 5005))
    app.run(host='0.0.0.0', port=port, debug=False)
    # 运行之后，直接触发窗口信号




# from PIL import Image

# # 模型初始化（只需一次）
# # chat, conv_template = load_minigpt4_model()

# # 加载图片与提问
# image = Image.open('/home/NCUT/teacher/xmy/MRAG/PMI/test/url2.jpg')
# question = "describe this picture"

# # 获取回答
# answer = minigpt4_answer(chat, conv_template, image, question)
# print("回答:", answer)




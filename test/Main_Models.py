from PIL import Image
import requests
import torch

# Flask接口
import requests
import base64
from PIL import Image
from io import BytesIO

# vucina7b_模型的调用，因为只需要Transformers的适配即可，所以不使用flask通信，直接载入即可
import sys
# sys.path.append("/home/lyuan/MQA/PMI/hugging_cache")
sys.path.append("/home/NCUT/teacher/xmy/MRAG/PMI/hugging_cache")
# import vicuna7b_api     # vicuna-7b-v1.5

device = "cuda:4" if torch.cuda.is_available() else "cpu"

class ModelManager:
    def __init__(self,model_name,half = False,port =None):
        # 默认为半精度模式
        self.models = {"BLIP2-OPT","MiniGPT-4","LLaVA-1.5","Qwen-VL","VICUNA7B","LLAMA2"}
        self.name = model_name     # 用于分类和调用不同的模型接口
        if port is None:
            port = 5005
        self.port = port
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 未在管理器中注册。请先注册模型。")
        else:
            if model_name == "BLIP2-OPT":
                # self.processor, self.init_model = self._load_model_BLIP2_OPT()
                self._load_model_BLIP2_OPT(half=half)    # 在blip2opt_api server上载入模型
            elif model_name == "MiniGPT-4":
                #self.init_model = self._load_model_MiniGPT_4()
                self._load_model_MiniGPT_4(half=half,port=port)    # 在minigpt4_api server上载入模型
            elif model_name == "LLaVA-1.5":
                self._load_model_LLaVA_1_5(half=half,port=port)
            elif model_name == "Qwen-VL":
                self.init_model = self._load_model_Qwen_VL()

            # 下面是文本模型的载入和调用
            elif model_name == "VICUNA7B":   
                self.model,self.tokenizer = vicuna7b_api.Load_model_tokenizer()  # 载入VICUNA7B模型，本地载入模型
            elif model_name == "LLAMA2":
                self._load_model_LLAMA(half=half)        # 在llama2_api server上载入模型



        # 评估模型代码，之后再添加对应的代码
        # if half:
        #     self.init_model = self.init_model.half()
        # eval(self.init_model)


    # blip2opt_api 加载模型
    def _load_model_BLIP2_OPT(self,half=True,port=5005):
        payload = {"half":half}  # 打包half状态
        response = requests.post("http://localhost:{0}/blip2opt/load_model".format(self.port) , json=payload) 
        if response.status_code == 200:
            print("blip2opt 模型加载成功")
        else:
            raise RuntimeError("blip2opt 模型加载失败: " + response.text)

    # minigpt4_api 加载模型
    def _load_model_MiniGPT_4(self,half=True,port=5005):
        payload = {"half":half}  # 打包half状态
        response = requests.post("http://localhost:{0}/minigpt4/load_model".format(self.port) , json=payload) 
        # 直接远程调用
        if response.status_code == 200:
            print("MiniGPT-4 模型加载成功")
        else:
            raise RuntimeError("MiniGPT-4 模型加载失败: " + response.text)


    def _load_model_LLaVA_1_5(self,half=True,port=5008):
        payload = {"half":half}  # 打包half状态
        response = requests.post("http://localhost:{0}/llava/load_model".format(self.port) , json=payload) 
        # 直接远程调用
        if response.status_code == 200:
            print("llava-v1.5-7b 模型加载成功")
        else:
            raise RuntimeError("llava-v1.5-7b 模型加载失败: " + response.text)


    def _load_model_Qwen_VL(self):
        # 这里加载你的模型D
        # model = torch.load('path_to_model_d.pt', map_location=device)
        model = "模型D实例"
        return model
    
    def _load_model_LLaVA(self):
        # LLaVA的模型载入
        model = "LLaVA模型"
        return model

    def _load_model_LLAMA(self,half = True):
        payload = {"half":half}  # 打包half状态
        response = requests.post("http://localhost:5005/llama2/load_model".format(self.port), json=payload) 
        if response.status_code == 200:
            print("LLAMA 模型加载成功")
        else:
            raise RuntimeError("blip2opt 模型加载失败: " + response.text)


    # 询问 minigpt4
    def ask_minigpt4(self,question,image_path):

        # 直接传给 minigpt4_api 图片地址和问题
        payload = {
            "image_path":image_path,
            "question":question
        } # 打包图片数据与问题
        link_ip = "http://localhost:{0}/minigpt4/answer".format(self.port)  # 端口号
        response = requests.post(link_ip,json = payload)
        if response.status_code == 200:
            return response.json()["answer"]  # 成功返回这个
        else:
            raise RuntimeError("error answer"+response.text)
        
    # 询问 blip2opt
    def ask_blip2opt(self,question,image_path):
        payload = {
            "image_path":image_path,
            "question":question
        } # 打包图片数据与问题
        response = requests.post("http://localhost:{0}/blip2opt/answer".format(self.port),json = payload)
        if response.status_code == 200:
            return response.json()["answer"]  # 成功返回这个
        else:
            raise RuntimeError("error answer"+response.text)

    # 询问 llama2
    def ask_llama2(self,question):
        payload = {
            "question":question
        }
        response = requests.post("http://localhost:{0}/llama2/answer".format(self.port),json = payload)
        if response.status_code == 200:
            return response.json()["answer"]  # 成功返回这个
        else:
            raise RuntimeError("error answer"+response.text)
    def ask_llava_v1_5_7b(self,question,image_path):
        payload = {
            "image_path":image_path,
            "question":question
        } # 打包图片数据与问题
        response = requests.post("http://localhost:{0}/llava/answer".format(self.port),json = payload)
        if response.status_code == 200:
            return response.json()["answer"]  # 成功返回这个
        else:
            raise RuntimeError("error answer"+response.text)


    # def get_out_put(self, prompt, image=None):
    #     inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(device=device, dtype=torch.bfloat16)
    #     generated_ids =self.init_model.generate(**inputs)
    #     generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    #     return generated_text
    
    def get_out_put(self,prompt,image_path=None):
        # print(f"ModelManager: {self.name} get_out_put")  # 打印当前模型名称
        # print(f"Prompt: {prompt}")  # 打印输入的提示文本

        # print(image_path)
        # 使用并且用于生成文本
        generated_text = None      # 回答文本
        if self.name == "MiniGPT-4":
            if image_path is None:   # 判断是否有图片地址
                RuntimeError("No image_path")
            generated_text = self.ask_minigpt4(question=prompt,image_path=image_path)
        
        if self.name == "BLIP2-OPT":
            if image_path is None:
                RuntimeError("No image_path")
            generated_text = self.ask_blip2opt(question=prompt,image_path=image_path)

        if self.name == "VICUNA7B":
            generated_text = vicuna7b_api.ask_vicuna(prompt,self.tokenizer,self.model)   # 生成回答
            return generated_text     # 返回回答

        if self.name == "LLAMA2":
            generated_text = self.ask_llama2(prompt)
        # print(f"Generated Text: {generated_text}")  # 打印生成的文本
        if self.name == "LLaVA-1.5":
            if image_path is None:
                RuntimeError("No image_path")
            generated_text = self.ask_llava_v1_5_7b(prompt,image_path=image_path)
        return generated_text
    





# image_path = ['/home/NCUT/teacher/xmy/MRAG/PMI/test/url1.jpg',
# '/home/NCUT/teacher/xmy/MRAG/PMI/test/url2.jpg',
# '/home/NCUT/teacher/xmy/MRAG/PMI/test/url3.jpg',
# '/home/NCUT/teacher/xmy/MRAG/PMI/test/url4.jpg'
# ]

# # test for minigpt4 #
# # minigpt4_model = ModelManager("MiniGPT-4")    # 载入模型

# # # # 询问minigpt4模型
# # # 四张图片，输入给minigpt4模型
# # for image_item in image_path:
# #     ans = minigpt4_model.ask_minigpt4("decribe the picture",image_item)
# #     print(ans)


# # test for blip2opt #
# # blip2opt_model = ModelManager("BLIP2-OPT")    # 载入模型

# # # 询问blip2opt模型
# # # 四张图片，输入给blip2opt模型
# # for image_item in image_path:
# #     ans = blip2opt_model.ask_blip2opt("decribe the picture",image_item)
# #     print(ans)




# questions = ["write a poem of Spring.",
# "where is China?"]

# # test for llama2 #
# # llama2_model = ModelManager("LLAMA2")
# # for ques_item in questions:
# #     ans = llama2_model.ask_llama2(ques_item)
# #     print(ans)


# # test for vicuna #
# # vicuna_model = ModelManager("VICUNA7B")
# # for ques_item in questions:
# #     ans = vicuna_model.get_out_put(ques_item)
# #     print(ans)
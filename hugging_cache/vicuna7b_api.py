# 使用vicuna7b-v1.5版本

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# 全局变量
# tokenizer = None
# model = None


# device default
device_id = 0 # 你想用的GPU号码
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

def Load_model_tokenizer(half = True):
    # global tokenizer,model  # 全局变量
    tokenizer = AutoTokenizer.from_pretrained("/home/NCUT/teacher/xmy/MRAG/PMI/hugging_cache/vicuna-7b-v1.5")
    model = AutoModelForCausalLM.from_pretrained("/home/NCUT/teacher/xmy/MRAG/PMI/hugging_cache/vicuna-7b-v1.5")
    if(half is True):
        model.half()  # 半精度模式
    model.eval()      # 评估模式 
    
    return model,tokenizer

# 用于引导模型，这里先不用
template_guide_model = (
    "A chat between a curious user and an artificial intelligence assistant.\n"
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
)

# 提取回答中的 ASSISTANT: 之后的部分
def extract_answer(text: str, keyword: str = "ASSISTANT:") -> str:
    if keyword in text:
        return text.split(keyword, 1)[1].strip()
    else:
        return text.strip()

# 使用文本模型
def ask_vicuna(user_prompt,tokenizer,model):
    template_QA = (
        f"USER: {user_prompt}\n"
        "ASSISTANT:"
    ) # 一个QA模板
    
    model.to(device)          # 放置到想要的GPU上

    #template_QA = template_guide_model + template_QA  # 完整的模板，这里先不用

    inputs = tokenizer(template_QA, return_tensors="pt").to(device)
    if(model is None or tokenizer is None):
        raise TypeError("Model and tokenizer must be loaded before calling ask_vicuna()")

    # 下面参数是可以调整的
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    # 带有问题和回答的raw answer
    raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)  
    
    # 只提取回答的real answer
    real_answer = extract_answer(raw_answer)                             

    return real_answer


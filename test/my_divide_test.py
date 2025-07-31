import openai
openai.api_key = "sk-oHf1AIjc3m5jl593F9Fc3d23Dd314031B83d1c22A5Bc429b"
openai.base_url = "https://api.gpt.ge/v1/"
openai.default_headers = {"x-foo": "true"}
def run_llm_divide(query):

    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":query}
    messages.append(message_prompt)
 
    model = "gpt-3.5-turbo"
    response = openai.chat.completions.create(
        model="{}".format(model),  # 填写需要调用的模型名称
        messages=messages,
        temperature=0,  
        
    )
    result = response.choices[0].message.content
    return result



q = "What capital is associated with the birthplace country associated with the individual depicted in the image?"
with open('../prompts/divide.txt','r') as f:
    divide_base_prompt = f.read()

index = 1
question_prompt = divide_base_prompt.replace("<<<<QUESTION>>>>", q)
question_prompt = question_prompt.replace("<<<STEP>>>", str(index+2))
print("Question prompt: " + question_prompt)

output = run_llm_divide(question_prompt)
sub_questions = output.split('\n')
print("Sub-questions: " + str(sub_questions))
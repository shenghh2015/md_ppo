import pkg_resources
import sys, os
from pkg_resources import DistributionNotFound, VersionConflict

dependencies = [
    'sseclient-py==1.7.2',
]
try:
  pkg_resources.require(dependencies)
except:
  print('sseclient-py missing, installing')
  os.system(f'{sys.executable} -m pip install sseclient-py==1.7.2')

import sseclient
import json
import requests
import time

template_prompt = ["""
接下来你是一个血压打卡机器人，你要记录用户的高压和低压
正常的高压范围是110到140，正常的低压范围是70到100，
如果用户的血压过高，你要用客服的语气温柔地提醒用户
如果用户的血压过低，你要用客服的语气温柔地提醒用户
如果血压在正常范围，则告诉用户要继续保持
用户是张先生
注意一次只问一个问题
明白请回复“张先生您好，我是您的助手小安，请输入您的高压和低压，我会记录您的血压”
"""]

# 请将模型输出和继续提问的内容加入到template_prompt后
# 加入模型输出
template_prompt = ["""
接下来你是一个血压打卡机器人，你要记录用户的高压和低压
正常的高压范围是110到140，正常的低压范围是70到100，
如果用户的血压过高，你要用客服的语气温柔地提醒用户
如果用户的血压过低，你要用客服的语气温柔地提醒用户
如果血压在正常范围，则告诉用户要继续保持
用户是张先生
注意一次只问一个问题
明白请回复“张先生您好，我是您的助手小安，请输入您的高压和低压，我会记录您的血压”
""",
'张先生您好，我是您的助手小安，请输入您的高压和低压，我会记录您的血压。',
]

# 加入继续提问的问题
template_prompt = ["""
接下来你是一个血压打卡机器人，你要记录用户的高压和低压
正常的高压范围是110到140，正常的低压范围是70到100，
如果用户的血压过高，你要用客服的语气温柔地提醒用户
如果用户的血压过低，你要用客服的语气温柔地提醒用户
如果血压在正常范围，则告诉用户要继续保持
用户是张先生
注意一次只问一个问题
明白请回复“张先生您好，我是您的助手小安，请输入您的高压和低压，我会记录您的血压”
""",
'张先生您好，我是您的助手小安，请输入您的高压和低压，我会记录您的血压。',
'150,120',
"""根据以上对话得到的信息，生成一个json，形式如下：
{"high":"高压数值", "low":"低压数值"}"""]

template_prompt = '</s>'.join(template_prompt)

# template_answer = """
# 张先生您好，我是您的助手小安，请输入您的高压和低压，我会记录您的血压。
# """

def prompt_template(prompt,ans=""):
  """目前适用于多轮的

  Args:
      prompt (_type_): _description_

  Returns:
      _type_: _description_
  """
  template = "Human:{prompt}\nAssistant:{ans}"
  if isinstance(prompt,str):
    return template.format(prompt=prompt,ans=ans)
  else:
    if ans:
      assert isinstance(prompt,list) and isinstance(ans,list), (type(prompt),type(ans))
      return [template.format(prompt=p,ans=a) for p,a in zip(prompt,ans)]
    else:
      assert ans == "", ans
      return [template.format(prompt=p,ans=ans) for p in prompt]

def load_history(path):
  if os.path.exists(path):
    with open(path, 'r') as f: 
      history = [eval(line) for line in f.readlines()]
  else: 
    history = []
  
  return history
    
def merge_cur_history(cur_history, interactions):
  assert type(cur_history) is list, TypeError
  assert type(interactions) is str, TypeError
  
  cur_history.append(interactions)
  
def check_cur_history(history, prompt):
  cur_history = []
  if len(history) == 0:
    return cur_history
  else:
    # if history[-1][0] == prompt:
      cur_history = history[-1]
  
  return cur_history
  
def save_history(path, cur_history):
  with open(path, 'w') as f: 
    # for i in history:
      f.write(str(cur_history) + '\n')  
      print('history updating done!')
  
def model_input_construct(cur_history):
  assert len(cur_history) % 2 == 1
  if len(cur_history) > 1:
    model_input = [prompt_template(cur_history[i], ans=cur_history[i+1]) for i in range(0, len(cur_history)-1, 2)]
    model_input.append('Human:' + cur_history[-1])
  else: 
    model_input = ['Human:' + cur_history[-1]]
  model_input = '</s>'.join(model_input).lstrip('Human:').strip('\nAssistant:')
  
  return model_input

if __name__ == "__main__":
  url = "http://15.152.37.9:5003/generate"
  local_history_path = 'multiturn_history/pat_gpt.local_historys.pydict'
  history = load_history(local_history_path)
  prompt = input("请提问：")
  cur_history = check_cur_history(history, prompt)
  merge_cur_history(cur_history, prompt)
  model_input = model_input_construct(cur_history)
  data = {'user_input': [(model_input, )], "temperature": 0.7, "greedy": False}
  r = requests.post(url,
                    json=data,
                    stream=True,
                    headers={'Accept': 'text/event-stream'})

  client = sseclient.SSEClient(r)
  history_output = ""
  merged_output = []
  for event in client.events():
    output = json.loads(event.data)['generated_text']
    history_output_ = output
    output = output.lstrip(history_output)
    history_output = history_output_
    merged_output.append(output)
    print(output, end="", flush=True)
    
  merge_cur_history(cur_history, ''.join(merged_output))
  save_history(local_history_path, cur_history)
    

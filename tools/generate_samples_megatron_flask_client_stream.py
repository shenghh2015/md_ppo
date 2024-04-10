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

if __name__ == "__main__":
  url = "http://10.10.10.40:8887/generate"
  data = {'user_input': [("请介绍一下北京。", )], "temperature": 0.7, "greedy": False}
  r = requests.post(url,
                    json=data,
                    stream=True,
                    headers={'Accept': 'text/event-stream'})

  client = sseclient.SSEClient(r)
  history_output = ""
  for event in client.events():
    output = json.loads(event.data)['generated_text']
    history_output_ = output
    output = output.lstrip(history_output)
    history_output = history_output_
    print(output, end="", flush=True)

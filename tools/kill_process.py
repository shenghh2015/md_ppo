import re
import os
import subprocess
if __name__ == "__main__":
  with open("hostfile_inner_64") as f:
    nodes = f.readlines()
  nodes = [_.strip().split()[0] for _ in nodes]
  launch_node = nodes[0]
  command = "\"ps -ef | grep python | grep -v grep | awk '{print \\$2}' | xargs kill -9\""
  print(command)
  available_nodes = []
  used_nodes = []
  for node in nodes[1:] + [nodes[0]]:
    try:
      #output = subprocess.check_output(f"ssh -o PasswordAuthentication=no {node}".split() + [command])
      os.system(f"ssh -o PasswordAuthentication=no {node} {command}")
      #output = str(output)
      #gpu_used_mem = list(map(int, re.findall(r'\d+', output)))
      #if max(gpu_used_mem) > 1000:
      #    used_nodes.append(node)
      #else:
      #    available_nodes.append(node)
      #print(node, "slots=8")
    except subprocess.CalledProcessError:
      print(node, "unavailable")
      continue
  print("available_nodes", available_nodes)
  print("used_nodes", used_nodes)

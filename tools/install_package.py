import os
import subprocess
if __name__ == "__main__":
  with open("hostfile_inner_64") as f:
    nodes = f.readlines()
  nodes = [_.strip().split()[0] for _ in nodes]
  launch_node = nodes[0]
  print(len(nodes))
  print(launch_node)
  available_nodes = [launch_node]
  for node in nodes[1:]:
    try:
      subprocess.check_call(
          f"ssh -o PasswordAuthentication=no {node} 'python3 -m pip install deepspeed==0.6.5'",
          stderr=subprocess.DEVNULL,
          stdout=subprocess.DEVNULL,
          shell=True)
      available_nodes.append(node)
      print(node)
    except subprocess.CalledProcessError:
      print(node, "unavailable")
      continue
  print(available_nodes, "total: ", len(available_nodes))

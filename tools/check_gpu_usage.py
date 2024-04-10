import re
import os
import subprocess
if __name__ == "__main__":
    with open("hostfile_inner_64") as f:
        nodes = f.readlines()
    nodes = [_.strip().split(' ')[0] for _ in nodes]
    launch_node = nodes[0]
    print(len(nodes))
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    available_nodes = []
    used_nodes = []
    for node in nodes:
        try:
            output = subprocess.check_output(f"ssh -o PasswordAuthentication=no {node}".split() + [command])
            output = str(output)
            gpu_used_mem = list(map(int, re.findall(r'\d+', output)))
            if max(gpu_used_mem) > 1000:
                used_nodes.append(node)
                print(node, "used")
            else:
                available_nodes.append(node)
                print(node, "available")
        except subprocess.CalledProcessError:
            print(node, "unavailable")
            continue
    print("available_nodes", available_nodes)
    print("used_nodes", used_nodes)
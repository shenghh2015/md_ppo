"""
实现各类装饰器
"""
import time

def log_exe_time(fn):
  def wrapper(*args,**kwargs):
    start_time = time.time()
    ret = fn(*args,**kwargs)
    print(f'total execute time: {time.time()-start_time}s')
    return ret
  return wrapper
    


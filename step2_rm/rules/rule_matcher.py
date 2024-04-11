from palframe import *
from palframe import nlp
from palframe.nlp import Logger
from step2_rm.rules.rule_base import RuleBase
from step2_rm.rules.find_duplicate import FindDuplicate

def detect_abnormal_generation(text)-> dict:
  '''
  return {} if no anomaly is found, otherwise a dict with more details.
  '''
  is_english = RuleBase.is_english(text)
  if is_english:
    Logger.info(f"En detected and skip it: '{text}'")
    return {}

  solvers = [
    FindDuplicate(text),
  ]

  for solver in solvers:
    if solver.run():
      return solver.get_result()

  return {}


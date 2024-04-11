#coding: utf8
#author: Tian Xia

from palframe import *
from palframe import nlp
from palframe.chinese import _is_chinese_char

class RuleBase:
  def __init__(self, text):
    self._text = text
    self._matched = False

  @staticmethod
  def is_english(text):
    '''
    We only consider wheter the whole input is English.
    '''
    text = text.replace(",", " ").replace(".", " ")\
      .replace("，", " ").replace("。", " ")
    text = "".join(text.split())

    zh_num = len([1 for ch in text if _is_chinese_char(ord(ch)) ])
    ratio = 1 - zh_num / len(text)
    return ratio > 0.85

  def run(self)-> bool:
    '''
    return whether rule is activated.
    '''
    raise NotImplementedError()

  def _get_rule_description(self)-> str:
    raise NotImplementedError()

  def get_result(self)-> dict:
    if not self._matched:
      return {}

    return {
      "ruler": str(self.__class__),
      "rule_description": self._get_rule_description(),
      "matched": self._get_matched_information()
    }

  def _get_matched_information(self)-> dict:
    '''
    if self.run() return true, then this function returns more details of
    matched information.
    '''
    raise NotImplementedError()


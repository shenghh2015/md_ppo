from palframe import *
from palframe import nlp
from step2_rm.rules.rule_base import RuleBase

class FindDuplicate(RuleBase):
  def __init__(self, text):
    super().__init__(text)

    self._text = "".join(text.split())
    self._duplication = {"substr": "", "freq": 0}

  def _get_rule_description(self):
    return "We consider the case like 'abcabcabcabc'"

  def _get_min_freq(self, str_len):
    assert str_len > 0
    if str_len == 1:
      return 5
    elif 2 <= str_len < 5:
      return 3
    else:
      return 2

  def run(self):
    self._matched = self._search(self._text)
    return self._matched

  def _get_matched_information(self) -> dict:
    return self._duplication

  def _count_string(self, text, f, substr):
    size = len(substr)
    freq = 0
    while True:
      mstr = text[f: f + size]
      if mstr != substr:
        return freq
      freq += 1
      f += size

    return freq

  def _detect(self, l, pos, min_freq):
    text = self._text
    if pos >= len(text):
      return False

    ch = text[pos]
    for freq in range(min_freq - 1):
      pos += l
      if not (pos < len(text) and text[pos] == ch):
        return False

    return True

  def _search(self, text):
    next_p_list = [None] * len(text)
    last_p_dict = {}
    for p, ch in enumerate(text):
      last_p = last_p_dict.setdefault(ch, p)
      if last_p == p:
        continue
      next_p_list[last_p] = p
      last_p_dict[ch] = p

    for f in range(len(text)):
      p = f
      while next_p_list[p] is not None:
        l = next_p_list[p] - f
        min_freq = self._get_min_freq(l)
        if l > math.ceil(len(text) / min_freq):
          break

        for rp in range(l):
          if not self._detect(l, f + rp, min_freq):
            p = next_p_list[p]
            break
        else:
          self._matched = True
          substr = text[f: f + l]
          self._duplication["substr"] = substr
          self._duplication["freq"] = self._count_string(text, f, substr)
          return True

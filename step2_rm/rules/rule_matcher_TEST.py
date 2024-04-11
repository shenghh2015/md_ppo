from palframe import *
from palframe import nlp
from step2_rm.rules.rule_matcher import detect_abnormal_generation

cases_to_test = {
  # "case001.txt",
  # "case002.txt",
  # "case003.txt",
  # "case004.txt",
  # "case005.txt",
  # "case006.txt",
  # "case007.txt",
  # "case008.txt",
  "temp.txt"
}

def get_test_file():
  case_dir = os.path.join(os.path.split(__file__)[0], "cases")
  for case_file in sorted(nlp.get_files_in_folder(case_dir, ["txt"])):
    short_file = os.path.basename(case_file)
    if short_file not in cases_to_test:
      continue

    with open(case_file) as f:
      test_content = f.read()
      yield {
        "file": case_file,
        "base_name": os.path.basename(case_file),
        "content": test_content
      }

def test_cases():
  cases = list(get_test_file())
  with nlp.Timer("All cases"):
    for case in cases:
      print("=" * 64)
      f = case["file"]
      text = case["content"]

      with nlp.Timer("testing: " + case["base_name"]):
        status = detect_abnormal_generation(text)
        if status != {}:
          print("Found")
          print(status)

      print()

if __name__ == "__main__":
  test_cases()

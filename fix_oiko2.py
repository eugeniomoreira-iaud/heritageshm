import json
import sys

path = r'd:\Jupyter\neuralprophet\auxiliary\oiko.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the specific bad string
# The file has a literal newline inside a JSON string.
bad_str = '    "    print(\'\\u26a0 Missing values:\')\n",\n'
good_str = '    "    print(\'\\u26a0 Missing values:\\n\')\\n",\n'

content = content.replace(bad_str, good_str)

# Just in case the unicode is represented differently in python source:
bad_str2 = '    "    print(\'⚠ Missing values:\')\n",\n'
good_str2 = '    "    print(\'⚠ Missing values:\\n\')\\n",\n'
content = content.replace(bad_str2, good_str2)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

try:
    json.loads(content)
    print("JSON is now valid!")
except Exception as e:
    print("Error:", e)

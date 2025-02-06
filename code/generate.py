import os
import glob

path = "E:/vscode/code/check/data/datasets"

json_file = glob.glob(os.path.join(path, "*.json"))
for file in json_file:
    os.system("labelme_json_to_dataset.exe %s" % (file))

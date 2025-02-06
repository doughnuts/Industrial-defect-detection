import os
import shutil

print(os.getcwd())
count = 0
path = "E:/vscode/code/check/data/Label_road04/Label"
for i in os.listdir(path=path):
    path_next = os.path.join(path, i)
    for j in os.listdir(path=path_next):
        path_final = os.path.join(path_next, j)
        for k in os.listdir(path_final):
            path_move = os.path.join(path_final, k)
            shutil.move(path_move, path)

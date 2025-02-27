import os,json,sys

for file in os.listdir():
    if ".json" in file:
        os.remove(file)
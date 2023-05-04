import os

def hello() :
    curr_dir = os.path.dirname(__file__)
    print(curr_dir)
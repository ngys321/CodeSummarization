import json
import random


input_file_dir = './data.jsonl'

# 1. .jsonl 파일 읽기
with open(input_file_dir, "r") as file:
    lines = file.readlines()

# 데이터를 랜덤하게 섞는다.
random.shuffle(lines)

# 지정된 크기대로 데이터를 나눈다.
a = 23679  # 예시 크기
b = 1000  # 예시 크기
c = 1000   # 예시 크기

train_data = lines[:a]
valid_data = lines[a:a+b]
test_data = lines[a+b:a+b+c]



train_file_dir = './train.jsonl'
valid_file_dir = './valid.jsonl'
test_file_dir = './test.jsonl'

# 각각의 데이터를 새로운 .jsonl 파일에 저장한다.
with open(train_file_dir, "w") as file:
    file.writelines(train_data)

with open(valid_file_dir, "w") as file:
    file.writelines(valid_data)

with open(test_file_dir, "w") as file:
    file.writelines(test_data)

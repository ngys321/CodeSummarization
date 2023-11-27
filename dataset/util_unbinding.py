import json

def unbinding(input_jsonl_file, output_code_file, output_comment_file):

    # .jsonl 파일 읽기
    with open(input_jsonl_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 결과를 .java 파일에 저장하기
    with open(output_code_file, 'w', encoding='utf-8') as f:
        for line in lines:
            code = json.loads(line)["code_tokens"]
            f.write(' '.join(code) + '\n')

    # 결과를 .txt 파일에 저장하기
    with open(output_comment_file, 'w', encoding='utf-8') as f:
        for line in lines:
            comment = json.loads(line)["docstring_tokens"]
            f.write(' '.join(comment) + '\n')

input_jsonl_file = './train.jsonl'
output_code_file = './train.java'
output_comment_file = './train.txt'
unbinding(input_jsonl_file, output_code_file, output_comment_file)

input_jsonl_file = './valid.jsonl'
output_code_file = './valid.java'
output_comment_file = './valid.txt'
unbinding(input_jsonl_file, output_code_file, output_comment_file)

input_jsonl_file = './test.jsonl'
output_code_file = './test.java'
output_comment_file = './test.txt'
unbinding(input_jsonl_file, output_code_file, output_comment_file)
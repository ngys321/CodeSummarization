import json


def binding(input_code_file, input_comment_file, output_jsonl_file):
    # .java 파일 읽기
    with open(input_code_file, 'r', encoding='utf-8') as f:
        codes = f.readlines()

    # .txt 파일 읽기
    with open(input_comment_file, 'r', encoding='utf-8') as f:
        comments = f.readlines()

    # 결과를 .jsonl 파일에 저장하기
    with open(output_jsonl_file, 'w', encoding='utf-8') as f:
        for code, comment in zip(codes, comments):
            # print(code)
            # print(comment)
            new_data = {}
            new_data["code_tokens"] = code.split()
            new_data["docstring_tokens"] = comment.split()
            f.write(json.dumps(new_data) + '\n')


if __name__ == "__main__":



    input_code_file = './data.code'
    input_comment_file = './data.comment'
    output_jsonl_file = './data.jsonl'
    binding(input_code_file, input_comment_file, output_jsonl_file)

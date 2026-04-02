import json
dataset=[]
with open('/share/xiayinan/GRPO-LEAD/data/test/OlympiadBench/data/olympiad_bench.json', 'r') as f:
    # 读取所有行 每行会是一个字符串
    for j in f.readlines():
        # 将josn字符串转化为dict字典
        data = json.loads(j)
        data["problem"] = data["question"]
        del data["question"]
        data["solution"] = data["solution"][0]
        data["answer"] = data["final_answer"][0]
        del data["final_answer"]
        dataset.append(data)
with open('/share/xiayinan/GRPO-LEAD/deepscaler/data/test/olympiad_bench.json', 'w', encoding='utf-8') as file:
    file.write(json.dumps(dataset, indent=2, ensure_ascii=False))
# import pyarrow.parquet as pq

# # 读取Parquet文件
# table = pq.read_table('/share/xiayinan/GRPO-LEAD/data/test/OlympiadBench/data/train-00000-of-00001.parquet')
# df = table.to_pandas()
# json_data = df.to_json(orient='records', lines=True)
# with open('/share/xiayinan/GRPO-LEAD/data/test/OlympiadBench/data/olympiad_bench.json', 'w') as f:
#     f.write(json_data)
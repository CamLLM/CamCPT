import semchunk
from transformers import AutoTokenizer
import os
import pandas as pd
import json

chunker = semchunk.chunkerify(AutoTokenizer.from_pretrained(""), chunk_size=2048)

txt_dir = [""]
save_dir = [""]

for a,b in zip(txt_dir,save_dir):
    result = {
        "text":[],
        "length":[],
    }
    with open(a,"r",encoding="utf-8") as f:
        text = f.read()
        chunks = chunker(text,progress = True)
        result["text"] = [j for j in chunks if len(j)>20]
        result["length"] = [len(j) for j in chunks if len(j)>20]

    # df = pd.DataFrame(result)
    # df.to_csv("Mbook_merged.csv",)  # index=False 不保存行索引 

    with open(b,"w",encoding = "utf-8") as f:
        for line in result["text"]:
            dt = {
                "messages":[{
                    "role":"assistant",
                    "content":line
                }]
            }
            f.write(json.dumps(dt,ensure_ascii=False))
            f.write('\n')
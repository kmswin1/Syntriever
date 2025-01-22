from langchain.chat_models import ChatOpenAI
import json

prompt=\
"""
You are a subject matter expert in your field with substantial accumulated knowledge in a specific subject or topic, validated by academic degrees, certifications, and/or years of professional experience in that field.
Question: {question}
Passage #1: {passage1}
Passage #2: {passage2}
Based on your expert knowledge, choose which passage is more proper to answer the given question.
Only answer as Passage #1 or Passage #2
"""

llm = ChatOpenAI(temperature=0.0, 
                max_tokens=32,
                model_name="gpt-4o",
                )
import sys
model=sys.argv[1]
dataset=sys.argv[2]

k=5
with open(f"{model}_{dataset}_stage2_res_top100.json", "r") as f:
    with open(f"{model}_{dataset}_ranked_gpt4o_top{k}.json", "w") as wf:
        for line in f:
            line = json.loads(line)
            q=line["query"]
            passages = line["retrieved_passages"][:k]
            for i in range(len(passages)):
                for j in range(i+1, len(passages)):
                    try:
                        ranked = llm.invoke(prompt.format(question=q, passage1=passages[i], passage2=passages[j])).content
                        if "#1" not in ranked:
                            wf.write(json.dumps({"query": q, "positive_passage": passages[j], "negative_passage": passages[i]})+"\n")
                        else:
                            wf.write(json.dumps({"query": q, "positive_passage": passages[i], "negative_passage": passages[j]})+"\n")
                    except:
                        pass

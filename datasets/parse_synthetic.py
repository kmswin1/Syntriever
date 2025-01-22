from langchain.chat_models import ChatOpenAI
import json

import sys
dataset = sys.argv[1]

positive_prompt=\
"""
You are a subject matter expert in your field with substantial accumulated knowledge in a specific subject or topic, validated by academic degrees, certifications, and/or years of professional experience in that field.

Question: {question}

Write a passage that elaborates on the question. Ensure that no false information is provided; all content must be entirely accurate. Present everything you are aware of, offering a comprehensive and detailed explanation. Do not include any unverified or speculative information.
"""

negative_prompt=\
"""
You are a subject matter expert in your field with substantial accumulated knowledge in a specific subject or topic, validated by academic degrees, certifications, and/or years of professional experience in that field.

Question: {question}

Write a passage that contains plausible but irrelevant context given the question.
"""

verification_prompt=\
"""
You are a subject matter expert in your field with substantial accumulated knowledge in a specific subject or topic, validated by academic degrees, certifications, and/or years of professional experience in that field.

Question: {question}
Passage: {passage}

Is the above passage relevant to the aforementioned question?
Answer with yes or no.
"""

llm = ChatOpenAI(temperature=0.0, 
                  max_tokens=4096,
                  model_name="gpt-4o",
                  )

with open(f"{dataset}_synthetic.jsonl", "a") as wf:
   with open(f"{dataset}/queries.jsonl", "r") as f:
       for line in f:
          line = json.loads(line)
          q=line["query"]
          query=positive_prompt.format(question=q)
          positive_response = llm.invoke(query).content
          query=negative_prompt.format(question=q)
          negative_response = llm.invoke(query).content
          query = q+"\nLet's Think step-by-step"
          cot = llm.invoke(query).content
          query=verification_prompt.format(question=q, passage=line["positive_passage"])
          pos_verification = llm.invoke(query).content
          query=verification_prompt.format(question=q, passage=line["negative_passage"])
          wf.write(json.dumps({"q_id" : line["q_id"], "query": q, "positive_passage": positive_response, "negative_passage": negative_response, "metadata": line["metadata"], "cot": cot, "positive_verification": pos_verification})+"\n")

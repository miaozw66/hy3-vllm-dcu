#!/usr/bin/env python3
"""把文档内容发给 hy3 模型（vLLM 服务），请求生成 50-100 字总结。"""
import json
import sys
import urllib.request

API_URL = "http://10.18.17.71:8000/v1/completions"
MODEL = "/data/model/hygon/Hy3-Channel-INT8-w8a8/models/hygon--Hy3-Channel-INT8-w8a8/snapshots/master"
DOC_PATH = "/data/mzw/海光卡大模型推理框架移植流程.md"

with open(DOC_PATH, "r", encoding="utf-8") as f:
    doc = f.read()

# max-model-len=8192，输入必须留出输出空间；按字符保守截断
MAX_CHARS = 6000
if len(doc) > MAX_CHARS:
    doc = doc[:MAX_CHARS] + "\n...(文档过长，此处截断)"

prompt = (
    "请阅读下面这份关于\"海光卡大模型推理框架移植流程\"的文档，"
    "然后用 50-100 个字生成一份中文总结。\n\n"
    "<文档开始>\n" + doc + "\n<文档结束>\n\n总结："
)

payload = {
    "model": MODEL,
    "prompt": prompt,
    "max_tokens": 256,
    "temperature": 0.3,
}

req = urllib.request.Request(
    API_URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.load(resp)
    print(out["choices"][0]["text"].strip())
    print("\n---meta---")
    print("usage:", out.get("usage"))
except urllib.error.HTTPError as e:
    print("HTTP error:", e.code, e.read().decode("utf-8", "ignore")[:2000], file=sys.stderr)
    sys.exit(1)

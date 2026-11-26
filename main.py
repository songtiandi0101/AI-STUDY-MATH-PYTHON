import base64
import json
import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict

# ===================== 配置区 =====================
DEEPSEEK_API_KEY = ""
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-flash"

PROMPT_QUESTION = """
请识别图片里的数学题目。请严格按以下 JSON 格式输出，不要包含任何 Markdown 标记或多余文字：
{
  "question": "提取的纯题干文字内容",
  "has_image": true 或 false (判断原图中是否包含几何图形、函数图像等视觉元素，纯文字为 false)
}
"""
# ===================================================

app = FastAPI(title="习题识别接口")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GraphState(TypedDict):
    img_data_url: str
    question_text: str
    has_image: bool


def recognize_question(state: GraphState):
    llm = ChatOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, model=MODEL_NAME, temperature=0.1)
    msg = HumanMessage(content=[
        {"type": "text", "text": PROMPT_QUESTION},
        {"type": "image_url", "image_url": {"url": state["img_data_url"]}}
    ])
    resp = llm.invoke([msg])

    try:
        content = resp.content.strip().replace("```json", "").replace("```", "")
        data = json.loads(content)
        return {
            "question_text": data.get("question", ""),
            "has_image": data.get("has_image", False)
        }
    except Exception as e:
        print("JSON解析失败:", e)
        return {"question_text": resp.content, "has_image": False}


builder = StateGraph(GraphState)
builder.add_node("recognize_question", recognize_question)
builder.add_edge(START, "recognize_question")
builder.add_edge("recognize_question", END)
graph = builder.compile()


# ==================== 智能图像裁剪核心逻辑 ====================
def crop_image_keep_only_graphics(img_bytes: bytes):
    """
    使用 OpenCV 投影法自动去除顶部题干文字，只保留下方的图像区域
    """
    try:
        # 1. 将字节流转换为 OpenCV 图像
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return img_bytes  # 解析失败原图返回

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2. 二值化处理（黑字白底）
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # 3. 计算水平方向上的黑像素投影（每一行有多少个黑点）
        row_sums = np.sum(thresh, axis=1)

        # 4. 找到连续有黑像素（或一定量黑像素）的起始和结束行
        # 如果某行黑像素极少，说明是空白处，视为文字与图像的分割点
        valid_rows = np.where(row_sums > 5)[0]
        if len(valid_rows) == 0:
            return img_bytes  # 全白图，直接返回

        # 寻找中间最大的空白跳跃段
        gaps = np.diff(valid_rows)
        max_gap_idx = np.argmax(gaps)

        # 如果找到明显的空白间隔，且间隔下方仍有图像，则切分
        if gaps[max_gap_idx] > 20:
            # 取间隔下方作为图像的起始位置
            start_y = valid_rows[max_gap_idx + 1]
            # 底部留一点边距
            end_y = min(valid_rows[-1] + 20, img.shape[0])

            # 裁切图像
            cropped = img[start_y:end_y, :]
            # 可选：对裁剪后的图像进行高清放大（2倍）
            # cropped = cv2.resize(cropped, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        else:
            cropped = img

        # 5. 转换回 bytes 格式
        _, buffer = cv2.imencode('.png', cropped)
        return buffer.tobytes()
    except Exception as e:
        print("图片裁剪失败:", e)
        return img_bytes


# ===================================================


# ==================== 接口定义 ====================
@app.post("/recognize_only")
async def recognize_only(file: UploadFile = File(...)):
    img_bytes = await file.read()
    mime = file.content_type

    b64 = base64.b64encode(img_bytes).decode("utf-8")
    img_data_url = f"data:{mime};base64,{b64}"

    # 跑大模型链路
    out = graph.invoke({"img_data_url": img_data_url, "question_text": "", "has_image": False})

    image_base64 = ""
    if out["has_image"]:
        # 【核心修改】：自动裁剪，去掉题干文字，仅保留图像
        cropped_bytes = crop_image_keep_only_graphics(img_bytes)
        # 将裁剪后的图片转为 Base64 发给前端展示
        b64_cropped = base64.b64encode(cropped_bytes).decode("utf-8")
        image_base64 = f"data:image/png;base64,{b64_cropped}"

    return {
        "code": 200,
        "msg": "识别完成",
        "data": {
            "question": out["question_text"],
            "has_image": out["has_image"],
            "image_base64": image_base64
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
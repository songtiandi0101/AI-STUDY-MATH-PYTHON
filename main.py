import base64
import oss2
import uuid
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import font_manager
from fastapi import FastAPI, UploadFile, File
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict
from fastapi.middleware.cors import CORSMiddleware

# ===================== 配置区 =====================
DEEPSEEK_API_KEY = ""
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-flash"  # 确保你的 API 支持多模态

OSS_ACCESS_KEY_ID = ""
OSS_ACCESS_KEY_SECRET = ""
OSS_ENDPOINT = "oss-cn-beijing.aliyuncs.com"
OSS_BUCKET_NAME = "math-question-img"

PROMPT_QUESTION = "识别图片里面的数学题干，只输出题干文字，不要答案、不要解析、不要多余内容"
PROMPT_EXPR = """
从下面数学题干中提取函数表达式，只输出表达式，严格遵守：
1. 有函数就输出，例如：(np.exp(x)-np.exp(-x))/(x**2)
2. 没有函数图像、不需要绘图，直接输出 NONE
3. 不要解释、不要markdown、不要多余文字
题干：
"""
# ===================================================

# ================= 全局 Matplotlib 风格配置 =================
# 1. 配置数学公式渲染器 (用于漂亮地显示公式)
plt.rcParams['mathtext.fontset'] = 'cm'  # 使用 Computer Modern 学术字体
# 2. 自动寻找系统中文字体
system_fonts = set(f.name for f in font_manager.fontManager.ttflist)
for font in ['Microsoft YaHei', 'PingFang SC', 'WenQuanYi Zen Hei', 'SimHei']:
    if font in system_fonts:
        plt.rcParams['font.sans-serif'] = [font]
        break
plt.rcParams['axes.unicode_minus'] = False

# 主题配色（干净、淡雅）
COLOR_LINE = "#409EFF"  # 主线条颜色（蓝色）
COLOR_AXIS = "#909399"  # 坐标轴颜色（浅灰）
COLOR_GRID = "#F2F6FC"  # 网格线颜色（极浅蓝灰）
COLOR_TEXT = "#606266"  # 刻度文字颜色
# ============================================================

app = FastAPI(title="手机习题识别+自动绘图（不存原图）")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)


class GraphState(TypedDict):
    img_data_url: str
    question_text: str
    expr_text: str


def recognize_question(state: GraphState):
    llm = ChatOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, model=MODEL_NAME, temperature=0.1)
    msg = HumanMessage(content=[
        {"type": "text", "text": PROMPT_QUESTION},
        {"type": "image_url", "image_url": {"url": state["img_data_url"]}}
    ])
    resp = llm.invoke([msg])
    return {"question_text": resp.content}


def extract_expr(state: GraphState):
    llm = ChatOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, model=MODEL_NAME, temperature=0.1)
    msg = HumanMessage(content=PROMPT_EXPR + state["question_text"])
    resp = llm.invoke([msg])
    return {"expr_text": resp.content.strip()}


# 节点3：终极美化绘图
def draw_function(expr: str):
    # 创建画布，高分辨率
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    x_min, x_max = -10, 10
    y_min, y_max = -10, 10
    x = np.linspace(x_min, x_max, 2000)

    safe_ns = {"np": np, "x": x}
    try:
        y = eval(expr, safe_ns, {})
        # 画函数曲线（渐变色视觉：深蓝实线，加上一点阴影或者直接实线）
        ax.plot(x, y, color=COLOR_LINE, linewidth=3, solid_capstyle='round', antialiased=True)
    except Exception as e:
        print(f"绘图失败，表达式可能不支持: {e}")
        return None

    # 将计算机公式转为 LaTeX 格式用于渲染
    # 例如：(np.exp(x)-np.exp(-x))/(x**2) -> \frac{e^x - e^{-x}}{x^2}
    latex_expr = expr
    latex_expr = latex_expr.replace("np.exp(x)", "e^x").replace("np.exp(-x)", "e^{-x}")
    latex_expr = latex_expr.replace("**", "^").replace("*", " \\cdot ")
    if "/" in latex_expr and "^" in latex_expr:
        # 尝试给简单除法加个大括号让渲染像分数（注意：复杂正则处理，如果没括号可能会显示不完美）
        latex_expr = latex_expr.replace("(", "\\left(").replace(")", "\\right)")

    # 在图表左上角标注公式，使用数学字体
    # 使用 bbox 画一个半透明白色圆角背景，防止公式被曲线遮挡
    ax.text(0.05, 0.95, f"$y = {latex_expr}$",
            transform=ax.transAxes,
            fontsize=20,
            color="#303133",
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFFF", ec="#EBEEF5", alpha=0.9, linewidth=1))

    # 隐藏上、右、左、下边框，完全依靠网格线支撑视觉
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # 绘制十字坐标轴（带箭头指示正方向）
    ax.axhline(y=0, color=COLOR_AXIS, linewidth=1.2, zorder=3)
    ax.axvline(x=0, color=COLOR_AXIS, linewidth=1.2, zorder=3)

    # 坐标轴箭头
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False, color=COLOR_AXIS, markersize=5)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, color=COLOR_AXIS, markersize=5)
    # 原点标签
    ax.text(0, 0, 'O', fontsize=12, color=COLOR_TEXT, ha='right', va='top', transform=ax.transData)

    # 范围与刻度设置
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # 隐藏默认的边框刻度线，只保留文字
    ax.tick_params(axis='both', colors=COLOR_TEXT, labelsize=10, length=0, pad=8)

    # 网格（坐标轴对齐）
    ax.grid(True, color=COLOR_GRID, linestyle='-', linewidth=1, zorder=0)

    # 刻度对齐设置，每2格显示一次
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))

    # 留白自适应
    plt.tight_layout()

    # 导出内存图片
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2, facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf.read()


# 构建 LangGraph 链路
builder = StateGraph(GraphState)
builder.add_node("recognize_question", recognize_question)
builder.add_node("extract_expr", extract_expr)
builder.add_edge(START, "recognize_question")
builder.add_edge("recognize_question", "extract_expr")
builder.add_edge("extract_expr", END)
graph = builder.compile()


@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    mime = file.content_type

    b64 = base64.b64encode(img_bytes).decode("utf-8")
    img_data_url = f"data:{mime};base64,{b64}"

    out = graph.invoke({"img_data_url": img_data_url, "question_text": "", "expr_text": ""})
    question = out["question_text"]
    expr = out["expr_text"]

    draw_img_url = None
    draw_img_key = None
    if expr != "NONE":
        draw_bytes = draw_function(expr)
        if draw_bytes is not None:
            draw_key = f"question_draw/{uuid.uuid4()}.png"
            bucket.put_object(draw_key, draw_bytes)
            draw_img_url = f"https://{OSS_BUCKET_NAME}.{OSS_ENDPOINT}/{draw_key}"
            draw_img_key = draw_key

    return {
        "code": 200,
        "msg": "识别完成",
        "data": {
            "question": question,
            "func_expr": expr,
            "draw_img_key": draw_img_key,
            "draw_img_url": draw_img_url
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
# -*- coding: utf-8 -*-
# 导入所需库
import gradio as gr
from openai import OpenAI
import webbrowser  # 导入webbrowser模块用于自动打开链接

# -------------------- 客户端初始化 --------------------
# 初始化OpenAI客户端（适配阿里云平台）
client = OpenAI(
    api_key="sk-145d98eda0454bc2b0fccd036efada63",  # 从环境变量获取API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云兼容接口
)


# 模型配置
AVAILABLE_MODELS = [
    "deepseek-r1",
    "deepseek-v3",
    "qwen-omni-turbo-latest",
    "llama3.3-70b-instruct"
]
HISTORY_ROUNDS = 10  # 默认保留10轮

# 初始化消息历史记录
messages_history = []


# -------------------- 核心聊天函数 --------------------
def chat_with_model(user_input, chat_output, selected_model):
    """
    与DeepSeek模型进行交互的核心函数

    参数：
    user_input (str): 用户输入的文本
    chat_output (str): 当前聊天输出内容
    selected_model (str): 选择的模型名称

    返回：
    generator: 通过yield逐步返回生成的聊天内容
    """
    # 将用户输入添加到消息历史记录
    messages_history.append({'role': 'user', 'content': user_input})

    recent_messages = messages_history[-HISTORY_ROUNDS*2:]  # 每轮包含user和assistant两条

    # 初始化思考过程和回答内容
    reasoning_content = ""  # 存储模型的推理过程
    answer_content = ""  # 存储最终回答

    # 格式化输出内容
    chat_output += f"\n\n"
    chat_output += f"  \n**输入:** {user_input}\n"  # 显示用户输入
    chat_output += f"  \n<span style='color: gray;'>**思考过程:** </span> \n"  # 灰色思考过程标题

    # 调用阿里云API进行流式响应
    completion = client.chat.completions.create(
        model=selected_model,  # 使用选择的模型
        messages=recent_messages,  # 传入最近对话上下文
        stream=True  # 启用流式传输
    )

    begin_answer = 0  # 回答部分开始标记
    for chunk in completion:
        delta = chunk.choices[0].delta

        # 获取模型的推理过程内容（如果存在）
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_chunk = delta.reasoning_content
            reasoning_content += reasoning_chunk
            # 用灰色显示思考过程
            chat_output += f"<span style='color: gray;'>{reasoning_chunk}</span>"

        # 获取模型的回答内容
        if hasattr(delta, "content") and delta.content:
            if begin_answer == 0:  # 首次出现回答时添加标题
                chat_output += f"  \n**回答:** \n"
                begin_answer = 1
            answer_chunk = delta.content
            answer_content += answer_chunk
            chat_output += answer_chunk  # 直接显示回答内容

        yield chat_output  # 逐步返回生成内容

    # 将助手的最终回答添加到消息历史记录
    messages_history.append({'role': 'assistant', 'content': answer_content})


# -------------------- 清空对话函数 --------------------
def clear_conversation(chat_output):
    """
    清空对话历史和聊天界面

    参数：
    chat_output (str): 当前聊天输出内容

    返回：
    str: 清空后的空字符串
    """
    global messages_history
    messages_history = []  # 重置消息历史
    chat_output = ""  # 清空聊天显示
    return chat_output


# -------------------- Gradio界面构建 --------------------
with gr.Blocks() as demo:  # 创建块状布局界面
    gr.Markdown("# rain's Chatbot with Multi-Model Support")  # 更新标题

    # 模型选择区域
    with gr.Row():
        model_selector = gr.Dropdown(
            label="选择模型",
            choices=AVAILABLE_MODELS,
            value=AVAILABLE_MODELS[0],
            interactive=True
        )

    # 聊天显示区域
    with gr.Row():
        with gr.Column():
            chat_output = gr.Markdown(label="Chat", value="")

    # 输入控制区域
    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(
                label="输入",
                placeholder="Type your message here...",
                lines=2
            )
            with gr.Row():
                submit_button = gr.Button("Submit")  # 提交按钮
                clear_button = gr.Button("开始新对话")  # 清空按钮

    # 绑定按钮事件
    submit_button.click(
        fn=chat_with_model,  # 触发聊天函数
        inputs=[user_input, chat_output, model_selector],  # 新增模型选择输入
        outputs=chat_output  # 输出目标
    )

    clear_button.click(
        fn=clear_conversation,  # 触发清空函数
        inputs=[chat_output],  # 输入参数
        outputs=chat_output  # 输出目标
    )

# -------------------- 启动应用 --------------------
# 启动Gradio应用并自动打开链接
webbrowser.open("http://127.0.0.1:7860")
demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False)
# 支持图片理解
# -*- coding: utf-8 -*-
# 导入所需库
import gradio as gr
from openai import OpenAI
import base64
import webbrowser

# -------------------- 客户端初始化 --------------------
# 初始化OpenAI客户端（适配阿里云平台）
client = OpenAI(
    api_key="your_key",  # 输入API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云兼容接口
)


# -------------------- 全局配置 --------------------
MODEL_CONFIG = {
    "available_models": [
        "deepseek-r1",
        "deepseek-v3",
        "qwen-omni-turbo-latest",
        "llama3.3-70b-instruct"
    ],
    "default_model": "deepseek-r1",
    "max_history": 8
}

# -------------------- 系统核心功能 --------------------
class ChatSystem:
    def __init__(self):
        self.history = []
        self.chat_records = []
        self.stop_flag = False  # 新增停止标志

# 用于处理图片，调用qwen-vl-plus模型
    def _process_image(self, image_path):
        try:
            with open(image_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")
            
            response = client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "详细描述图片内容，包括主要物体、场景特征、文字信息等"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_data}"
                        }}
                    ]
                }]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ 图片解析失败：{str(e)}"

    def stop_generation(self):  # 停止方法
        self.stop_flag = True

    def generate_response(self, user_input, selected_model, image_path=None):
        self.stop_flag = False  # 每次生成前重置标志
        
        # 处理图片并合并到用户输入
        combined_input = user_input
        if image_path:
            analysis_result = self._process_image(image_path)
            combined_input = f"{user_input}\n[图片分析]: {analysis_result}"
        
        # 记录用户输入（仅显示原始内容）
        user_entry = f'<div class="user-message">{user_input}</div>'
        self.chat_records.append(user_entry)
        
        # 实际发送合并后的内容
        self.history.append({"role": "user", "content": combined_input})
        context = self.history[-MODEL_CONFIG["max_history"]*2:]
        
        full_response = ""
        reasoning_content = ""
        try:
            stream = client.chat.completions.create(
                model=selected_model,
                messages=context,
                stream=True,
                temperature=0.7
            )
            
            for chunk in stream:
                if self.stop_flag:  # 检查停止标志
                    full_response += "<br>（回答已中断）"
                    break
                
                delta = chunk.choices[0].delta
                
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                
                if hasattr(delta, "content") and delta.content:
                    full_response += delta.content
                
                current_display = "<div class='chat-container'>" + "\n".join(self.chat_records) + \
                    f'<div class="assistant-message"><div class="response">{full_response}</div>' + \
                    (f'<div class="reasoning">{reasoning_content}</div>' if reasoning_content else "") + "</div></div>"
                yield current_display
            
            self.history.append({"role": "assistant", "content": full_response})
            self.chat_records.append(
                f'<div class="assistant-message"><div class="response">{full_response}</div>' + \
                (f'<div class="reasoning">{reasoning_content}</div>' if reasoning_content else "") + "</div>"
            )
        except Exception as e:
            error_msg = f'<div class="error-message">🔴 系统错误：{str(e)}</div>'
            self.chat_records.append(error_msg)
            yield "<div class='chat-container'>" + "\n".join(self.chat_records) + "</div>"

    def clear_history(self):
        self.history = []
        self.chat_records = []
        return "<div style='color:#7f8c8d; text-align: center;'>✅ 对话历史已重置</div>"

# -------------------- 界面实现 --------------------
class ChatInterface:
    def __init__(self, system):
        self.system = system
        self.demo = self._create_interface()
    
    def _create_interface(self):
        custom_css = """
        #chat-box { 
            height: 55vh !important;
            overflow-y: auto !important;
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 10px;
            background: #f9f9f9;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .user-message {
            align-self: flex-end;
            background: #e3f2fd;
            padding: 10px 15px;
            border-radius: 15px;
            max-width: 70%;
            word-break: break-word;
            margin-left: 30%;
        }
        .assistant-message {
            align-self: flex-start;
            background: #f5f5f5;
            padding: 10px 15px;
            border-radius: 15px;
            max-width: 70%;
            word-break: break-word;
            margin-right: 30%;
        }
        .reasoning {
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px dashed #ddd;
        }
        .error-message {
            color: #e74c3c;
            padding: 10px;
            text-align: center;
        }
        .input-section {
            padding: 12px;
            background: #ffffff;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* 新增按钮高度样式 */
        #submit-btn {
            height: 70px !important;  /* 设置按钮高度为60px */
        }
        """

        with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="多模态AI助手") as interface:
            gr.Markdown("""<h1 style="text-align: center; color: #2c3e50;">🌈 多模态AI助手</h1>
            <p style="text-align: center; color: #7f8c8d;">支持图片理解 | 多模型切换 | 上下文记忆</p>
            <hr style="margin-bottom: 20px;">""")
            
            with gr.Row():
                with gr.Column(scale=3):
                    chat_display = gr.HTML(
                        elem_id="chat-box",
                        value="<div style='color:#7f8c8d; text-align: center;'>欢迎使用智能助手！</div>"
                    )
                
                with gr.Column(scale=1):
                    with gr.Accordion("📷 图像处理区", open=True):
                        image_upload = gr.Image(
                            label="上传图片",
                            type="filepath",
                            height=140,
                            interactive=True
                        )
                    
                    with gr.Accordion("⚙️ 模型控制", open=True):
                        model_selector = gr.Dropdown(
                            label="选择AI模型",
                            choices=MODEL_CONFIG["available_models"],
                            value=MODEL_CONFIG["default_model"],
                            interactive=True
                        )
                        with gr.Row():
                            clear_btn = gr.Button("🔄 新对话", variant="secondary")
                            stop_btn = gr.Button("⏹️ 停止生成", variant="stop")
            
            with gr.Row(variant="panel"):
                with gr.Column(scale=3):
                    user_input = gr.Textbox(
                        placeholder="请输入您的问题...",
                        lines=2,
                        max_lines=4,
                        show_label=False
                    )
                with gr.Column(scale=1):
                    submit_btn = gr.Button("🚀 发送", variant="primary", elem_id="submit-btn")
            
            submit_event = submit_btn.click(
                self.system.generate_response,
                [user_input, model_selector, image_upload],
                chat_display
            ).then(
                lambda: [None, ""],
                outputs=[image_upload, user_input]
            )
            
            user_input.submit(
                self.system.generate_response,
                [user_input, model_selector, image_upload],
                chat_display
            ).then(
                lambda: [None, ""],
                outputs=[image_upload, user_input]
            )
            
            clear_btn.click(
                self.system.clear_history,
                outputs=chat_display
            )
            
            stop_btn.click(
                lambda: self.system.stop_generation(),
                inputs=None,
                outputs=None,
            )
        
        return interface

# -------------------- 启动系统 --------------------
if __name__ == "__main__":
    chat_system = ChatSystem()
    interface = ChatInterface(chat_system)
    webbrowser.open("http://127.0.0.1:7860")
    interface.demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
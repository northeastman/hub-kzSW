"""
简化版多轮对话 ReAct Agent
"""

import os
import json
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

COLORS = {
    "thought": "\033[36m", "action": "\033[33m", "obs": "\033[32m",
    "final": "\033[35m", "error": "\033[31m", "user": "\033[34m", "reset": "\033[0m"
}

def _c(color, text):
    return f"{COLORS[color]}{text}{COLORS['reset']}"

class ChatAgent:
    """支持多轮对话的 ReAct Agent"""
    
    def __init__(self, max_steps=10):
        self.max_steps = max_steps
        self.conversation_history = []
        
        from react_manual import run as run_manual, MODEL
        self.run_fn = run_manual
        self.model_name = MODEL
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        print(_c("action", "🔄 对话历史已清空"))
    
    def get_history_for_llm(self):
        """获取对话历史"""
        history = []
        for entry in self.conversation_history:
            if entry["role"] == "user":
                history.append({"role": "user", "content": entry["content"]})
            elif entry["role"] == "assistant" and "answer" in entry:
                history.append({"role": "assistant", "content": entry["answer"]})
        return history
    
    def process_question(self, question):
        """处理单个问题"""
        print(f"\n{'='*60}")
        print(f"问题: {question}")
        print(f"模型: {self.model_name}  历史轮数: {len(self.conversation_history)//2}")
        print('='*60)
        
        final_answer = ""
        history = self.get_history_for_llm()
        
        for step_data in self.run_fn(question, max_steps=self.max_steps, history=history):
            stype = step_data["type"]
            
            if stype == "action":
                print(f"\n[Step {step_data['step']}]")
                print(_c("thought", f"🧠 {step_data['thought']}"))
                print(_c("action",  f"🔧 {step_data['action']}({json.dumps(step_data['action_input'], ensure_ascii=False)})"))
                print(_c("obs",     f"👁 {step_data['observation'][:300]}"))
            
            elif stype == "final":
                print(f"\n{'─'*60}")
                print(_c("final", f"✅ {step_data['answer']}"))
                final_answer = step_data["answer"]
            
            elif stype in ("error", "max_steps"):
                print(_c("error", f"⚠️ {step_data.get('answer', step_data.get('observation', ''))}"))
                final_answer = step_data.get("answer", "处理失败")
        
        return final_answer
    
    def chat_loop(self):
        """启动交互式对话"""
        print(_c("final", "\n🚀 ReAct Financial Agent 多轮对话模式"))
        print(_c("action", "输入问题开始，输入 'exit' 退出，'clear' 清空历史"))
        print('='*60)
        
        while True:
            try:
                question = input(_c("user", "\n请输入问题: ")).strip()
            except (KeyboardInterrupt, EOFError):
                print(_c("action", "\n👋 再见！"))
                break
            
            if not question:
                continue
            
            if question.lower() in ("exit", "quit", "bye"):
                print(_c("action", "👋 再见！"))
                break
            
            if question.lower() == "clear":
                self.clear_history()
                continue
            
            answer = self.process_question(question)
            
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "answer": answer})
            
            print(_c("action", f"\n📝 当前对话轮数: {len(self.conversation_history)//2}"))

def chat_main(max_steps=10):
    """启动多轮对话"""
    agent = ChatAgent(max_steps=max_steps)
    agent.chat_loop()

if __name__ == "__main__":
    chat_main()
import openai
import time
import yaml
import tiktoken

class SkillExecutor:
    def __init__(self, config):
        self.config = config
        self.client = openai.OpenAI(
            api_key=self.config["llm"]["api_key"],
            base_url=self.config["llm"]["base_url"],
        )
        self.model = config["llm"]["model"]
        self.max_tokens = config["llm"]["max_tokens"]
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")



    def load_skill(self, skill_path):
        """加载技能文件"""
        with open(skill_path,"r",encoding = "utf-8") as f:
            return f.read()

    def count_tokens(self, text):
        """计算文本中的token数"""
        return len(self.encoding.encode(text))
    def execute_task(self,skill_content,task_prompt):
        """执行单个任务"""
        #构建完整 prompt
        full_prompt = f"{skill_content}\n\n---\n\n用户请求:{task_prompt}"
        #记录输入token数
        input_tokens = self.count_tokens(full_prompt)
        #记录开始时间
        start_time = time.time()
        #调用LLM
        response = self.client.chat.completions.create(
            model = self.model,
            messages=[
                {"role":"system","content":skill_content},
                {"role":"user","content":task_prompt},
            ],
            max_tokens = self.max_tokens
        )
        #记录结束时间
        end_time = time.time()
        #获取输出
        output_content = response.choices[0].message.content
        output_tokens = self.count_tokens(output_content)

        #计算耗时
        response_time = end_time - start_time

        #返回结果
        return {
            "input_tokens":input_tokens,
            "output_tokens":output_tokens,
            "total_tokens":input_tokens + output_tokens,
            "response_time":response_time,
            "output_content":output_content,

        }

import openai
import yaml

class SkillOptimizer:
    def __init__(self, config):
        """初始化优化器"""
        self.client = openai.OpenAI(
            api_key=config["llm"]["api_key"],
            base_url=config["llm"]["base_url"]
        )
        self.model = config["llm"]["model"]

    def analyze_skill(self, skill_content):
        """分析 Skill 并生成优化建议"""
        prompt = f"""请分析以下 Skill 内容，找出可以精简的地方：
    
    【严格限制】
    - 只能删除或合并重复内容
    - 禁止添加原 Skill 没有的内容
    - 禁止改变核心关键词（如"是什么"、"为什么"、"实践"、"测试"）
    - 禁止使用 Markdown 表格
    - 目标：Token 减少 30% 以上
    
    【分析要求】
    请输出：
    1. 找出重复/冗余的段落（列出具体位置）
    2. 建议合并哪些章节
    3. 预估优化后 Token 数量
    
    Skill 内容：
    {skill_content[:3000]}
    """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content

    def generate_optimized_skill(self, skill_content, analysis):
        """根据分析结果生成优化版 Skill"""
        prompt = f"""请根据分析结果，精简这个 Skill：
    
    【原始 Skill】
    {skill_content[:3000]}
    
    【优化分析】
    {analysis}
    
    【严格优化规则】
    1. 只能删除冗余内容，不能添加新内容
    2. 只能合并重复章节，不能改变结构
    3. 保留所有核心关键词：是什么、为什么、例子、实践、测试
    4. 禁止使用 Markdown 表格
    5. 禁止添加代码块模板
    6. 保持原有的教学逻辑顺序
    7. 目标：Token 减少 30% 以上
    
    【输出要求】
    - 直接输出优化后的完整 Skill
    - 不要解释优化过程
    - 不要省略任何核心教学模块
    """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000
        )
        return response.choices[0].message.content

    def optimize(self, skill_content, output_path):
        """完整优化流程"""
        print("\n📊 正在分析 Skill...")
        analysis = self.analyze_skill(skill_content)
        print(f"分析结果：\n{analysis}")

        print("\n🔧 正在生成优化版 Skill...")
        optimized_content = self.generate_optimized_skill(skill_content, analysis)

        # 保存 V2
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(optimized_content)

        print(f"\n✅ 优化版已保存到: {output_path}")

        return optimized_content
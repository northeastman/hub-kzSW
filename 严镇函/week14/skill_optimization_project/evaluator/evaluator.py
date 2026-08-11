import json
import os
from datetime import datetime

class Evaluator:
    def __init__(self,config):
        self.config = config
        self.logs_dir = config["experiment"]["logs_dir"]
        self.quality_weight = config["evaluation"]["quality_weight"]
        self.token_weight = config["evaluation"]["token_weight"]
    def evaluate_quality(self,output_content,taks_prompt):
        score = 0.0
        #长度评分
        length = len(output_content)
        if length > 200:
            score += 0.3
        if length > 500:
            score += 0.2
        # 2. 是否包含关键教学元素
        keywords = ["教学","教学方法","教学技巧","教学资源","是什么", "为什么", "例子", "实践", "测试"]
        keywords_count = sum(1 for kw in keywords if kw in output_content)
        score += (keywords_count /len(keywords)) *0.5

        return score
    def calculate_score(self,result,quality_score):
        """计算最终评分"""
        #Token评分 越少越好基准2000token
        token_baseline = 2000
        token_efficiency= max(0,1-result["total_tokens"]/token_baseline)

        #综合评分
        total_score = (quality_score * self.quality_weight+token_efficiency * self.token_weight)
        return {
            "quality_score": quality_score,
            "token_efficiency": token_efficiency,
            "total_score": total_score
        }
    def save_result(self,result,scores,taks_id,skill_version):
        """保存评估结果"""
        log_data = {
            "task_id" :taks_id,
            "skill_version" :skill_version,
            "timestamp" :datetime.now().isoformat(),
            "metrics" :{
                "input_tokens" : result["input_tokens"],
                "output_tokens" : result["output_tokens"],
                "total_tokens" : result["total_tokens"],
                "response_time" : result["response_time"]
            },
            "score" : scores,
            "output_preview" : result["output_content"][:200]+"..."

        }
        # 保存文件
        filename = f"{self.logs_dir}/run_{skill_version}_{taks_id}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        return filename

    def compare_results(self, v1_results, v2_results):
        """对比 V1 和 V2 的结果"""
        print("\n" + "="*50)
        print("=== Skill 优化对比报告 ===")
        print("="*50)

        print(f"\n{'指标':<15} {'V1':<15} {'V2':<15} {'变化':<15}")
        print("-"*60)

        # Token 对比
        v1_tokens = v1_results["metrics"]["total_tokens"]
        v2_tokens = v2_results["metrics"]["total_tokens"]
        token_change = ((v2_tokens - v1_tokens) / v1_tokens) * 100
        print(f"{'Total Tokens':<15} {v1_tokens:<15} {v2_tokens:<15} {token_change:+.1f}%")

        # 质量对比
        v1_quality = v1_results["scores"]["quality_score"]
        v2_quality = v2_results["scores"]["quality_score"]
        quality_change = ((v2_quality - v1_quality) / v1_quality) * 100 if v1_quality > 0 else 0
        print(f"{'Quality Score':<15} {v1_quality:<15.2f} {v2_quality:<15.2f} {quality_change:+.1f}%")

        # 综合评分
        v1_total = v1_results["scores"]["total_score"]
        v2_total = v2_results["scores"]["total_score"]
        total_change = ((v2_total - v1_total) / v1_total) * 100 if v1_total > 0 else 0
        print(f"{'Total Score':<15} {v1_total:<15.2f} {v2_total:<15.2f} {total_change:+.1f}%")

        print("="*50)
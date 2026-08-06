"""
渐进式 Skill Harness — 主编排器

流水线:
  Stage 0 INDEX    → scan_skills_dir，仅元数据
  Stage 1 MATCH    → 规则/LLM 选 skill（仍只用元数据）
  Stage 2 BODY     → 加载 SKILL.md 正文
  Stage 3 RESOURCE → 按意图加载 references/scripts
  Stage 4 EXECUTE  → 调用脚本或 LLM 完成任务
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from skill_executor import ExecResult, execute_skill
from skill_loader import LoadStage, ProgressiveSkillLoader
from skill_matcher import MatchResult, get_skill_by_name, match_skill
from skill_registry import SkillMeta, build_index_prompt, scan_skills_dir


@dataclass
class HarnessReport:
    query: str
    skills_index: list[SkillMeta] = field(default_factory=list)
    index_chars: int = 0
    match: MatchResult | None = None
    loaded_paths: list[str] = field(default_factory=list)
    total_loaded_chars: int = 0
    exec_result: ExecResult | None = None
    stage_log: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.stage_log.append(msg)


class ProgressiveSkillHarness:
    def __init__(self, skills_root: Path | None = None, output_dir: Path | None = None):
        root = skills_root or (Path(__file__).parent.parent / "skills")
        self.skills_root = root.resolve()
        self.output_dir = (output_dir or Path(__file__).parent / "output").resolve()
        self._index: list[SkillMeta] | None = None

    def index(self, force: bool = False) -> list[SkillMeta]:
        if self._index is None or force:
            self._index = scan_skills_dir(self.skills_root)
        return self._index

    def run(
        self,
        query: str,
        *,
        prefer_llm_match: bool = False,
        use_llm_exec: bool = False,
        verbose: bool = True,
    ) -> HarnessReport:
        report = HarnessReport(query=query)

        # Stage 0
        skills = self.index()
        report.skills_index = skills
        report.index_chars = sum(s.index_chars for s in skills)
        report.log(f"[Stage 0 INDEX] 扫描 {self.skills_root} → {len(skills)} 个 skill，索引共 {report.index_chars} 字符")
        if verbose:
            for s in skills:
                report.log(f"  · {s.name}: {s.description[:80]}…")

        # Stage 1
        match = match_skill(query, skills, prefer_llm=prefer_llm_match)
        report.match = match
        if not match or not match.skill_name:
            report.log("[Stage 1 MATCH] 未匹配到 skill")
            return report

        report.log(
            f"[Stage 1 MATCH] {match.method} → {match.skill_name} "
            f"(confidence={match.confidence:.2f}, {match.reason})"
        )

        meta = get_skill_by_name(skills, match.skill_name)
        if not meta:
            report.log(f"[Stage 1 MATCH] 错误: 找不到 {match.skill_name}")
            return report

        loader = ProgressiveSkillLoader(meta)

        # Stage 2
        body = loader.ensure_body()
        report.log(f"[Stage 2 BODY] 加载 {meta.skill_md.name} → {len(body)} 字符")

        # Stage 3（execute 内也会按 skill 类型再加载）
        hints = loader.load_resources_for_query(query)
        if hints:
            report.log(f"[Stage 3 RESOURCE] 预加载: {', '.join(hints)}")
        else:
            report.log("[Stage 3 RESOURCE] 暂无预加载资源（执行阶段可能继续加载）")

        report.loaded_paths = loader.loaded.loaded_paths()
        report.total_loaded_chars = loader.loaded.total_chars

        # Stage 4
        report.log("[Stage 4 EXECUTE] 开始执行…")
        exec_result = execute_skill(
            meta,
            loader,
            query,
            self.output_dir,
            use_llm=use_llm_exec,
        )
        report.exec_result = exec_result
        report.loaded_paths = loader.loaded.loaded_paths()
        report.total_loaded_chars = loader.loaded.total_chars

        for step in exec_result.steps:
            flag = "OK" if step.ok else "FAIL"
            report.log(f"  [{flag}] {step.action}: {step.detail[:200]}")

        report.log(f"[完成] {exec_result.message}")
        return report

    def format_report(self, report: HarnessReport) -> str:
        lines = [
            "=" * 70,
            "渐进式 Skill Harness 运行报告",
            "=" * 70,
            f"Query: {report.query}",
            "",
            "--- Stage 0: 索引（仅元数据）---",
            f"Skills 数量: {len(report.skills_index)}",
            f"索引总字符: {report.index_chars}",
            "",
            build_index_prompt(report.skills_index)[:500] + "\n...(truncated)",
            "",
            "--- Stage 1: 匹配 ---",
        ]
        if report.match:
            lines.append(
                f"选中: {report.match.skill_name} | 方式: {report.match.method} | "
                f"置信度: {report.match.confidence:.2f} | {report.match.reason}"
            )
        else:
            lines.append("无匹配")

        lines.extend([
            "",
            "--- Stage 2–3: 渐进加载 ---",
            f"已加载路径: {report.loaded_paths}",
            f"累计字符: {report.total_loaded_chars}（含索引 {report.index_chars}）",
            "",
            "--- Stage 4: 执行 ---",
        ])
        if report.exec_result:
            for step in report.exec_result.steps:
                lines.append(f"  [{'OK' if step.ok else 'FAIL'}] {step.action}: {step.detail}")
            lines.append(f"结果: {report.exec_result.message}")
            if report.exec_result.outputs:
                lines.append("输出文件:")
                for p in report.exec_result.outputs:
                    lines.append(f"  - {p}")

        lines.extend(["", "--- 完整阶段日志 ---", *report.stage_log, "=" * 70])
        return "\n".join(lines)

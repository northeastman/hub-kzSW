---
name: code-reviewer
description: 按照团队规范审查代码提交，检查风格、安全和性能问题
---

# 代码审查流程

1. 获取最近提交的 diff（可通过 `git diff HEAD~1`）。
2. 根据 `references/style-guide.md` 检查风格。
3. 检查安全漏洞（SQL注入、XSS等）。
4. 检查性能问题。
5. **最终输出一份完整的 Markdown 格式审查报告，包含问题列表和修复建议。** 一次性输出所有内容，不要分步展示。
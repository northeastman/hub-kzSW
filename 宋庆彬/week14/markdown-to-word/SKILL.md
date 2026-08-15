---
name: markdown-to-word
description: 将 Markdown（.md 或 .markdown）文件转换为 Microsoft Word（.docx）文档。用于用户要求把 Markdown、README、笔记或说明文档导出、另存或转换成 Word，并可按需生成目录或套用参考 DOCX 样式。
---

# Markdown 转 Word

使用随技能提供的脚本，通过 Pandoc 生成 DOCX。

## 工作流

1. 确认 Markdown 输入文件。未指定输出路径时，在输入文件旁生成同名 `.docx`。
2. 从本 `SKILL.md` 所在目录运行：

   ```bash
   python3 scripts/markdown_to_word.py INPUT.md [-o OUTPUT.docx] [--toc] [--reference-doc TEMPLATE.docx]
   ```

3. 输出已存在时不要覆盖；只有用户明确要求替换时才添加 `--force`。
4. 检查脚本成功退出，并确认输出文件存在且非空。若排版质量很重要，再渲染 DOCX 做视觉检查。
5. 向用户返回生成文件的绝对路径链接。

## 转换规则

- 保留标题、段落、粗体、斜体、列表、表格、链接、代码块和本地图片等 Pandoc 支持的 Markdown 内容。
- 按 Markdown 文件所在目录解析相对图片路径。
- 需要目录时添加 `--toc`。
- 需要指定 Word 样式时添加 `--reference-doc TEMPLATE.docx`。
- 遇到 Mermaid、交互组件或原始 HTML 等 DOCX 无法直接表达的内容时，先告知用户可能存在降级，不要声称完全保真。
- 不修改原始 Markdown 文件。

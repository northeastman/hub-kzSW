---
name: code-review
description: Review Python code for correctness, security, and maintainability. Use when reviewing pull requests, examining .py changes, or when the user asks for a code review.
---

# Python Code Review

## Workflow

1. Read the full diff before commenting
2. Run automated checks: `python scripts/lint_check.py <file>`
3. Report findings using the template below

## Checklist

- [ ] Logic correct; edge cases (None, empty, boundaries) handled
- [ ] No security issues: SQL/command injection, hardcoded secrets, unsafe pickle
- [ ] Python idioms: no mutable defaults, `is None`, context managers for I/O
- [ ] Readable: clear names, focused functions, docstrings on public API
- [ ] Tests cover new behavior

## Comment Severity

| Level | When |
|-------|------|
| 🔴 Critical | Bug, security flaw, breaking change — must fix |
| 🟡 Suggestion | Refactor, naming, testability — should consider |
| 🟢 Nice to have | Docs, minor style — optional |

## Output Template

```markdown
# Code Review

## Summary
[One paragraph]

## Critical
- [issues or "None"]

## Suggestions
- [items]

## Positive
- [good practices observed]

## Verdict
APPROVE | REQUEST CHANGES | NEEDS DISCUSSION
```

## Reference

Security examples and anti-patterns: [reference.md](reference.md)

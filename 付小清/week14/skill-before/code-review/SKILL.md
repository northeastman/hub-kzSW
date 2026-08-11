---
name: code-review
description: Review Python code for quality, security, and best practices. Use when reviewing pull requests, examining code changes, or when the user asks for a code review of Python files.
---

# Python Code Review Skill

## Introduction

Code review is an essential software engineering practice where developers examine each other's code before it gets merged into the main codebase. The purpose of code review is to catch bugs early, improve code quality, ensure consistency across the team, share knowledge among team members, and maintain high standards for the codebase.

When you are asked to perform a code review, you should approach the task systematically and thoroughly. Do not rush through the review. Take your time to understand what the code is trying to accomplish before judging whether it accomplishes that goal correctly.

This skill applies specifically to **Python** code. Python is a dynamically typed, interpreted programming language known for its readability and extensive standard library. When reviewing Python code, you should be familiar with PEP 8 (the Python style guide), common Python idioms, and the Python standard library.

## When to Use This Skill

You should use this skill when:
- The user asks you to review code
- The user asks you to review a pull request or PR
- The user asks you to check code quality
- The user asks you to find bugs in code
- The user mentions "code review" in their message
- You are examining changes in a Python file (.py extension)

## Pre-Review Checklist

Before starting your review, make sure you have the following information:
- [ ] What is the purpose of the code change?
- [ ] What files were modified?
- [ ] Are there any related tests?
- [ ] What is the expected behavior?
- [ ] Are there any known constraints or requirements?

If any of this information is missing, you should ask the user for clarification before proceeding with the review.

## Step-by-Step Review Process

### Step 1: Read and Understand the Code

First, read through the entire code change from beginning to end. Do not start commenting on individual lines until you have a holistic understanding of what the change does. Ask yourself:
- What problem is this code solving?
- What is the overall architecture or design pattern being used?
- Are there any obvious entry points and exit points?

### Step 2: Check for Correctness

Correctness is the most important aspect of any code review. The code must do what it is supposed to do. Check for:
- Logic errors (off-by-one errors, wrong operators, incorrect conditions)
- Edge cases (empty inputs, null/None values, boundary conditions)
- Error handling (are exceptions caught appropriately?)
- Resource management (are files/connections properly closed?)

For Python specifically, watch out for:
- Mutable default arguments (e.g., `def func(items=[])`)
- Using `==` instead of `is` for None comparisons (prefer `is None`)
- Incorrect indentation (Python is indentation-sensitive)
- Import errors or circular imports

### Step 3: Check for Security Issues

Security vulnerabilities can have serious consequences. Always check for:
- **SQL Injection**: Never concatenate user input directly into SQL queries. Use parameterized queries or an ORM.
- **Command Injection**: Never pass user input to `os.system()`, `subprocess.call()` with `shell=True`, or similar.
- **Path Traversal**: Validate file paths when handling user-provided filenames.
- **Hardcoded Secrets**: Look for API keys, passwords, or tokens in the source code.
- **Insecure Deserialization**: Be careful with `pickle.loads()` on untrusted data.

Example of SQL injection vulnerability:
```python
# BAD - vulnerable to SQL injection
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)

# GOOD - use parameterized queries
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, (user_id,))
```

### Step 4: Check Code Style and Readability

Readable code is maintainable code. Check for:
- Meaningful variable and function names
- Appropriate function length (generally under 50 lines)
- Consistent naming conventions (snake_case for functions/variables in Python)
- Proper docstrings for public functions and classes
- Type hints where appropriate (Python 3.5+)

You can use tools like flake8, pylint, or black to check style automatically:
```bash
flake8 myfile.py
pylint myfile.py
black --check myfile.py
```

### Step 5: Check Performance

While premature optimization is discouraged, obvious performance issues should be flagged:
- Nested loops that could be replaced with better data structures
- Repeated database queries in loops (N+1 problem)
- Loading large files entirely into memory when streaming would work
- Using list comprehensions vs generators appropriately for large datasets

### Step 6: Check Tests

Good code should have tests. Verify:
- Are there unit tests for the new functionality?
- Do existing tests still pass?
- Are edge cases covered?
- Is test coverage adequate?

## Review Comment Format

When providing feedback, use this format to categorize your comments:

**Critical (Must Fix)**: Issues that must be fixed before the code can be merged. These include bugs, security vulnerabilities, and breaking changes.

**Suggestion (Should Consider)**: Improvements that would make the code better but are not blocking. These include refactoring opportunities, better naming, and minor style issues.

**Nice to Have (Optional)**: Minor improvements that are optional. These include additional comments, minor formatting changes, and alternative approaches.

Example review comment:
```
🔴 Critical: Line 42 uses string formatting for SQL query, which is vulnerable to SQL injection. Use parameterized queries instead.

🟡 Suggestion: Consider extracting the validation logic into a separate function for better testability.

🟢 Nice to have: Add a docstring explaining the expected input format.
```

## Common Python Anti-Patterns to Watch For

1. **Bare except clauses**: `except:` catches everything including KeyboardInterrupt. Use `except SpecificException:` instead.

2. **Using print for logging**: Use the `logging` module instead of print statements in production code.

3. **Global variables**: Avoid modifying global state. Pass values as function arguments instead.

4. **Not using context managers**: Always use `with` statement for file operations:
```python
# BAD
f = open('file.txt')
data = f.read()
f.close()

# GOOD
with open('file.txt') as f:
    data = f.read()
```

5. **Catching and ignoring exceptions**:
```python
# BAD
try:
    do_something()
except Exception:
    pass

# GOOD
try:
    do_something()
except SpecificError as e:
    logger.error(f"Failed to do something: {e}")
    raise
```

## Output Template

Structure your review response as follows:

```markdown
# Code Review Report

## Summary
[One paragraph overview of the changes and overall assessment]

## Critical Issues
- [List critical issues, or "None found"]

## Suggestions
- [List suggestions]

## Positive Observations
- [What was done well]

## Verdict
[APPROVE / REQUEST CHANGES / NEEDS DISCUSSION]
```

## Final Reminders

- Be constructive and respectful in your feedback
- Explain WHY something is an issue, not just WHAT is wrong
- Acknowledge good practices when you see them
- If you're unsure about something, say so rather than making assumptions
- Focus on the most important issues first; don't nitpick minor style issues if there are critical bugs

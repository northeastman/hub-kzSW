# Code Review Reference

## Security

```python
# SQL injection — BAD
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")

# GOOD
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
```

- Never pass user input to `os.system()` or `subprocess` with `shell=True`
- Validate file paths from user input
- No hardcoded API keys or passwords
- Avoid `pickle.loads()` on untrusted data

## Common Anti-Patterns

| Pattern | Fix |
|---------|-----|
| `def f(items=[])` | Use `None` default, create list inside |
| `except:` bare | Catch specific exceptions |
| `print()` in prod | Use `logging` module |
| Manual file open/close | Use `with open(...) as f:` |
| Silent `except: pass` | Log and re-raise or handle explicitly |

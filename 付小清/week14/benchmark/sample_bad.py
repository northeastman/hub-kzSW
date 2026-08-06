"""Sample file with intentional issues for lint_check demo."""

def process_items(items=[]):
    try:
        eval("1+1")
    except:
        pass

def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return query

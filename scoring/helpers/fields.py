from typing import Any

def fields_norm(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""

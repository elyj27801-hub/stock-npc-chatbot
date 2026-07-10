import json
import sys
from pathlib import Path


def log_storage_error(context, error):
    """JSON 파일 문제를 터미널에 남깁니다. 민감한 값은 출력하지 않습니다."""
    print(f"[storage-error] {context}: {type(error).__name__}: {error}", file=sys.stderr)


def read_json_file(path, default_value):
    """JSON 파일을 안전하게 읽고, 실패하면 기본값을 반환합니다."""
    json_path = Path(path)
    if not json_path.exists():
        write_json_file(json_path, default_value)
        return default_value, None

    try:
        with json_path.open("r", encoding="utf-8") as json_file:
            return json.load(json_file), None
    except (json.JSONDecodeError, OSError) as error:
        log_storage_error(f"{json_path.name} 읽기 실패", error)
        return default_value, error


def write_json_file(path, data):
    """JSON 파일을 안전하게 저장하고 성공 여부를 반환합니다."""
    json_path = Path(path)
    try:
        with json_path.open("w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=2)
        return True, None
    except OSError as error:
        log_storage_error(f"{json_path.name} 저장 실패", error)
        return False, error

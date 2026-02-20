import json


def load_report(report_path: str) -> dict:
    """
    report.json 파일을 읽어서 dict(파이썬 자료구조)로 반환한다.
    """
    with open(report_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def save_report(report_path: str, report: dict) -> None:
    """
    dict 형태의 report를 report.json에 저장한다.
    """
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
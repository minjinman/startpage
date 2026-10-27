"""autoeda — 데이터를 넣으면 분석 계획을 세우고 실행해 한국어 HTML 리포트를 만듭니다.

    from autoeda import analyze
    report = analyze("data.csv")           # 한 번에
    report.show(); report.to_html("report.html")

    from autoeda import plan_analysis
    plan = plan_analysis("data.csv")       # 계획만 먼저 보고
    plan.drop("outliers"); report = plan.run()
"""
from .planner import Plan, plan_analysis, run


def analyze(data, **kwargs):
    """plan_analysis → run 을 한 번에 수행."""
    return run(plan_analysis(data, **kwargs))


__all__ = ["analyze", "plan_analysis", "run", "Plan"]
__version__ = "0.1.0"

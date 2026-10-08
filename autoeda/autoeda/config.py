"""분석에 쓰는 모든 임계값을 한곳에 모았습니다.

리포트 부록에 이 값들이 그대로 출력되므로, 값을 바꾸면 결과 해석도 함께 달라집니다.
"""
SEED = 0

# --- 정제 -------------------------------------------------------------
MISSING_TOKENS = {
    "", "-", "--", "---", "?", "n/a", "na", "nan", "null", "none", "nil", "#n/a",
    "해당없음", "해당 없음", "결측", "미기재", "무응답",
}
NUMERIC_PARSE_MIN_RATIO = 0.98   # 이 비율 이상이 숫자로 변환되면 숫자 컬럼으로 판정
DATE_PARSE_MIN_RATIO = 0.90      # 이 비율 이상이 날짜로 변환되면 날짜 컬럼으로 판정

# --- 컬럼 타입 --------------------------------------------------------
ID_MIN_N = 50                    # 표본이 이보다 작으면 ID성 판정을 하지 않음
ID_UNIQUE_RATIO = 0.95
TEXT_MEAN_LEN = 30
DISCRETE_MAX_UNIQUE = 10         # 정수형이면서 고유값이 이 이하면 '이산형'으로 표시

# --- 통계 -------------------------------------------------------------
FDR_ALPHA = 0.05
MIN_N_PAIR = 20                  # 상관·연관성 검정에 필요한 최소 쌍 표본 수
MIN_GROUP_N = 5                  # 집단 비교에서 한 집단의 최소 표본 수
MAX_LEVELS = 20                  # 범주형 분석에서 허용하는 최대 수준 수
EFFECT_CORR = 0.30               # |스피어만 ρ| 이상이어야 '의미 있는' 상관
EFFECT_CRAMERS_V = 0.30
EFFECT_EPSILON2 = 0.06           # 크러스칼-월리스 ε² (중간 효과)
REDUNDANT_CORR = 0.95
MISSING_WARN = 0.20
MISSING_DROP = 0.50
SKEW_WARN = 2.0
DOMINANT_LEVEL = 0.90
OUTLIER_WARN_RATIO = 0.05
IQR_K = 1.5

# --- 규모 제한 --------------------------------------------------------
MAX_CORR_COLS = 60
MAX_CAT_COLS = 30
MAX_CHARTS = 12
PLOT_SAMPLE_ROWS = 50_000
TOP_TABLE_ROWS = 15
KEY_FINDINGS = 5
KEY_FINDINGS_PER_STEP = 2
BRIEF_MIN_GROUP = 5              # llm_brief에서 범주값을 노출하는 최소 집단 크기(현재는 범주값 자체를 생략)

THRESHOLD_TABLE = [
    ("FDR 유의수준 (q)", FDR_ALPHA),
    ("상관/연관성 검정 최소 쌍 표본 수", MIN_N_PAIR),
    ("집단 비교 시 집단당 최소 표본 수", MIN_GROUP_N),
    ("범주형 분석 최대 수준 수", MAX_LEVELS),
    ("의미 있는 상관 기준 |ρ|", EFFECT_CORR),
    ("의미 있는 연관성 기준 Cramér's V", EFFECT_CRAMERS_V),
    ("의미 있는 집단 차이 기준 ε²", EFFECT_EPSILON2),
    ("중복 의심 상관 |ρ|", REDUNDANT_CORR),
    ("결측 주의 / 제외 고려 비율", f"{MISSING_WARN:.0%} / {MISSING_DROP:.0%}"),
    ("이상치 기준 (IQR 배수)", IQR_K),
    ("ID성 판정 최소 표본 / 고유값 비율", f"{ID_MIN_N} / {ID_UNIQUE_RATIO:.0%}"),
    ("차트 표시 상한 / 차트용 샘플링 행 수", f"{MAX_CHARTS} / {PLOT_SAMPLE_ROWS:,}"),
    ("난수 시드", SEED),
]

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

# --- 타깃 모델링 -----------------------------------------------------
MIN_N_MODEL = 50                 # 모델링에 필요한 최소 행 수
MODEL_MAX_ROWS = 50_000          # 이보다 많으면 무작위 표본으로 모델링
MAX_FEATURES = 60
CV_FOLDS = 5
PERM_REPEATS = 5
PERM_TEST_ROWS = 2000
LEAK_AUC = 0.98                  # 변수 하나만으로 이 AUC 이상이면 누수 의심
LEAK_R2 = 0.95
IMBALANCE_WARN = 0.10            # 소수 클래스 비율이 이보다 작으면 불균형 경고

# --- 시계열 ----------------------------------------------------------
MIN_N_TS = 30
MAX_TS_SERIES = 6                # 시계열 상세 분석 대상 수치 컬럼 상한
MAX_TS_ENTITIES = 4              # 패널 데이터에서 상세 분석할 개체 수 상한
TS_MAX_MISSING = 0.20            # 격자 결측이 이보다 많으면 분해 생략
SEASON_STRONG = 0.40             # STL 계절성 강도 기준
ANOMALY_Z = 4.0                  # 잔차 z 기준(MAD 척도). 순수 잡음 시뮬레이션으로 오탐 ≈0.1점/720 되도록 보정
CP_PEN_FACTOR = 4.0              # 변화점 탐지 벌점 계수
CP_MAX = 3
SEASON_TIE = 0.05                # 주기 후보 강도 차이가 이 이하면 더 짧은 주기를 선택(과적합 방지)
ANOMALY_MAX_RATIO = 0.10         # 이상 판정 비율이 이보다 크면 '이상구간'이라 부르지 않음
CP_RESID_RHO = 0.80              # 변화점 제거 후에도 잔차 자기상관이 이보다 크면(랜덤워크 성격) 변화점 폐기
STL_MAX_N = 20_000

# --- 설문 -----------------------------------------------------------
SURVEY_MIN_N = 30                # 신뢰도 계산에 필요한 완전 응답 수
STRAIGHTLINE_WARN = 0.05         # 모든 문항에 같은 값으로 응답한 비율 경고 기준
CEILING = 0.60                   # 한 응답 단계가 이 비율 이상이면 천장/바닥 효과
ALPHA_BOOT = 300
ALPHA_LOW = 0.60

# --- 공정 -----------------------------------------------------------
PROC_MIN_N = 20
CAP_GOOD = 1.33
CAP_OK = 1.00
FA_R1 = 0.0027                   # 안정 공정(정규)에서 점 하나가 R1 신호를 낼 확률
FA_OTHER = 0.012                 # R2·R3·R5 신호가 점당 우연히 나올 대략의 확률(시뮬레이션 근사)
STABLE_Q = 0.99                  # 허용 신호 수 = 위 기대 신호 수의 포아송 분위수
MAX_PROC_SERIES = 6
MIN_SUBGROUPS = 8

# --- 규모 제한 --------------------------------------------------------
MAX_CORR_COLS = 60
MAX_CAT_COLS = 30
MAX_CHARTS = 12
PLOT_SAMPLE_ROWS = 50_000
TOP_TABLE_ROWS = 15
KEY_FINDINGS = 5
KEY_FINDINGS_PER_STEP = 3
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
    ("모델링 최소 행 수 / 교차검증 폴드 수", f"{MIN_N_MODEL} / {CV_FOLDS}"),
    ("누수 의심 기준 (변수 1개 단독 AUC / R²)", f"{LEAK_AUC} / {LEAK_R2}"),
    ("시계열 최소 길이 / 계절성 강도 기준 / 이상 z 기준", f"{MIN_N_TS} / {SEASON_STRONG} / {ANOMALY_Z}"),
    ("설문 최소 완전응답 수 / 일률응답 경고 / 천장·바닥 기준", f"{SURVEY_MIN_N} / {STRAIGHTLINE_WARN:.0%} / {CEILING:.0%}"),
    ("안정성 허용 신호 수 기준(안정 공정의 우연 신호 분위수)", STABLE_Q),
    ("공정능력 Cpk 양호/보통 기준", f"{CAP_GOOD} / {CAP_OK}"),
    ("난수 시드", SEED),
]

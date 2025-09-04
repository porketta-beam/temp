# CAN SLIM 한국 주식 스크리닝 시스템

윌리엄 오닐(William J. O'Neil)의 **CAN SLIM** 투자 전략을 한국 주식시장에 적용하는 Python 기반 스크리닝 시스템입니다.

## 📊 프로젝트 소개

이 프로젝트는 pykrx 라이브러리를 활용하여 한국거래소(KRX) 데이터를 수집하고, CAN SLIM의 7가지 핵심 지표를 분석하여 성장주를 발굴하는 자동화 시스템입니다.

### CAN SLIM 지표

- **C** (Current Earnings): 현재 분기 실적 - EPS 25% 이상 성장
- **A** (Annual Earnings): 연간 실적 - 연간 EPS 25% 이상 성장, ROE 17% 이상
- **N** (New): 신제품, 신경영진, 신고가 - 52주 최고가 85% 이상
- **S** (Supply and Demand): 수급 - 적절한 유통주식수와 거래량
- **L** (Leader or Laggard): 업종 선도주 - 상대강도 80 이상
- **I** (Institutional Sponsorship): 기관 투자자 - 기관 매수 증가
- **M** (Market Direction): 시장 방향 - 전체 시장 상승 추세

## 🚀 시작하기

### 필수 요구사항

- Python 3.8 이상
- Jupyter Notebook 또는 JupyterLab

### 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/canslim-korea.git
cd canslim-korea
```

2. 가상환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 필수 패키지 설치
```bash
pip install -r requirements.txt
```

## 📚 튜토리얼 구성

### 기초 학습
1. **00_setup_and_introduction.ipynb**
   - 환경 설정 및 CAN SLIM 전략 소개
   - pykrx 기본 사용법

2. **01_data_collection.ipynb**
   - 주가, 재무, 투자자별 데이터 수집
   - 데이터 저장 및 관리

### CAN SLIM 지표별 분석
3. **02_current_earnings_analysis.ipynb**
   - C 지표: 분기 EPS 성장률 분석
   - 실적 가속화 패턴 감지

4. **03_annual_earnings_analysis.ipynb**
   - A 지표: 연간 EPS 성장률, CAGR, ROE 분석
   - 장기 성장성 평가

5. **04_new_highs_technical.ipynb**
   - N 지표: 52주 신고가 분석
   - 차트 패턴 및 거래량 분석

6. **05_supply_demand_analysis.ipynb**
   - S 지표: 수급 분석
   - 유통주식수 및 거래량 변화

7. **06_leader_laggard_analysis.ipynb**
   - L 지표: 상대강도 분석
   - 업종 내 순위 평가

8. **07_institutional_sponsorship.ipynb**
   - I 지표: 기관/외국인 매매 동향
   - 스마트머니 추적

9. **08_market_direction.ipynb**
   - M 지표: 시장 방향성 분석
   - KOSPI/KOSDAQ 추세 판단

### 통합 시스템
10. **09_complete_screening.ipynb**
    - 모든 지표 통합 스크리닝
    - 종합 점수 산출 및 포트폴리오 구성
    - 백테스팅 및 성과 분석

## 💻 사용 예시

### 기본 스크리닝
```python
from canslim import CANSLIMAnalyzer

# 종목 분석
analyzer = CANSLIMAnalyzer('005930', start_date, end_date)
result = analyzer.analyze()

# 종합 점수 확인
print(f"Total Score: {result['Total_Score']}")
```

### 대량 스크리닝
```python
# KOSPI 상위 종목 스크리닝
tickers = get_kospi_top_stocks(100)
results = screen_stocks(tickers, start_date, end_date)

# 상위 종목 선별
top_stocks = results[results['Total_Score'] >= 70]
```

## 📈 주요 기능

- ✅ **자동 데이터 수집**: pykrx를 통한 실시간 데이터
- ✅ **지표 계산**: 모든 CAN SLIM 지표 자동 계산
- ✅ **점수 산출**: 가중치 기반 종합 점수
- ✅ **포트폴리오 구성**: 리스크 분산 고려
- ✅ **시각화**: 대시보드 및 차트
- ✅ **백테스팅**: 과거 성과 검증

## 📊 데이터 소스

- **주 데이터**: pykrx (한국거래소)
- **보조 데이터**: yfinance (선택사항)
- **업데이트 주기**: 일별 (장 마감 후)

## ⚠️ 주의사항

1. **투자 책임**: 이 시스템은 교육 목적이며, 실제 투자 결정은 본인 책임
2. **데이터 지연**: pykrx 데이터는 실시간이 아닌 지연 데이터
3. **API 제한**: 과도한 요청 시 제한될 수 있음
4. **시장 특성**: 한국 시장 특성상 미국 시장과 다른 결과 가능

## 🔧 커스터마이징

### 기준 조정
```python
# config.py에서 기준 수정
CANSLIM_CRITERIA = {
    'C': {'quarterly_eps_growth': 0.25},  # 25% → 조정 가능
    'A': {'roe_threshold': 0.17},         # 17% → 조정 가능
    # ...
}
```

### 가중치 변경
```python
# 종합 점수 가중치 조정
weights = {
    'C': 0.20,  # 현재 실적 비중
    'A': 0.15,  # 연간 실적 비중
    # ...
}
```

## 📝 추가 개발 계획

- [ ] 실시간 알림 시스템
- [ ] 웹 대시보드 구현
- [ ] 자동 매매 연동
- [ ] 머신러닝 예측 모델
- [ ] 리스크 관리 강화
- [ ] 섹터 로테이션 전략

## 🤝 기여하기

기여는 언제나 환영합니다! 다음 방법으로 참여할 수 있습니다:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📧 연락처

질문이나 제안사항이 있으시면 이슈를 등록해주세요.

## 🙏 감사의 글

- William J. O'Neil - CAN SLIM 전략 창시자
- pykrx 개발팀 - 한국 주식 데이터 API
- 모든 오픈소스 기여자들

---

**면책조항**: 이 프로젝트는 교육 목적으로 제작되었으며, 투자 권유가 아닙니다. 모든 투자 결정과 그 결과에 대한 책임은 투자자 본인에게 있습니다.
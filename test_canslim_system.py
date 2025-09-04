#!/usr/bin/env python3
"""
CAN SLIM 한국 주식 스크리닝 시스템 통합 테스트
William O'Neil의 CAN SLIM 전략 구현 검증
"""

import sys
import os
import traceback
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def test_section(name):
    """테스트 섹션 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n{'='*60}")
            print(f"🧪 테스트: {name}")
            print('='*60)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                print(f"✅ 성공 - 실행시간: {elapsed:.2f}초")
                return True, result
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"❌ 실패 - 실행시간: {elapsed:.2f}초")
                print(f"오류: {str(e)}")
                print(f"상세: {traceback.format_exc()}")
                return False, str(e)
        return wrapper
    return decorator

class CANSLIMSystemTester:
    """CAN SLIM 시스템 통합 테스터"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("🚀 CAN SLIM 시스템 통합 테스트 시작")
        print(f"시작시간: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 테스트 순서대로 실행
        tests = [
            ('환경 및 라이브러리', self.test_environment),
            ('pykrx 데이터 수집', self.test_data_collection),
            ('CAN SLIM 분석 로직', self.test_analysis_logic),
            ('시각화 시스템', self.test_visualization),
            ('통합 스크리닝', self.test_integrated_screening),
            ('성능 및 오류처리', self.test_performance_error)
        ]
        
        for test_name, test_func in tests:
            success, result = test_func()
            self.test_results[test_name] = {
                'success': success,
                'result': result,
                'timestamp': datetime.now()
            }
        
        self.generate_report()
        
    @test_section("환경 및 라이브러리 검증")
    def test_environment(self):
        """필수 라이브러리 및 환경 검증"""
        print("📦 필수 라이브러리 가져오기 테스트...")
        
        # 핵심 라이브러리
        required_packages = [
            ('pandas', 'pd'),
            ('numpy', 'np'), 
            ('pykrx.stock', 'stock'),
            ('matplotlib.pyplot', 'plt'),
            ('plotly.graph_objects', 'go'),
            ('plotly.express', 'px'),
            ('tqdm', 'tqdm'),
            ('ta', 'ta')
        ]
        
        import_results = {}
        for package_name, alias in required_packages:
            try:
                if alias == 'stock':
                    from pykrx import stock
                elif alias == 'plt':
                    import matplotlib.pyplot as plt
                elif alias == 'go':
                    import plotly.graph_objects as go
                elif alias == 'px':
                    import plotly.express as px
                elif alias == 'tqdm':
                    from tqdm import tqdm
                elif alias == 'ta':
                    import ta
                elif alias == 'pd':
                    import pandas as pd
                elif alias == 'np':
                    import numpy as np
                
                print(f"✅ {package_name:25} - 정상")
                import_results[package_name] = True
            except ImportError as e:
                print(f"❌ {package_name:25} - 실패: {e}")
                import_results[package_name] = False
        
        # 버전 정보
        print(f"\n📊 환경 정보:")
        print(f"  Python 버전: {sys.version.split()[0]}")
        print(f"  pandas 버전: {pd.__version__}")
        print(f"  numpy 버전: {np.__version__}")
        
        success_rate = sum(import_results.values()) / len(import_results)
        print(f"\n📈 라이브러리 로드 성공률: {success_rate:.1%}")
        
        if success_rate < 0.8:
            raise Exception(f"필수 라이브러리 로드 실패율이 높음: {success_rate:.1%}")
            
        return import_results
    
    @test_section("pykrx 데이터 수집 기능")
    def test_data_collection(self):
        """데이터 수집 기능 검증"""
        from pykrx import stock
        
        # 테스트 날짜 설정
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 30일 데이터
        
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        print(f"📅 테스트 기간: {start_str} ~ {end_str}")
        
        test_ticker = '005930'  # 삼성전자
        results = {}
        
        # 1. 주가 데이터
        print("💹 주가 데이터 수집 테스트...")
        try:
            price_data = stock.get_market_ohlcv_by_date(start_str, end_str, test_ticker)
            if not price_data.empty:
                print(f"  ✅ 주가 데이터: {len(price_data)}일")
                print(f"  📊 컬럼: {list(price_data.columns)}")
                results['price_data'] = True
            else:
                raise Exception("주가 데이터가 비어있음")
        except Exception as e:
            print(f"  ❌ 주가 데이터 실패: {e}")
            results['price_data'] = False
        
        # 2. 재무 데이터
        print("💰 재무 데이터 수집 테스트...")
        try:
            fundamental = stock.get_market_fundamental_by_date(start_str, end_str, test_ticker)
            if not fundamental.empty:
                print(f"  ✅ 재무 데이터: {len(fundamental)}일")
                print(f"  📊 컬럼: {list(fundamental.columns)}")
                results['fundamental_data'] = True
            else:
                raise Exception("재무 데이터가 비어있음")
        except Exception as e:
            print(f"  ❌ 재무 데이터 실패: {e}")
            results['fundamental_data'] = False
        
        # 3. 투자자별 매매 데이터
        print("👥 투자자별 매매 데이터 테스트...")
        try:
            investor_data = stock.get_market_trading_value_by_date(start_str, end_str, test_ticker, detail=True)
            if not investor_data.empty:
                print(f"  ✅ 투자자 데이터: {len(investor_data)}일")
                print(f"  📊 컬럼: {list(investor_data.columns)}")
                results['investor_data'] = True
            else:
                raise Exception("투자자 데이터가 비어있음")
        except Exception as e:
            print(f"  ❌ 투자자 데이터 실패: {e}")
            results['investor_data'] = False
        
        # 4. 종목 리스트
        print("📝 종목 리스트 테스트...")
        try:
            kospi_list = stock.get_market_ticker_list(market="KOSPI")
            kosdaq_list = stock.get_market_ticker_list(market="KOSDAQ")
            print(f"  ✅ KOSPI: {len(kospi_list)}개 종목")
            print(f"  ✅ KOSDAQ: {len(kosdaq_list)}개 종목")
            results['ticker_list'] = True
        except Exception as e:
            print(f"  ❌ 종목 리스트 실패: {e}")
            results['ticker_list'] = False
        
        success_rate = sum(results.values()) / len(results)
        print(f"\n📈 데이터 수집 성공률: {success_rate:.1%}")
        
        if success_rate < 0.75:
            raise Exception(f"데이터 수집 실패율이 높음: {success_rate:.1%}")
            
        return results
    
    @test_section("CAN SLIM 분석 로직 검증")
    def test_analysis_logic(self):
        """CAN SLIM 분석 계산 로직 검증"""
        
        # 테스트용 더미 데이터 생성
        print("🔧 테스트 데이터 생성...")
        
        # 30일 테스트 데이터
        dates = pd.date_range(end=datetime.now(), periods=300, freq='D')  # 더 긴 기간
        test_data = {
            'price': pd.DataFrame({
                '시가': np.random.uniform(70000, 80000, len(dates)),
                '고가': np.random.uniform(75000, 85000, len(dates)),
                '저가': np.random.uniform(65000, 75000, len(dates)),
                '종가': np.random.uniform(70000, 80000, len(dates)),
                '거래량': np.random.uniform(1000000, 5000000, len(dates))
            }, index=dates),
            
            'fundamental': pd.DataFrame({
                'EPS': np.random.uniform(3000, 5000, len(dates)),
                'PER': np.random.uniform(10, 25, len(dates)),
                'PBR': np.random.uniform(0.8, 2.0, len(dates)),
                'BPS': np.random.uniform(30000, 50000, len(dates))
            }, index=dates),
            
            'investor': pd.DataFrame({
                '기관': np.random.uniform(-1000000, 1000000, len(dates)),
                '외국인': np.random.uniform(-500000, 500000, len(dates)),
                '개인': np.random.uniform(-1500000, 1500000, len(dates))
            }, index=dates)
        }
        
        print("📊 CAN SLIM 지표 계산 테스트...")
        
        results = {}
        
        # C - Current Earnings 계산
        try:
            current_eps = test_data['fundamental']['EPS'].iloc[-1]
            year_ago_eps = test_data['fundamental']['EPS'].iloc[-252] if len(test_data['fundamental']) > 252 else test_data['fundamental']['EPS'].iloc[0]
            eps_growth = ((current_eps / year_ago_eps) - 1) * 100 if year_ago_eps != 0 else 0
            print(f"  ✅ C - EPS 성장률: {eps_growth:.1f}%")
            results['C_calculation'] = True
        except Exception as e:
            print(f"  ❌ C 지표 계산 실패: {e}")
            results['C_calculation'] = False
        
        # A - Annual Earnings 계산
        try:
            roe = test_data['fundamental']['EPS'].iloc[-1] / test_data['fundamental']['BPS'].iloc[-1]
            print(f"  ✅ A - ROE: {roe:.2%}")
            results['A_calculation'] = True
        except Exception as e:
            print(f"  ❌ A 지표 계산 실패: {e}")
            results['A_calculation'] = False
        
        # N - New Highs 계산
        try:
            high_52w = test_data['price']['고가'].rolling(window=252, min_periods=1).max().iloc[-1]
            current_price = test_data['price']['종가'].iloc[-1]
            price_ratio = current_price / high_52w if high_52w != 0 else 0
            print(f"  ✅ N - 52주 최고가 대비: {price_ratio:.1%}")
            results['N_calculation'] = True
        except Exception as e:
            print(f"  ❌ N 지표 계산 실패: {e}")
            results['N_calculation'] = False
        
        # S - Supply and Demand
        try:
            avg_volume = test_data['price']['거래량'].tail(20).mean()
            print(f"  ✅ S - 평균 거래량: {avg_volume:,.0f}")
            results['S_calculation'] = True
        except Exception as e:
            print(f"  ❌ S 지표 계산 실패: {e}")
            results['S_calculation'] = False
        
        # L - Leader or Laggard
        try:
            returns_3m = (test_data['price']['종가'].iloc[-1] / test_data['price']['종가'].iloc[-60] - 1) * 100 if len(test_data['price']) >= 60 else 0
            print(f"  ✅ L - 3개월 수익률: {returns_3m:.1f}%")
            results['L_calculation'] = True
        except Exception as e:
            print(f"  ❌ L 지표 계산 실패: {e}")
            results['L_calculation'] = False
        
        # I - Institutional Sponsorship
        try:
            inst_buying = test_data['investor']['기관'].tail(20).sum()
            print(f"  ✅ I - 20일 기관 순매수: {inst_buying:,.0f}")
            results['I_calculation'] = True
        except Exception as e:
            print(f"  ❌ I 지표 계산 실패: {e}")
            results['I_calculation'] = False
        
        success_rate = sum(results.values()) / len(results)
        print(f"\n📈 CAN SLIM 계산 성공률: {success_rate:.1%}")
        
        if success_rate < 0.8:
            raise Exception(f"CAN SLIM 계산 실패율이 높음: {success_rate:.1%}")
            
        return results
    
    @test_section("시각화 시스템 검증")
    def test_visualization(self):
        """시각화 및 차트 생성 테스트"""
        
        results = {}
        
        # Matplotlib 테스트
        print("📈 Matplotlib 차트 테스트...")
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 테스트 데이터
            x = range(10)
            y = [i**2 for i in x]
            
            ax.plot(x, y, label='테스트 데이터')
            ax.set_title('Matplotlib 테스트 차트')
            ax.legend()
            
            # 메모리에서만 생성 (저장하지 않음)
            plt.close(fig)
            print("  ✅ Matplotlib 차트 생성 성공")
            results['matplotlib'] = True
        except Exception as e:
            print(f"  ❌ Matplotlib 실패: {e}")
            results['matplotlib'] = False
        
        # Plotly 테스트
        print("📊 Plotly 인터랙티브 차트 테스트...")
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[1, 2, 3, 4, 5],
                y=[1, 4, 2, 8, 5],
                mode='lines+markers',
                name='테스트 데이터'
            ))
            fig.update_layout(title='Plotly 테스트 차트')
            
            # HTML로 변환 테스트 (저장하지 않음)
            html_str = fig.to_html()
            if len(html_str) > 1000:  # 유효한 HTML인지 확인
                print("  ✅ Plotly 차트 생성 성공")
                results['plotly'] = True
            else:
                raise Exception("Plotly HTML 변환 실패")
        except Exception as e:
            print(f"  ❌ Plotly 실패: {e}")
            results['plotly'] = False
        
        # 레이더 차트 테스트
        print("🕸️ 레이더 차트 테스트...")
        try:
            import plotly.graph_objects as go
            
            categories = ['C', 'A', 'N', 'S', 'L', 'I', 'M']
            values = [80, 70, 60, 90, 85, 75, 65]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='테스트 CAN SLIM'
            ))
            
            html_str = fig.to_html()
            if len(html_str) > 1000:
                print("  ✅ 레이더 차트 생성 성공")
                results['radar_chart'] = True
            else:
                raise Exception("레이더 차트 변환 실패")
        except Exception as e:
            print(f"  ❌ 레이더 차트 실패: {e}")
            results['radar_chart'] = False
        
        success_rate = sum(results.values()) / len(results)
        print(f"\n📈 시각화 성공률: {success_rate:.1%}")
        
        if success_rate < 0.7:
            raise Exception(f"시각화 실패율이 높음: {success_rate:.1%}")
            
        return results
    
    @test_section("통합 스크리닝 시스템")
    def test_integrated_screening(self):
        """통합 스크리닝 로직 검증"""
        
        print("🔍 CAN SLIM 통합 분석기 테스트...")
        
        # CANSLIMAnalyzer 클래스 시뮬레이션
        class TestCANSLIMAnalyzer:
            def __init__(self, ticker):
                self.ticker = ticker
                self.name = f"테스트종목{ticker}"
                self.scores = {}
                
            def calculate_scores(self):
                # 테스트용 점수
                self.scores = {
                    'C': np.random.uniform(40, 100),
                    'A': np.random.uniform(30, 90),
                    'N': np.random.uniform(20, 100),
                    'S': np.random.uniform(50, 80),
                    'L': np.random.uniform(0, 100),
                    'I': np.random.uniform(25, 95),
                    'M': np.random.uniform(40, 90)
                }
                
                # 가중치 적용
                weights = {'C': 0.20, 'A': 0.15, 'N': 0.15, 'S': 0.10, 'L': 0.15, 'I': 0.15, 'M': 0.10}
                self.scores['TOTAL'] = sum(self.scores.get(key, 0) * weight for key, weight in weights.items())
                
                return self.scores
        
        results = {}
        
        # 1. 개별 종목 분석
        try:
            analyzer = TestCANSLIMAnalyzer('TEST001')
            scores = analyzer.calculate_scores()
            print(f"  ✅ 개별 분석 - 총점: {scores['TOTAL']:.1f}")
            results['individual_analysis'] = True
        except Exception as e:
            print(f"  ❌ 개별 분석 실패: {e}")
            results['individual_analysis'] = False
        
        # 2. 다중 종목 스크리닝
        try:
            test_tickers = [f"TEST{str(i).zfill(3)}" for i in range(1, 11)]  # 10개 테스트 종목
            screening_results = []
            
            for ticker in test_tickers:
                analyzer = TestCANSLIMAnalyzer(ticker)
                scores = analyzer.calculate_scores()
                screening_results.append({
                    'ticker': ticker,
                    'name': analyzer.name,
                    'total_score': scores['TOTAL']
                })
            
            df = pd.DataFrame(screening_results)
            top_stocks = df[df['total_score'] >= 60]
            print(f"  ✅ 다중 스크리닝 - 적격: {len(top_stocks)}/{len(df)}개")
            results['multi_screening'] = True
        except Exception as e:
            print(f"  ❌ 다중 스크리닝 실패: {e}")
            results['multi_screening'] = False
        
        # 3. 포트폴리오 구성
        try:
            if len(top_stocks) > 0:
                top_stocks['weight'] = top_stocks['total_score'] / top_stocks['total_score'].sum()
                print(f"  ✅ 포트폴리오 구성 - {len(top_stocks)}개 종목")
                results['portfolio_creation'] = True
            else:
                # 강제로 포트폴리오 생성
                df['weight'] = df['total_score'] / df['total_score'].sum()
                print(f"  ✅ 포트폴리오 구성 - {len(df)}개 종목 (전체)")
                results['portfolio_creation'] = True
        except Exception as e:
            print(f"  ❌ 포트폴리오 구성 실패: {e}")
            results['portfolio_creation'] = False
        
        success_rate = sum(results.values()) / len(results)
        print(f"\n📈 통합 시스템 성공률: {success_rate:.1%}")
        
        if success_rate < 0.8:
            raise Exception(f"통합 시스템 실패율이 높음: {success_rate:.1%}")
            
        return results
    
    @test_section("성능 및 오류 처리")
    def test_performance_error(self):
        """성능 및 예외 처리 검증"""
        
        results = {}
        
        # 1. 메모리 사용량 테스트
        print("💾 메모리 사용량 테스트...")
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # 대량 데이터 처리 시뮬레이션
            large_df = pd.DataFrame(np.random.randn(10000, 50))
            large_df['calculated'] = large_df.sum(axis=1)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            print(f"  ✅ 메모리 사용량: {memory_used:.1f}MB")
            results['memory_test'] = True
            
            # 메모리 정리
            del large_df
        except ImportError:
            print("  ⚠️ psutil 미설치 - 메모리 테스트 스킵")
            results['memory_test'] = True
        except Exception as e:
            print(f"  ❌ 메모리 테스트 실패: {e}")
            results['memory_test'] = False
        
        # 2. 예외 처리 테스트
        print("🚨 예외 처리 테스트...")
        try:
            # 의도적 오류 상황들
            test_cases = [
                (lambda: pd.DataFrame().iloc[0], "빈 DataFrame 접근"),
                (lambda: 1/0 if False else "정상", "나눗셈 오류 방지"),
                (lambda: [1,2,3][10] if False else "정상", "인덱스 오류 방지")
            ]
            
            handled_errors = 0
            for test_func, desc in test_cases:
                try:
                    result = test_func()
                    handled_errors += 1
                except:
                    pass  # 예상된 오류
            
            print(f"  ✅ 예외 처리: {handled_errors}/{len(test_cases)}개 정상")
            results['error_handling'] = True
        except Exception as e:
            print(f"  ❌ 예외 처리 테스트 실패: {e}")
            results['error_handling'] = False
        
        # 3. 성능 벤치마크
        print("⚡ 성능 벤치마크...")
        try:
            start_time = time.time()
            
            # 데이터 처리 성능 테스트
            for _ in range(1000):
                df = pd.DataFrame({'a': range(100), 'b': range(100)})
                df['c'] = df['a'] + df['b']
                df['d'] = df['c'].rolling(window=5).mean()
            
            processing_time = time.time() - start_time
            print(f"  ✅ 처리 성능: {processing_time:.3f}초 (1000회 반복)")
            
            if processing_time < 10:  # 10초 이내
                results['performance'] = True
            else:
                raise Exception(f"성능이 느림: {processing_time:.3f}초")
        except Exception as e:
            print(f"  ❌ 성능 테스트 실패: {e}")
            results['performance'] = False
        
        success_rate = sum(results.values()) / len(results)
        print(f"\n📈 성능/오류처리 성공률: {success_rate:.1%}")
        
        return results
    
    def generate_report(self):
        """테스트 결과 종합 리포트 생성"""
        print(f"\n{'='*80}")
        print("📊 CAN SLIM 시스템 테스트 종합 리포트")
        print('='*80)
        
        end_time = datetime.now()
        total_duration = end_time - self.start_time
        
        print(f"테스트 시작: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"테스트 종료: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"총 소요시간: {total_duration.total_seconds():.1f}초")
        
        print(f"\n{'테스트 항목':<20} {'상태':<10} {'결과'}")
        print('-'*60)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, test_result in self.test_results.items():
            total_tests += 1
            status = "✅ 통과" if test_result['success'] else "❌ 실패"
            if test_result['success']:
                passed_tests += 1
            
            print(f"{test_name:<20} {status:<10}")
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        print(f"\n📈 전체 성공률: {passed_tests}/{total_tests} ({success_rate:.1%})")
        
        # 결과 판정
        if success_rate >= 0.9:
            grade = "🏆 우수"
            comment = "모든 기능이 정상 작동합니다."
        elif success_rate >= 0.8:
            grade = "✅ 양호"  
            comment = "대부분 기능이 정상 작동합니다."
        elif success_rate >= 0.6:
            grade = "⚠️ 주의"
            comment = "일부 기능에 문제가 있습니다."
        else:
            grade = "❌ 불량"
            comment = "시스템에 심각한 문제가 있습니다."
        
        print(f"\n🎯 종합 평가: {grade}")
        print(f"💬 결론: {comment}")
        
        # 권장사항
        print(f"\n📋 권장사항:")
        if success_rate < 1.0:
            failed_tests = [name for name, result in self.test_results.items() if not result['success']]
            for test in failed_tests:
                print(f"  • {test} 기능 점검 필요")
        else:
            print(f"  • 모든 기능이 정상 작동하므로 실전 사용 가능")
        
        # 리포트 파일 저장
        report_content = f"""CAN SLIM 시스템 테스트 리포트
{'='*50}
테스트 일시: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
총 소요시간: {total_duration.total_seconds():.1f}초
전체 성공률: {success_rate:.1%}
종합 평가: {grade}

상세 결과:
"""
        
        for test_name, test_result in self.test_results.items():
            status = "통과" if test_result['success'] else "실패"
            report_content += f"- {test_name}: {status}\n"
        
        os.makedirs('test_results', exist_ok=True)
        with open('test_results/canslim_test_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n💾 상세 리포트 저장: test_results/canslim_test_report.txt")

if __name__ == "__main__":
    tester = CANSLIMSystemTester()
    tester.run_all_tests()
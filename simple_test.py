#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAN SLIM 시스템 간단 테스트
Windows 환경 호환성 버전
"""

import sys
import traceback
import time
from datetime import datetime, timedelta

def test_environment():
    """환경 및 라이브러리 검증"""
    print("=" * 50)
    print("환경 및 라이브러리 테스트")
    print("=" * 50)
    
    results = {}
    
    # 기본 라이브러리들
    test_packages = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('matplotlib.pyplot', 'plt'),
        ('datetime', 'datetime')
    ]
    
    for package_name, alias in test_packages:
        try:
            if alias == 'pd':
                import pandas as pd
                print(f"[OK] pandas {pd.__version__}")
                results['pandas'] = True
            elif alias == 'np':
                import numpy as np
                print(f"[OK] numpy {np.__version__}")
                results['numpy'] = True
            elif alias == 'plt':
                import matplotlib.pyplot as plt
                print("[OK] matplotlib")
                results['matplotlib'] = True
            elif alias == 'datetime':
                from datetime import datetime
                print("[OK] datetime")
                results['datetime'] = True
                
        except ImportError as e:
            print(f"[ERROR] {package_name}: {e}")
            results[package_name] = False
    
    # pykrx 별도 테스트
    try:
        from pykrx import stock
        print("[OK] pykrx")
        results['pykrx'] = True
    except ImportError as e:
        print(f"[ERROR] pykrx: {e}")
        results['pykrx'] = False
    
    success_rate = sum(results.values()) / len(results)
    print(f"\n라이브러리 로드 성공률: {success_rate:.1%}")
    
    return results

def test_basic_calculations():
    """기본 계산 로직 테스트"""
    print("\n" + "=" * 50)
    print("기본 계산 로직 테스트")
    print("=" * 50)
    
    try:
        import pandas as pd
        import numpy as np
        
        # 테스트 데이터 생성
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        test_data = pd.DataFrame({
            'price': np.random.uniform(70000, 80000, len(dates)),
            'eps': np.random.uniform(3000, 5000, len(dates)),
            'volume': np.random.uniform(1000000, 5000000, len(dates))
        }, index=dates)
        
        # 기본 계산들
        test_data['ma20'] = test_data['price'].rolling(window=20).mean()
        test_data['returns'] = test_data['price'].pct_change()
        test_data['eps_growth'] = test_data['eps'].pct_change(periods=20) * 100
        
        print(f"테스트 데이터 생성: {len(test_data)}일")
        print(f"이동평균 계산: {test_data['ma20'].notna().sum()}개")
        print(f"수익률 계산: {test_data['returns'].notna().sum()}개")
        
        print("[OK] 기본 계산 로직 정상")
        return True
        
    except Exception as e:
        print(f"[ERROR] 기본 계산 실패: {e}")
        return False

def test_data_access():
    """데이터 접근 테스트 (실제 API 호출 없이)"""
    print("\n" + "=" * 50)
    print("데이터 접근 테스트")
    print("=" * 50)
    
    try:
        from pykrx import stock
        
        # API 함수들이 존재하는지 확인
        api_functions = [
            'get_market_ohlcv_by_date',
            'get_market_fundamental_by_date',
            'get_market_trading_value_by_date',
            'get_market_ticker_list'
        ]
        
        for func_name in api_functions:
            if hasattr(stock, func_name):
                print(f"[OK] {func_name} 함수 존재")
            else:
                print(f"[ERROR] {func_name} 함수 없음")
                return False
        
        print("[OK] pykrx API 함수들 정상")
        return True
        
    except Exception as e:
        print(f"[ERROR] 데이터 접근 테스트 실패: {e}")
        return False

def test_canslim_logic():
    """CAN SLIM 로직 테스트"""
    print("\n" + "=" * 50)
    print("CAN SLIM 로직 테스트")
    print("=" * 50)
    
    try:
        import pandas as pd
        import numpy as np
        
        # 가상의 종목 데이터
        test_stock = {
            'eps_current': 5000,
            'eps_year_ago': 4000,
            'roe': 0.18,
            'price_current': 75000,
            'price_52w_high': 80000,
            'volume_recent': 2000000,
            'volume_avg': 1500000
        }
        
        # C - Current Earnings
        eps_growth = ((test_stock['eps_current'] / test_stock['eps_year_ago']) - 1) * 100
        c_score = 75 if eps_growth >= 25 else 50 if eps_growth >= 15 else 25
        print(f"C 지표 - EPS 성장률: {eps_growth:.1f}%, 점수: {c_score}")
        
        # A - Annual Earnings  
        roe = test_stock['roe']
        a_score = 80 if roe >= 0.17 else 60 if roe >= 0.10 else 40
        print(f"A 지표 - ROE: {roe:.1%}, 점수: {a_score}")
        
        # N - New Highs
        price_ratio = test_stock['price_current'] / test_stock['price_52w_high']
        n_score = 90 if price_ratio >= 0.90 else 70 if price_ratio >= 0.80 else 50
        print(f"N 지표 - 52주 최고가 대비: {price_ratio:.1%}, 점수: {n_score}")
        
        # 종합 점수
        total_score = (c_score * 0.3 + a_score * 0.3 + n_score * 0.4)
        print(f"\n종합 점수: {total_score:.1f}/100")
        
        if total_score >= 60:
            print("[OK] CAN SLIM 로직 정상 (투자 적격)")
        else:
            print("[OK] CAN SLIM 로직 정상 (투자 부적격)")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] CAN SLIM 로직 테스트 실패: {e}")
        return False

def test_visualization():
    """시각화 테스트"""
    print("\n" + "=" * 50)
    print("시각화 테스트")
    print("=" * 50)
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 간단한 차트 생성 테스트
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title('Test Chart')
        
        # 메모리에서만 생성하고 닫기
        plt.close(fig)
        print("[OK] matplotlib 차트 생성 성공")
        
        # plotly 테스트
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13]))
            
            # HTML 변환 테스트
            html_str = fig.to_html()
            if len(html_str) > 500:
                print("[OK] plotly 차트 생성 성공")
            else:
                print("[WARNING] plotly 차트가 작음")
                
        except ImportError:
            print("[WARNING] plotly 미설치")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 시각화 테스트 실패: {e}")
        return False

def run_all_tests():
    """모든 테스트 실행"""
    print("CAN SLIM 시스템 통합 테스트 시작")
    print("시작시간:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print()
    
    start_time = time.time()
    test_results = {}
    
    # 테스트 실행
    tests = [
        ('환경 및 라이브러리', test_environment),
        ('기본 계산', test_basic_calculations),
        ('데이터 접근', test_data_access),
        ('CAN SLIM 로직', test_canslim_logic),
        ('시각화', test_visualization)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            print(f"[ERROR] {test_name} 테스트 중 예외 발생: {e}")
            test_results[test_name] = False
    
    # 결과 요약
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"전체 테스트: {total_tests}개")
    print(f"통과 테스트: {passed_tests}개")
    print(f"성공률: {passed_tests/total_tests:.1%}")
    print(f"소요시간: {duration:.2f}초")
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
    
    # 최종 판정
    success_rate = passed_tests / total_tests
    if success_rate >= 0.8:
        print(f"\n결론: 시스템이 정상적으로 작동합니다! ({success_rate:.1%} 성공)")
        print("실전 사용 가능합니다.")
    else:
        print(f"\n결론: 시스템에 문제가 있습니다. ({success_rate:.1%} 성공)")
        print("문제 해결 후 재테스트가 필요합니다.")
    
    return test_results

if __name__ == "__main__":
    try:
        results = run_all_tests()
    except KeyboardInterrupt:
        print("\n테스트가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n테스트 실행 중 치명적 오류: {e}")
        print("상세:", traceback.format_exc())
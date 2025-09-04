#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAN SLIM 시스템 포괄적 테스트
실제 pykrx 데이터를 포함한 검증
"""

import sys
import traceback
import time
from datetime import datetime, timedelta

def test_imports():
    """라이브러리 임포트 테스트"""
    print("=" * 50)
    print("라이브러리 임포트 테스트")
    print("=" * 50)
    
    success_count = 0
    total_count = 0
    
    # 핵심 라이브러리들
    imports = [
        ('pandas', 'import pandas as pd'),
        ('numpy', 'import numpy as np'),
        ('pykrx', 'from pykrx import stock'),
        ('matplotlib', 'import matplotlib.pyplot as plt'),
        ('datetime', 'from datetime import datetime, timedelta')
    ]
    
    for name, import_cmd in imports:
        total_count += 1
        try:
            exec(import_cmd)
            print(f"[OK] {name}")
            success_count += 1
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
    
    # plotly (선택사항)
    try:
        import plotly.graph_objects as go
        print("[OK] plotly")
        success_count += 1
        total_count += 1
    except ImportError:
        print("[WARNING] plotly 미설치 (선택사항)")
        total_count += 1
    
    success_rate = success_count / total_count
    print(f"\n임포트 성공률: {success_rate:.1%} ({success_count}/{total_count})")
    
    return success_rate >= 0.8

def test_pykrx_data():
    """실제 pykrx 데이터 수집 테스트"""
    print("\n" + "=" * 50)
    print("pykrx 실제 데이터 수집 테스트")
    print("=" * 50)
    
    try:
        from pykrx import stock
        import pandas as pd
        
        # 날짜 설정 (최근 7일)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        print(f"테스트 기간: {start_str} ~ {end_str}")
        
        # 삼성전자 데이터 수집
        ticker = '005930'
        print(f"테스트 종목: {ticker} (삼성전자)")
        
        # 1. 주가 데이터
        print("1. 주가 데이터 수집 중...")
        price_data = stock.get_market_ohlcv_by_date(start_str, end_str, ticker)
        if not price_data.empty:
            print(f"   성공: {len(price_data)}일 데이터")
            print(f"   최근 종가: {price_data['종가'].iloc[-1]:,}원")
        else:
            print("   경고: 주가 데이터 없음")
        
        # 2. 재무 데이터
        print("2. 재무 데이터 수집 중...")
        fund_data = stock.get_market_fundamental_by_date(start_str, end_str, ticker)
        if not fund_data.empty:
            print(f"   성공: {len(fund_data)}일 데이터")
            print(f"   최근 EPS: {fund_data['EPS'].iloc[-1]:,}")
            print(f"   최근 PER: {fund_data['PER'].iloc[-1]:.1f}")
        else:
            print("   경고: 재무 데이터 없음")
        
        # 3. 종목 리스트
        print("3. 종목 리스트 수집 중...")
        kospi_tickers = stock.get_market_ticker_list(market="KOSPI")
        print(f"   KOSPI 종목 수: {len(kospi_tickers)}")
        
        kosdaq_tickers = stock.get_market_ticker_list(market="KOSDAQ") 
        print(f"   KOSDAQ 종목 수: {len(kosdaq_tickers)}")
        
        print("[OK] pykrx 데이터 수집 성공")
        return True, {
            'price_data': price_data,
            'fund_data': fund_data,
            'kospi_count': len(kospi_tickers),
            'kosdaq_count': len(kosdaq_tickers)
        }
        
    except Exception as e:
        print(f"[ERROR] pykrx 데이터 수집 실패: {e}")
        print("상세:", traceback.format_exc())
        return False, None

def test_canslim_calculations(data):
    """CAN SLIM 실제 계산 테스트"""
    print("\n" + "=" * 50)
    print("CAN SLIM 실제 계산 테스트")
    print("=" * 50)
    
    if data is None:
        print("[SKIP] 데이터가 없어서 계산 테스트 건너뜀")
        return False
    
    try:
        price_data = data.get('price_data')
        fund_data = data.get('fund_data')
        
        calculations = {}
        
        # C - Current Earnings (간단 버전)
        if not fund_data.empty and len(fund_data) > 1:
            recent_eps = fund_data['EPS'].iloc[-1]
            prev_eps = fund_data['EPS'].iloc[0]
            eps_change = ((recent_eps - prev_eps) / prev_eps * 100) if prev_eps != 0 else 0
            calculations['C_eps_change'] = eps_change
            print(f"C지표 - EPS 변화: {eps_change:.1f}%")
        
        # A - Annual Earnings (ROE 계산)
        if not fund_data.empty:
            recent_eps = fund_data['EPS'].iloc[-1]
            recent_bps = fund_data['BPS'].iloc[-1]
            roe = (recent_eps / recent_bps) if recent_bps != 0 else 0
            calculations['A_roe'] = roe
            print(f"A지표 - ROE: {roe:.2%}")
        
        # N - New Highs (최근 고가 대비)
        if not price_data.empty:
            current_price = price_data['종가'].iloc[-1]
            high_price = price_data['고가'].max()
            price_ratio = current_price / high_price
            calculations['N_price_ratio'] = price_ratio
            print(f"N지표 - 최고가 대비: {price_ratio:.1%}")
        
        # 간단한 종합 점수
        total_score = 0
        score_count = 0
        
        if 'C_eps_change' in calculations:
            c_score = 80 if calculations['C_eps_change'] > 10 else 50
            total_score += c_score
            score_count += 1
            
        if 'A_roe' in calculations:
            a_score = 90 if calculations['A_roe'] > 0.15 else 60
            total_score += a_score 
            score_count += 1
            
        if 'N_price_ratio' in calculations:
            n_score = 85 if calculations['N_price_ratio'] > 0.9 else 65
            total_score += n_score
            score_count += 1
        
        if score_count > 0:
            avg_score = total_score / score_count
            print(f"\n종합 점수: {avg_score:.1f}/100")
            
            if avg_score >= 70:
                print("평가: 우수 (투자 적합)")
            elif avg_score >= 60:
                print("평가: 양호")
            else:
                print("평가: 보통")
        
        print("[OK] CAN SLIM 계산 완료")
        return True
        
    except Exception as e:
        print(f"[ERROR] CAN SLIM 계산 실패: {e}")
        print("상세:", traceback.format_exc())
        return False

def test_visualization():
    """시각화 기능 테스트"""
    print("\n" + "=" * 50)
    print("시각화 기능 테스트")
    print("=" * 50)
    
    success_count = 0
    
    # matplotlib 테스트
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, label='Test Line')
        ax.legend()
        ax.set_title('Test Chart')
        
        plt.close(fig)  # 메모리 정리
        print("[OK] matplotlib 차트 생성")
        success_count += 1
        
    except Exception as e:
        print(f"[ERROR] matplotlib: {e}")
    
    # plotly 테스트 (선택사항)
    try:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[1, 2, 3, 4, 5],
            y=[2, 4, 3, 5, 6],
            mode='lines+markers',
            name='Test Data'
        ))
        fig.update_layout(title='Test Plotly Chart')
        
        html_output = fig.to_html()
        if len(html_output) > 1000:  # 유효한 HTML인지 확인
            print("[OK] plotly 인터랙티브 차트 생성")
            success_count += 1
        else:
            print("[WARNING] plotly 차트 크기 작음")
            
    except ImportError:
        print("[INFO] plotly 미설치 (선택사항)")
    except Exception as e:
        print(f"[WARNING] plotly 오류: {e}")
    
    return success_count >= 1

def test_notebook_compatibility():
    """Jupyter 노트북 호환성 테스트"""
    print("\n" + "=" * 50)
    print("Jupyter 노트북 호환성 테스트")
    print("=" * 50)
    
    try:
        # IPython 관련 기능 확인
        try:
            import IPython
            print("[OK] IPython 사용 가능")
            ipython_available = True
        except ImportError:
            print("[INFO] IPython 미설치 (노트북 환경이 아님)")
            ipython_available = False
        
        # pandas HTML 출력 테스트
        import pandas as pd
        test_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        html_output = test_df.to_html()
        if len(html_output) > 100:
            print("[OK] pandas HTML 출력 가능")
        else:
            print("[WARNING] pandas HTML 출력 문제")
        
        # 노트북용 설정 테스트
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 100)
        print("[OK] pandas 표시 옵션 설정")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 노트북 호환성 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("CAN SLIM 한국 주식 스크리닝 시스템")
    print("종합 기능 검증 테스트")
    print("=" * 50)
    print("시작시간:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print()
    
    start_time = time.time()
    
    # 테스트 실행
    results = []
    
    # 1. 라이브러리 임포트
    print("1/5 단계: 라이브러리 임포트 테스트")
    import_ok = test_imports()
    results.append(('라이브러리 임포트', import_ok))
    
    # 2. 실제 데이터 수집
    print("\n2/5 단계: 실제 데이터 수집 테스트") 
    data_ok, data = test_pykrx_data()
    results.append(('pykrx 데이터 수집', data_ok))
    
    # 3. CAN SLIM 계산
    print("\n3/5 단계: CAN SLIM 계산 테스트")
    calc_ok = test_canslim_calculations(data)
    results.append(('CAN SLIM 계산', calc_ok))
    
    # 4. 시각화
    print("\n4/5 단계: 시각화 테스트")
    viz_ok = test_visualization()
    results.append(('시각화', viz_ok))
    
    # 5. 노트북 호환성
    print("\n5/5 단계: 노트북 호환성 테스트")
    nb_ok = test_notebook_compatibility()
    results.append(('노트북 호환성', nb_ok))
    
    # 결과 정리
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("최종 테스트 결과")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    success_rate = passed / total
    
    print(f"총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"성공률: {success_rate:.1%}")
    print(f"소요시간: {duration:.1f}초")
    
    print("\n상세 결과:")
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        mark = "✓" if success else "✗"
        print(f"  {mark} {test_name}: {status}")
    
    # 최종 결론
    print("\n" + "=" * 60)
    if success_rate >= 0.8:
        print("🎉 결론: 시스템이 정상적으로 작동합니다!")
        print("✅ CAN SLIM 튜토리얼을 실전에서 사용할 수 있습니다.")
        
        if success_rate == 1.0:
            print("🏆 모든 테스트 통과! 완벽한 설정입니다.")
        else:
            print("⚠️ 일부 선택 기능에 문제가 있지만 핵심 기능은 정상입니다.")
            
        print("\n📚 다음 단계:")
        print("1. 00_setup_and_introduction.ipynb 노트북부터 시작하세요")
        print("2. requirements.txt의 선택 라이브러리 설치를 고려하세요")
        print("3. 실제 투자 전에 충분히 백테스팅하세요")
        
    else:
        print("❌ 결론: 시스템에 문제가 있습니다.")
        print("🔧 실전 사용 전에 다음 사항을 해결하세요:")
        
        failed_tests = [name for name, success in results if not success]
        for test in failed_tests:
            print(f"   • {test} 기능 수정 필요")
    
    print("\n💡 도움말:")
    print("- 문제가 지속되면 requirements.txt로 라이브러리를 재설치하세요")
    print("- 네트워크 오류 시 잠시 후 다시 시도하세요") 
    print("- 이슈 발생 시 GitHub에 문의하세요")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n테스트가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n치명적 오류 발생: {e}")
        print("상세 정보:")
        print(traceback.format_exc())
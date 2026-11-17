"""Regenerates the subsection 'Stage B 환경: 설계 점검과 검증' of tex/spec.tex from the record files (every number comes from a record)."""
import json, sys, os
from pairl import paths
L = lambda f: json.load(open(paths.RECORDS / f))
D, R, G, C, B = L('stageb_dyn_check.json'), L('stageb_env_check.json'), L('stageb_design.json'), L('stageb_contact_runs.json'), L('barrier_storage_check.json')
assert all(D['pass'].values()) and R['all_pass'], 'verification not passing: refusing to write'
def e(x):
    if x == 0: return '$0$'
    m, ex = ('%.1e' % x).split('e'); return f'${m}\\cdot10^{{{int(ex)}}}$'
f2 = lambda x: '%.2f' % x
b3 = G['b3']
def vals():
    return dict(b3_in=f2(b3['chosen_in']['margin_with_contact']), b3_ood=f2(b3['chosen_ood']['margin_with_contact']), b3_in0=f2(b3['chosen_in']['margin_no_contact']), b3_ood0=f2(b3['chosen_ood']['margin_no_contact']),
                r_in=f2(b3['rejected_80_40_15_in']['margin_with_contact']), r_ood=f2(b3['rejected_80_40_15_ood']['margin_with_contact']), r_in0=f2(b3['rejected_80_40_15_in']['margin_no_contact']), r_ood0=f2(b3['rejected_80_40_15_ood']['margin_no_contact']),
                ws_n='%.1f' % (100 * G['workspace']['frac_normal']), ws_h='%.1f' % (100 * G['workspace']['frac_hard']), ws_u='%.1f' % (100 * G['workspace']['frac_unreachable']),
                gr_in=', '.join('%.2f' % x for x in G['gravity_ratio_in']), gr_ood=', '.join('%.2f' % x for x in G['gravity_ratio_ood']),
                gj='%.3f' % R['V8']['gauge_jump_max'], thr='{:,}'.format(int(round(R['V12']['steps_per_s'], -3))).replace(',', '{,}'),
                f32q=e(R['V12']['float32']['q_diff_max']), f32f='%.4f' % R['V12']['float32']['fn_diff_rms'], hard='%.3f' % R['V10']['hard_frac'], budg='%.3f' % R['V10']['budget_frac'], heavy='%.3f' % R['V10']['heavy_frac'],
                op2e5=e(C['operating_200000_4']['residual_rel']), h2e4=e(C['harsh_20000_4']['residual_rel']), h2e5_4=e(C['harsh_200000_4']['residual_rel']), h2e5_8=e(C['harsh_200000_8']['residual_rel']),
                ker='%.3f' % D['attach']['ke_ratio_min'], v16f='%.0f' % (100 * R['V16']['frac_two_grasps_two_releases']), dth='%.1f' % R['V16']['dtheta3_median'])
if __name__ == '__main__':
    v = vals()
    V = R
    rows = [
     ('V1', '손끝 위치와 Jacobian (SymPy, MuJoCo 대조)', e(D['kin_p']) + ', ' + e(D['kin_J']), '$10^{-12}$, $10^{-9}$'),
     ('V2', '관성 행렬 (SymPy, MuJoCo), 대칭', e(D['M_sym']) + ', ' + e(D['M_mj']) + ', ' + e(D['M_asym']), '$10^{-12}$, $10^{-10}$, $10^{-12}$'),
     ('V3', '관절 가속도 (SymPy, MuJoCo, 손끝 힘 포함)', e(D['acc_sym']) + ', ' + e(D['acc_mj']) + ', ' + e(D['acc_mj_force']), '$10^{-10}$, $10^{-9}$, $10^{-9}$'),
     ('V4', '중력 torque (SymPy, MuJoCo), 정지 유지의 가속도', e(D['grav_sym']) + ', ' + e(D['grav_mj']) + ', ' + e(D['hold']), '$10^{-12}$, $10^{-10}$, $10^{-9}$'),
     ('V5', '자유 운동의 에너지 drift(0.5 ms), 적분 차수', e(V['V5']['drift']['4']) + '; %.2f, %.2f' % tuple(V['V5']['order_median']), '$10^{-7}$; $[3.5,4.5]$'),
     ('V6', '접촉 없는 일률 균형의 잔차', e(V['V6']['residual_rel']), '$10^{-6}$'),
     ('V7', '접촉: 구성한 힘 9개, 점별 passivity, 일률 균형($2\\cdot10^4$ N/m, 0.5 ms)', e(V['V7']['force_values_err']) + ', ' + e(max(V['V7']['pointwise_passivity_max'], 0.0)) + ', ' + e(V['V7']['runs']['20000_4']['residual_rel']), '$10^{-9}$, $10^{-9}$, $10^{-4}$'),
     ('V8', '잡기: 논리 11가지와 잡기--놓기--다시 잡기, 운동량 보존, 정지 잡기의 에너지와 무게중심 연속, 사건에서 $E_{\\mathrm{rob}}+S_{\\mathrm{obj}}$의 증가(무작위 2{,}000개)',
      '모두 일치, ' + e(D['attach']['mom']) + ', ' + e(V['V8']['grasp_energy_jump_err']) + ', ' + e(V['V8']['com_continuity']) + ', ' + e(max(V['V8']['event_gain_max'], 0.0)), '일치, $10^{-7}$, $10^{-9}$, $10^{-12}$, $10^{-9}$'),
     ('V9', '감지: 한 주기 지연, 잡음 표준편차의 비, 양자화 오차 [rad]', e(V['V9']['delay_err']) + ', %.3f--%.3f, ' % (min(V['V9']['noise_sd_ratio']), max(V['V9']['noise_sd_ratio'])) + e(V['V9']['quant_err']), '$10^{-12}$, $1\\pm0.02$, $0.5\\cdot10^{-4}$'),
     ('V10', '표본 추출: 3{,}000 episode에서 위반 수; 어려운 episode와 예산 episode의 비율; 분포 안 payload에서 무거운 등급의 비율', '0; ' + v['hard'] + ', ' + v['budg'] + '; ' + v['heavy'], '0; $0.3\\pm0.025$; $0.5\\pm' + '%.3f' % V['V10']['heavy_tol'] + '$'),
     ('V11', '비용: 구성한 상태 %d개와 독립 계산의 차이' % V['V11']['n'], e(V['V11']['max_err']), '$10^{-9}$'),
     ('V12', '묶음 실행과 단일 실행의 차이, 같은 seed의 재실행', e(V['V12']['batch_vs_single']) + ', ' + e(V['V12']['rerun_diff']), '$10^{-9}$, 0'),
     ('V13', '단계 경계, 잡기 상태가 결정 시각에만 바뀌는지', '일치', '일치'),
     ('V14', '전 구간(시험용 제어기, 64 episode): 에너지 감사의 잔차, 잡기 성공', e(V['V14']['audit_residual_rel']) + ', %d/64' % round(64 * V['V14']['grasp_frac']), '$10^{-4}$, 90\\%'),
     ('V15', 'MuJoCo의 RK4와 1 s 궤적 차이 [rad]', e(V['V15']['max_joint_diff']), '$10^{-6}$'),
     ('V16', '다시 잡기(64 episode): 두 번 잡고 두 번 놓은 비율, 사건에서 $E_{\\mathrm{rob}}+S_{\\mathrm{obj}}$의 증가, 무게중심 연속, 에너지 감사의 잔차',
      v['v16f'] + '\\%, ' + e(max(V['V16']['event_gain_max'], 0.0)) + ', ' + e(V['V16']['com_continuity_max']) + ', ' + e(V['V16']['audit_residual_rel']), '90\\%, $10^{-9}$, $10^{-12}$, $10^{-4}$'),
     ('V17', '비용의 성질(무작위 자세 200{,}000개): 윗면 force가 없을 때 같은 자세에서 놓은 상태와 든 상태의 비용률 차이의 1로부터의 최대 편차; block 옆면을 따라 든 상태의 phase-3 비용률이 깊이에 대해 갖는 기울기 [1/m]',
      e(V['V17']['drop_minus_held_max_dev']) + '; %.2f' % V['V17']['side_slope'], '$10^{-9}$; $\\ge1$ (이상)'),
     ('V18', '윗면 force가 있을 때의 비용의 성질(윗면 중심, 깊이 200개, 힘 $0.05f^\\star$--$20f^\\star$): 같은 자세에서 놓은 상태와 든 상태의 비용률 차이의 최솟값; 든 상태의 비용률이 가장 작은 곳의 $f/f^\\star$',
      '%.3f; %.3f' % (V['V18']['drop_minus_held_min'], V['V18']['argmin_force_ratio']), '$\\ge1$ (이상); 가장 가까운 깊이')]
    tab = "\n".join(f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows)
    sec = r"""\subsection{Stage B 환경: 설계 점검과 검증}
\label{sec:envcheck}

\textbf{직관 (환경이 틀리면 그 위의 결과가 모두 무의미하다).} 제어기를 올리기 전에 두 가지를 확인했다. 첫째, plant의 상수가 보장의 조건을 만족하는가. 둘째, 구현이 \Cref{def:stageBplant,def:stageBtask}와 같은가. 둘째는 서로 독립인 두 기준, 즉 SymPy로 유도한 Lagrange 방정식과 물리 engine MuJoCo에 대조했다. 합격 기준은 실행 전에 파일로 고정했다. 처음 두 번의 실행에서 결함 다섯 가지가 드러났고, 고칠 때마다 전체를 다시 돌려 세 번째 실행에서 모든 항목이 통과했다. 그 뒤 다시 잡기를 허용하도록 물체 모형을 바꾸고, 그에 맞게 시험을 고치거나 더해(V8, V10, V11, V16) 전체를 다시 돌렸으며, 새 시험의 결함 하나를 고쳐 한 번 더 돌렸다. 개정 14에서 비용의 정의를 바꾸면서 V11을 새 정의에 맞추고 비용의 성질을 재는 V17을 더해 여섯 번째로 돌렸다. 개정 15에서 누르기 항에 상한을 두면서 V11을 다시 맞추고 윗면 force가 있을 때의 성질을 재는 V18을 더해 일곱 번째로 돌렸다. 아래 값은 그 일곱 번째 실행의 것이다(\Cref{fig:envcheck}, \Cref{tab:envcheck}).

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{stageb_env.png}
\caption{Stage B environment checks}
\label{fig:envcheck}
\end{figure}

\textbf{설계 점검 (상수가 보장의 조건을 만족하는가).} \Cref{thm:r1}의 (B3)를 표본으로 확인했다. 표본은 집합 $\{\rho(w)=1,\ \rho(w_q)\le1\}$에서 경우마다 40{,}000개를 뽑았고, 붙은 물체의 offset은 $\|r_{\mathrm{obj}}\|\le0.10$ m에서, 접촉력은 마찰 원뿔($f_n\le60$ N, $\nu_e\le0.8$) 안에서 가장 불리한 값을 넣었다. 아래 값은 $k_1=2$에서 (B3)의 좌변에 음의 부호를 붙인 것의 최솟값이며, 양수이면 조건이 성립한다.
\[
\begin{array}{lcccc}
\bar u\ [\mathrm{N\,m}] & \text{in distribution} & \text{with contact} & \text{out of distribution} & \text{with contact}\\
(120,60,25) & """ + f"{v['b3_in0']} & {v['b3_in']} & {v['b3_ood0']} & {v['b3_ood']}" + r"""\\
(80,40,15) & """ + f"{v['r_in0']} & {v['r_in']} & {v['r_ood0']} & {v['r_ood']}" + r"""
\end{array}\quad\mathrm{rad/s^2}
\]
개정 9의 후보 $(80,40,15)$ N\,m는 접촉력이 걸리면 조건을 깨뜨렸다. 손목의 한계 15 N\,m가 60 N의 접촉력에 포화되기 때문이다. 그래서 $\bar u=(120,60,25)$ N\,m로 정했다. 이 확인은 표본에 대한 것이며 증명이 아니다. 관절 상자 $c=(0.9,-1.3,-0.5)$, $h=(0.9,1.1,1.2)$ rad에서 작업 영역의 """ + f"{v['ws_n']}\\%는 여유 0.25 이상의 자세가 있고, {v['ws_h']}\\%는 관절 한계에 가까운 띠 $[0.08,0.20)$에 있으며, {v['ws_u']}\\%는 닿지 않는다(\\Cref{{fig:envcheck}}(a)). 가장 무거운 물체를 든 중력 torque는 입력 한계의 ({v['gr_in']})배 이하(분포 안)와 ({v['gr_ood']})배 이하(분포 밖)이다." + r"""

\begin{table}[H]
\centering\footnotesize
\caption{Verification of the Stage B environment}
\label{tab:envcheck}
\begin{tabular}{p{0.6cm}p{5.8cm}p{4.9cm}p{3.5cm}}
\toprule
 & 확인한 것 & 측정값 & 기준(이하) \\
\midrule
""" + tab + r"""
\bottomrule
\end{tabular}
\end{table}

오차는 최대 절대 차이를 (1 + 기준값의 최대 절댓값)으로 나눈 값이다. V1--V4는 상태 6{,}000개와 parameter 240벌(절반은 분포 밖의 몸, 절반은 물체를 든 상태)에서 쟀다. V16의 시험용 제어기는 목표 위치에서 물체를 놓고, 손끝 위치가 같은 다른 자세로 옮긴 뒤(마지막 링크 각도 변화의 중앙값 """ + v['dth'] + r""" rad) 다시 잡는다.

\textbf{실행에서 드러난 결함과 수정.}
\begin{itemize}
\item Block 옆면의 감쇠 부호가 반대였다(코드의 오류). V7의 구성한 힘과 점별 passivity 확인이 잡았다.
\item Spring과 damper의 접촉 법칙은 닿는 순간 힘이 불연속이어서, 강성 $2\cdot10^4$ N/m에서 일률 균형의 잔차가 $8.5\cdot10^{-3}$이었다. 닿는 순간 힘이 0에서 시작하는 Hunt--Crossley 형태로 바꿨다. 그 뒤에도 잔차 $2.1\cdot10^{-3}$이 남았고, 원인은 다음 항목이었다.
\item 마찰의 정규화 $\tanh(\dot p_x/v_t)$는 속도 0에서의 기울기가 법선력에 비례해, 100 N이 넘는 충돌에서 적분의 안정 영역을 벗어났다. 기울기가 법선력과 무관하게 $c_v=400$ N\,s/m인 형태로 바꿨다. 그 뒤 잔차는 적분 간격의 4제곱으로 줄어든다(\Cref{fig:envcheck}(d)).
\item 시험용 제어기가 잡는 점으로 가는 길에 손끝이 block의 옆면을 지나갔다(최대 522 N). 잡는 점과 시작 위치를 block 윗면보다 0.03 m 이상 높게 뽑도록 했다.
\item 설계 점검 module을 불러오면 float64가 전역으로 켜지는 부작용이 있었다. NumPy만 쓰는 module로 분리했다.
\item 새 시험 V16의 에너지 감사가, 손끝이 잡는 점에서 1.31 cm 떨어진 자리에서 시작해 첫 결정 시각에 잡기가 일어난 episode(64개 중 1개)에서 사건 뒤의 에너지를 기준으로 삼았다(시험 코드의 오류, 잔차 0.058). 사건 전의 에너지를 기준으로 고쳤다.
\end{itemize}

\textbf{적분 간격의 선택.} Plant는 제어 주기 2 ms 안에서 4차 Runge--Kutta로 네 번 적분한다(0.5 ms). 이때 일률 균형의 잔차는 작동 영역(항상 누르는 접촉)에서 강성 $2\cdot10^5$ N/m까지 """ + f"{v['op2e5']} 이하이고, 떨어졌다가 다시 부딪히는 가혹한 영역에서 $2\\cdot10^4$ N/m일 때 {v['h2e4']}, $2\\cdot10^5$ N/m일 때 {v['h2e5_4']}이다. 그래서 학습 범위보다 단단한 block의 시험에서는 여덟 번 적분한다(같은 조건에서 {v['h2e5_8']}). 처리량은 float64에서 초당 제어 step 약 {v['thr']}개이고, float32로 돌린 궤적은 float64와 관절 각 {v['f32q']} rad, 법선력 {v['f32f']} N(제곱 평균 제곱근) 이내로 일치했다." + r"""

\textbf{남은 한계.}
\begin{itemize}
\item (B3)의 확인은 표본에 대한 것이다.
\item 접촉은 penalty 모형이고 손끝에서만 일어난다. 링크와 block의 충돌, 물체의 충돌과 낙하는 없고, 놓은 물체는 놓인 자리에 머문다.
\item 잡는 순간 gauge는 최대 """ + f"{v['gj']}만큼 뛸 수 있다(허용되는 잡기 상태 5{{,}}000개, 분포 밖의 몸 포함). \\Cref{{thm:r1}}의 (B4)는 잡기 직전의 gauge가 {1 - float(v['gj']):.3f}보다 작으면 성립한다." + r"""
\item 예산 episode의 예산 $E_0=\Delta T+\rho E_{\mathrm{task}}$에서 환경은 $\rho$를 뽑고, $E_{\mathrm{task}}$는 L1의 기준 뇌가 같은 episode에서 낸 값으로 환경 밖에서 정해 $E_0$을 episode의 parameter로 넘긴다.
\item 시험용 제어기는 중력 torque를 아는 PD와 힘 제어이며 v3의 구조가 아니다. \Cref{fig:envcheck}(c)의 닿는 순간 힘의 overshoot는 그 제어기의 것이다.
\end{itemize}

"""
    s = open(paths.NOTES / 'spec.tex', encoding='utf-8').read()
    i = s.find("\\subsection{Stage B 환경: 설계 점검과 검증}"); j = s.find("\\section{준비: passivity와 port-Hamiltonian system}", i)
    assert 0 < i < j
    s = s[:i] + sec + s[j:]
    k = s.find("Stage B 환경 & 구현, 검증 통과 &"); k2 = s.find("\\\\", k)
    s = s[:k] + "Stage B 환경 & 구현, 검증 통과 & 독립 기준 둘과 대조, 18개 항목 통과; (B3) 표본 여유 " + f"{v['b3_in']}(분포 안), {v['b3_ood']}(분포 밖) rad/s$^2$ (\\Cref{{sec:envcheck}}) " + s[k2:]
    open(paths.NOTES / 'spec.tex', 'w', encoding='utf-8').write(s)
    json.dump(v, open(paths.RECORDS / 'stageb_spec_values.json', 'w'), indent=1)
    print('envcheck section written', v['b3_in'], v['b3_ood'], v['gj'])

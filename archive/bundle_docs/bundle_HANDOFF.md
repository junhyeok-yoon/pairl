# 인수인계: v3 brain–hand RL framework, Stage B (2026-11-15 기준)

이 문서는 cloud 환경에서 진행한 연구를 사용자의 로컬 Linux PC(RTX 5080)의 Claude Code가 이어받기 위한 문서이다. 사실은 모두 번들 안의 기록(`records/`)과 사양(`tex/spec.tex`)에 근거하며, 여기 적은 수치는 그 기록에서 옮긴 것이다. 기록과 이 문서가 다르면 기록이 옳다.

## 0. 읽는 순서

1. 이 문서 전체.
2. `docs/Research_Principles_and_Guidelines.md` — 사용자의 작성·검증 규칙. 다른 프로젝트(SAG, PI servo)에서 만든 문서이므로 **§1–§4의 기술 내용(SAG 구조, settled 사실, DO NOT의 SAG 항목)은 이 연구에 적용하지 않는다.** 적용하는 것은 §5(원칙), §5a(공통 작성 원칙: A–E), §5b(`.tex` 규칙), §6(소통), §5c 안의 K7·K7b·K8(검증값 기록, divergence ledger, 정합 게이트; 모든 산출물에 적용), §7의 렌더링·일괄 편집·번들 규칙이다. §5c의 나머지는 발표자료를 만들 때만 적용한다.
3. `tex/spec.tex` (렌더본 `spec.pdf`, 67쪽). 연구의 단일 원본이다. 먼저 읽을 절(label):
   - `Overview`, `이전 설계에서 배운 것` — 목표와 설계 원칙
   - `System과 과제` 안의 `sec:stageB`(과제 정의, `def:stageBtask`, `prop:costprog`), `sec:envcheck`
   - `구조`(손 `def:hand`, tank `thm:fulltank`, guard `prop:static`, 제약층 `thm:r1`, 뇌 `def:brain`와 개정 16의 행동)
   - `학습 framework`(`def:coupled`, `lem:restfloor`)
   - `구현, 학습, 검증 계획` 안의 L1(`sec:l1`), L2 첫·두 번째 시도(`sec:l2a1`, `sec:l2a2`), pilot R(`sec:l2pr`), pilot T(`sec:l2pt`), 세 번째 시도(`sec:l2a3`), 반복 300의 학습 상태(`sec:l2status`)
   - `요구 사항`, `현재 상태`(표 `tab:status3`), `지시 사항과의 대응`
4. `README.md` — 코드와 기록의 지도.
5. `records/l2_attempt3_plan.md` — 가장 최근의 사전 등록과 결과(hash 기록 `records/l2_attempt3_plan.hash`).

## 1. 연구 목표와 바꾸면 안 되는 것

**목표.** 적응 제어의 철학을 구조에 담은 RL framework. 정책 = 뇌(학습된 NN, task 수준의 의도) + 손(가상 impedance와, 정지 상태에서 측정된 mismatch로 갱신되는 적응 기억). 보장(제약 만족, passivity)은 학습과 무관하게 구조로 성립한다. 핵심 기여는 두 층이 서로의 역할을 알고 정보를 주고받는 결합(뇌→손: goal과 강성; 손→뇌: 에너지 장부)이다. 사용자가 지금 가장 중요하게 보는 연구 질문은 **"구조 안에서 각 parameter가 의도한 역할대로 학습되게 하는 학습 체계(reward, update 설계)"**이다. 최종적으로 7-DOF로 간다.

**불변 조건 (사용자 승인 없이 바꾸지 않는다).**
- 보장은 학습되지 않는 요소(energy tank와 gate, 제약층, direction-preserving saturation, 접촉 guard)에서 나온다. 학습이 최적이어야 보장이 서는 방식은 받아들이지 않는다.
- NN의 비선형성과 universal approximation을 유지한다(선형화·선형 head로 바꾸지 않는다).
- class 수준의 구조만 쓴다. NN 안에 instance 모형(예: 중력 항)을 넣지 않는다. 공칭 기구학(공칭 링크 길이의 Jacobian), torque sensing은 class 수준으로 허용되어 쓰고 있다.
- State feedback만 다룬다(output feedback은 보류).
- 단순한 적층(RL이 reference를 만들고 적응 제어가 추종)은 기여로 보지 않는다.

## 2. 사용자 작업 규칙 (항상 적용)

**소통.** 한국어, 전문 용어는 영어 그대로, 반말 금지(지금까지 "-습니다"체), 이모지 금지, 한자 금지, 핵심 먼저, 장황·중복 금지, 인과 순서로 설명. 불가능하면 틀린 답을 만들지 말고 이유를 말한다. Code comment는 영어로 짧게, `###`·`---` 같은 구분선 금지.

**연구 노트.** LaTeX(`tex/spec.tex`에 절을 더하는 방식). Definition·statement는 영어, proof와 해설 prose는 한국어 한다체, figure caption은 영어 제목만, meta 서술 금지, 수식 중심. 가이드라인 §5a–§5b를 따른다. 렌더는 `./build_tex.sh spec`(fontspec + Noto Serif CJK KR, XeLaTeX 3-pass; 원본은 `kotex` 유지), 게이트는 errors/undefined/multiply-defined/overfull/missing-char 모두 0. 렌더 후 `build/spec.pdf`를 번들 루트 `spec.pdf`로 복사한다.

**실험.**
- **설계 변경은 제안 후 사용자 승인 뒤에만** 한다. 설계 변경 = 비용, 관측, 행동, network, loss, hyperparameter, 시작 상태 분포, 평가 규칙 등 결과에 영향을 주는 모든 것. 학습 run(pilot 포함)도 승인 후에 돌린다. 승인이 필요 없는 것: 재현·환경 점검(`port_check.py`, `verify_*.py`), 기록을 읽는 분석, 학습 없는 rollout 진단, 문서 빌드, `check_sync.py`.
- **사전 등록.** 시험은 돌리기 전에 `records/<이름>_plan.md`에 질문·구성·명령·판정 규칙을 쓰고, `sha256` 앞 8자리를 `records/<이름>_plan.hash`에 한 줄 추가한다(개정할 때마다 새 줄). 결과는 같은 파일 끝의 `## Outcome ...`에 기록하고 다시 hash를 남긴다. 등록하지 않은 분석은 "not registered; descriptive"로 표시한다.
- **수치는 기록에 있는 그대로** 보고한다. 기록에 없는 값은 쓰지 않는다.
- **Seed 규칙 (사용자 원문):** "앞으로 내 별도 지시 없이는 seed별 학습은 절대 하지마. 한 seed만 할 것. 시간 낭비임." 작은 pilot은 허용.
- **학습이 잘 안 되면** 파고들어 각 구성 요소가 의도대로 학습·작동했는지 검증한다(분해).
- **학습 보고에는 데이터 그림과 rollout GIF**를 넣고, 긴 run은 **중간 보고**를 한다(지금까지 50 반복마다).
- 인용을 지어내지 않는다. 새로움을 주장하기 전에 web search로 확인한다.
- 사용자의 개인 정보(email 등)를 외부 서비스에 보내지 않는다.

**정합 게이트.** 문서를 고치거나 전달하기 전에 `cd code && python3 check_sync.py`가 `sync findings: 0`이어야 한다. 이 스크립트는 spec이 인용한 수치를 기록 JSON에서 다시 계산해 문자열로 대조하고, plan hash, 폐기된 문자열, `records/verified_values.md`의 `## Divergence (open)`가 비었는지를 검사한다. spec에 새 수치를 쓰면 같은 방식의 검사를 `check_sync.py`에 함께 추가한다(기존 블록 `# attempt 3 outcome`, `# status at iteration 300` 참고).

## 3. 번들 구조

```
HANDOFF.md, CLAUDE.md, README.md
spec.pdf, spec_v2.pdf, note_core.pdf, note_learning.pdf    렌더본
build_tex.sh                    tex 렌더 (./build_tex.sh spec -> build/spec.pdf)
docs/                           사용자의 작성 규칙, 노트 preamble 예시
tex/                            spec.tex (현재), spec_v2.tex (중단된 v2), note_core.tex, note_learning.tex, TikZ 그림
code/                           check_sync.py (정합 게이트), 앞선 단계의 점검 스크립트
code/stageb/                    Stage B 환경과 학습 코드 (아래)
figures/                        PNG 그림과 rollout GIF
records/                        사전 등록(plan, hash), run 디렉터리, 진단 JSON, verified_values.md
archive/earlier_bundle/         v3 이전 방향들의 파일. 현재 작업에는 필요 없음
```

`code/stageb/`의 핵심(모두 `code/stageb`에서 실행, 경로는 상대 경로 `../../records`):
- 환경: `params.py`(상수), `model.py`(동역학, 접촉), `env.py`(과제, 비용 `cost_rate`, restart 상태 `restart`), `controller.py`(손, tank, guard, 제약층, 장부), `l1.py`(pool, 기준 뇌, 평가).
- 학습: `l2.py`(network, rollout, PPO loss, role loss, 진단), `l2_train.py`(한 seed run).
- 진단: `l2_dissect.py`, `l2_restart_behavior.py`, `l2_signal_check.py`, `l2_hand_check.py`, `l2_hand_replay.py`, `l2_hand_fresh.py`, `l2_transfer_exposure.py`, `l2_ledger_check.py`, `l2_traj_check.py`, `l2_never_rest.py`, `anim_rollout.py`(GIF), `fig_l2_*.py`(그림), `report_checkpoint.sh`(checkpoint 보고 일괄).
- 점검: `verify_dyn.py`, `verify_env.py`(V5–V18, 기준 `records/stageb_env_criteria.md`), `f32_run.py`(float32 궤적과 float64 기준 비교, `verify_env.py`가 만든 기준 파일 필요), `port_check.py`(새 기계의 재현과 처리량).

기록 규약: 진단 JSON의 key는 `"<run 디렉터리>:<반복>"`(예: `l2_attempt3:300`, signal check는 `:start2`/`:start3` 접미사). Run 디렉터리에는 `train.log`, `hist.json`(반복별 통계), `evals.json`(10 반복마다 평가, 세 시작에서 64 episode씩), `params_itXXXX.pkl`(10 반복마다; 뇌, 손, 행동 모드 `act`, optimizer 상태 `opt`), `dissect_itXXXX.json`이 있다.

## 4. 현재 시스템 (spec 개정 16)

**Plant와 제약.** 평면 3관절 팔, gripper, 물체, block. 관절 상자 $c=(0.9,-1.3,-0.5)$, $h=(0.9,1.1,1.2)$ rad. 입력 상자 $\bar u=(120,60,25)$ N·m. 제어 주기 2 ms(RK4 네 번), 뇌의 결정 주기 50 ms(25 제어 step), episode 최대 11 s(결정 220개).

**과제의 네 단계** (길이는 episode마다 무작위: 1.5–2.5, 2–3, 2–3, 1.5–2.5 s): (1) 물체로 가서 잡기, (2) $p^\star$로 옮기기, (3) block 윗면을 $f^\star$로 누르기, (4) 놓고 $p^\star$로 물러나기. 잡기 조건: gripper 닫힘 명령, 손끝이 잡는 점에서 2 cm 안, 관절 속도 최대값 0.1 rad/s 이하. 목표 힘은 질량 등급(무거움: 물체 질량 > 0.85 kg)에 따라 $f_{\mathrm{light}}$ 또는 $f_{\mathrm{heavy}}$이고, 뇌는 두 후보를 관측하지만 등급은 관측하지 않는다(장부의 하중으로 판단해야 한다).

**비용(개정 15, `env.cost_rate`, `env.COST_V = 4`).** 보상은 결정 구간 비용률 적분의 음수. 거리는 0.1 m 단위. 단계 1: 손끝–잡는 점 거리. 단계 2: 들었으면 $\|p-p^\star\|$, 아니면 $1+\|p-o\|+\|o-p^\star\|$. 단계 3: 들었으면 누르기 항 $+|p_x-p^\star_x|$, 아니면 $1+\|p-o\|+(1+\text{물체–윗면 중심 거리}+|o_x-p^\star_x|)$; 누르기 항은 윗면 힘이 있으면 $\min\{((f-f^\star)/f^\star)^2,1\}$, 없으면 $1+\|p-p_c\|$. 단계 4: $\|p-p^\star\|+\xi$. 모든 단계에 $5\max(f-2f^\star,0)/f^\star$.

**뇌.** Actor MLP 31–128–128–13(tanh). 출력: 평균 6, 표준편차 6(상태 의존, 범위 goal [0.005, 0.3], 강성 [0.05, 0.5]), gripper logit 1. Critic MLP 37–128–128–1(관측 31 + privileged 6: 단계의 남은 시간, 물체 질량, $f^\star$, $p^\star$와 윗면의 높이 차, 등급, 예산).
- 관측(31): 관절 자세(정규화) 3, 속도 3, 측정 외력 torque 3, 단계 one-hot 4, 잡는 점 2, $p^\star$ 2, $(f_{\mathrm{light}},f_{\mathrm{heavy}})$ 2, 손끝 2, 잡음 $\xi$ 1, tank 수준 1, 장부 $\bar\lambda$ 1, 장부 하중 $u_{\mathrm{load}}$ 3, 상대 위치 $o-p$ 2, $p^\star-p$ 2.
- 행동(개정 16, `l2.goal_of`, 모드 `task`): $q_d=\Pi_{\mathcal C_q}(q+J_{\mathrm{nom}}^+\Delta p+n\,v_N)$, $\Delta p=0.3\tanh(u_{1:2})$ m, $n=0.3\tanh(u_3)$ rad, $J_{\mathrm{nom}}^+$은 감쇠(0.02) pseudo-inverse, $v_N$은 kernel 방향. 강성 $K=K_{lo}(K_{hi}/K_{lo})^{\mathrm{sigmoid}(u_{4:6})}$, $K_{lo}=(50,25,10)$, $K_{hi}=(400,200,80)$. Gripper는 Bernoulli. 탐색 잡음은 10 결정(0.5 s) 동안 유지. 평가는 결정론적(평균 행동, logit > 0).

**손.** $u_{\mathrm{ad}}=G_W(q)^\top(\bar a+z)$, $G_W$: MLP $(\sin q,\cos q)$ 6–64–64–21 → tanh → 7×3, $\bar a=50\tanh(\cdot)\in\mathbb R^7$. 기억 $z\in\mathbb R^7$은 정지 지시자 $\zeta$(관절 속도 ≤ 0.3 rad/s가 0.1 s 이상)가 켜진 동안만 $\dot z=-\lambda_s G(G^\top G+\epsilon I)^{-1}\hat m$, $\hat m=u_{\mathrm{ad}}-\tau_{\mathrm{prev}}-\hat\tau_{\mathrm{ext}}$, $\lambda_s=5$, $|z|\le 50$.

**Role loss(손의 학습).** 기록된 rollout을 replay해 $\mathrm{mean}\,w(t)\|u_{\mathrm{ad}}-u^\star\|^2$, $u^\star=g(q)-d$(privileged). $w=1-e^{-\lambda_s r(t)}$, $r(t)$는 마지막 짐 변화(잡기, 놓기, 물체를 든 채 시작한 episode의 시작) 이후의 정지 누적 시간. 반복마다 Adam 16 step × 32 episode, lr $10^{-3}$, gradient clip 1.0.

**PPO.** $\gamma=0.99$(결정 단위), GAE $\lambda=0.95$, clip 0.2, value 계수 0.5, entropy 계수 $10^{-3}$, Adam lr $3\cdot10^{-4}$, gradient norm clip 0.5, 5 epoch, minibatch 4096 결정, batch마다 advantage 정규화. 반복당 128 episode(6144개 pool에서): 원래 시작 64, 단계 2 시작 32, 단계 3 시작 32(`restart=0.5`). L2의 모든 run이 이 hyperparameter를 썼고 학습률을 바꾼 적이 없다.

## 5. 경과 요약

| 단계 | 결과 | 기록 |
|---|---|---|
| M1, M2, L0 | 기구 검증과 부품 시험 통과(수정 반영). M2(Stage A)에서 과제·검증이 아이디어를 시험하지 못해 Stage B로 재설계 | spec `sec:m1`, `sec:m2audit`, `sec:l0` |
| L1 | 정해진 기준 뇌로 과제 타당성 확인, 13개 기준 통과(hash 5abe91f5) | `records/preregistration_l1.md`, `l1_results.json` |
| L2 첫 시도 (개정 13) | 반복 100에서 중단. 비용 결함(놓기 유인)과 role loss 귀속 결함 발견 | `records/l2_attempt1_plan.md` |
| L2 두 번째 시도 (개정 14) | 반복 300. 누르기 0%, (H1) 실패. 뒤 단계의 data 부족 진단 | `records/l2_attempt2_plan.md` |
| 개정 15 | 누르기 항 상한 1(과힘 놓기 유인 제거) | spec `prop:costprog` |
| Pilot R | 시작 상태 분포 0.5; 판정 규칙 (P) 실패로 그 pilot에서는 채택하지 않음. 분해에서 관절 공간 행동의 목표 방향 학습 신호가 없었음 | `records/l2_pilotR_plan.md` |
| Pilot T | task 공간 행동 채택(개정 16). 두 arm 모두 시작 상태 분포 0.5 사용 | `records/l2_pilotT_plan.md` |
| L2 세 번째 시도 | pilot T의 arm T를 반복 300까지(시작 상태 분포 0.5 포함). 결과는 6절 | `records/l2_attempt3_plan.md` |

## 6. 현재 상태 (반복 300, `records/l2_attempt3`)

**등록 기준** (spec `sec:l2a3`):
- (G) 충족: 반복 1–300 모두 관절 위반 0, 비유한 상태 0, 최소 tank-bound margin $2.6\cdot10^{-4}$ J.
- (H1) 불충족: 평가 260–300의 손 전이 잔차 평균 3.196 N·m(기준 1.768).
- (H2) 불충족: 기준 뇌 + 학습된 손의 등급 결정 0.922(기준 0.95).
- 중단 규칙 미적용: 처음 시작에서 든 채 힘 구간에 윗면을 누른 몫 0.688.

**뇌.** 단계 3 시작에서 정확히 누른 몫 0.641, 단계 2 시작의 유지 오차 0.0278 m, 처음 시작의 평가 비용 17.37(L1 기준 뇌 2.290). 잡기가 비용을 지배한다: 잡지 못한 19개(64개 중)의 평균 비용 37.460(spec `sec:l2status`). 누르는 힘은 등급과 무관(두 목표의 가중 평균). 결정론적 평가는 놓기를 과소평가한다(단계 4에 든 45개 중 0.089만 놓지만 표본 gripper로는 1.000).

**손.** prior는 학습됨. 자세 사이의 옮김(전이 잔차)은 반복 30 무렵 0.44에서 300에 4.43(중앙값)으로 표류. 접촉 step을 빼도 표류함(새 data 오프라인 실험, `records/l2_hand_fresh_b140.json`, `l2_hand_fresh_b300.json`). 해석: role loss는 정지마다 재보정되는 rollout 안의 오차만 재므로 옮김을 정하지 못한다(가설, spec `sec:l2a3`).

**장부 → 뇌.** 장부에는 등급 정보가 있으나(L1 규칙으로 0.911) 학습된 뇌는 쓰지 않는다(등급 결정 0.45–0.48).

**최적화.** 반복당 KL 중앙값 0.0555, clip 몫 0.40, goal 표준편차 0.211 → 0.0507, 훈련 비용 기울기 $-0.024$/반복(251–300). 수렴하지 않았다.

## 7. 계산 환경과 이식

**지금까지:** cloud container, CPU 2 core(Xeon 2.1 GHz), RAM 7 GB, GPU 없음. Python 3.13.18, jax/jaxlib 0.12.0.1(CPU), numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, Pillow 12.3.0, sympy 1.14.0, mujoco 4.0.1(참조 검증 `refs.py`에만), XeTeX(TeX Live 2023). 반복당 30 s(rollout 15.5 s). 세 번째 시도의 반복 101–300은 1.66 h(평가 포함 1.94 h). 모든 코드는 float64(`jax_enable_x64`).

**이식 순서 (승인 불필요; 학습하지 않는다).**
1. 의존성: `python3 -m venv .venv && . .venv/bin/activate`; `pip install numpy scipy matplotlib pillow sympy mujoco`; JAX GPU는 [JAX 설치 문서](https://docs.jax.dev/en/latest/installation.html)에 따라 CUDA 13 wheel(`pip install --upgrade "jax[cuda13]"`, Linux NVIDIA driver ≥ 580) 또는 CUDA 12 wheel(`"jax[cuda12]"`, driver ≥ 525). 문서는 RTX 50 시리즈를 따로 명시하지 않으므로 `nvidia-smi`와 `python3 -c "import jax; print(jax.devices())"`로 GPU가 보이는지 확인한다. 같은 버전(0.12.0.1)을 먼저 시도하고, 다르면 버전을 기록한다. TeX: `texlive-xetex texlive-latex-extra texlive-pictures fonts-noto-cjk fonts-dejavu poppler-utils`.
2. `cd code && python3 check_sync.py` → `sync findings: 0`.
3. `./build_tex.sh spec` → 게이트 모두 0, 67쪽.
4. `cd code/stageb && JAX_PLATFORMS=cpu python3 port_check.py` → 세 시작 모두 기록과 일치해야 한다(이 cloud에서는 episode별 상대 차이 0, 평균 비용 17.370263, 6.421145, 4.681267; `records/port_check_cloud.log`). CPU에서 다르면 버전 차이이므로 원인을 먼저 찾는다.
5. GPU에서 `python3 port_check.py bench=1 sizes=64,256,1024,4096`. GPU의 float64 결과는 연산 순서 때문에 CPU와 조금 다를 수 있다(스크립트가 EXACT/CLOSE/DIFFERENT로 판정). 참고로 이 cloud CPU에서는 64 episode 7.79 s/batch(1498 결정/s), 256 episode 26.99 s/batch(1716 결정/s)였다(`records/port_check_cloud.log`).
6. 결과(버전, 장치, 재현 판정, batch별 처리량)를 사용자에게 보고한다.

**알아 둘 점.**
- GeForce GPU의 float64 연산 처리량은 float32보다 크게 낮다. rollout은 작은 행렬의 긴 순차 scan(결정 220 × 제어 25 × RK4 4)이라 batch를 키울수록 이득이 난다. 실제 배수는 5의 측정으로 판단한다.
- float32 전환은 설계 변경으로 다룬다(승인 필요). spec `sec:envcheck`의 기록: float32 궤적은 float64와 관절 각 0.13 rad, 법선력 2.46 N(RMS) 이내로 일치했다. tank margin 판정(−0.01 J 기준)과 보장 검사에 미치는 영향을 먼저 확인해야 한다.
- 손의 role loss는 긴 scan을 역전파하므로 메모리를 많이 쓴다(`jax.checkpoint` 사용 중). batch를 키우면 VRAM(16 GB) 한계를 확인한다.
- `l2_train.py`의 반복당 episode 수는 세 번째 위치 인자(`N_EP`)이고, 손의 minibatch는 `min(32, N_EP)`이다.

## 8. 다음 단계 (제안; 각 항목은 사용자 승인과 사전 등록 후)

spec `sec:l2status`의 제안을 따른다. 사용자는 학습률 감소·scheduling도 검토하라고 했다(2026-11).
1. **이식과 재현** (7절; 승인 불필요).
2. **평가 규칙**: 표본 gripper 평가를 병기하고 판정용 평가의 episode 수를 늘린다.
3. **최적화 pilot** (반복 300 checkpoint에서 짧게): 반복당 episode 수(예: 128, 1024, 4096), KL에 맞춘 학습률(목표 0.01–0.02) 또는 고정 감쇠, entropy 계수. 손의 학습률 감쇠도 같은 틀에서 본다.
4. **손의 역할**: role loss에 옮김 항을 넣어 오프라인으로 먼저 시험(`l2_hand_fresh.py`의 틀, 통과 기준: 30 반복 동안 전이 잔차 < 1.768, role loss 증가 ≤ 10%).
   $$\mathcal L_{\mathrm{tr}}=\mathbb E_{a,b}\|G(q_b)^\top(\bar a+z_a^\star)-u^\star_b\|^2,\quad z_a^\star=G(q_a)(G(q_a)^\top G(q_a)+\epsilon I)^{-1}(u^\star_a-G(q_a)^\top\bar a)$$
5. **잡기 관문과 장부**: 3의 결과를 본 뒤 정한다(단계 1 비용의 속도 항 또는 물체 근처 시작 상태; 장부 하중을 공칭 기구학으로 손끝 하중으로 표현).
6. **Attempt 4**: 통과한 항목만 묶어 수렴까지. 이후 분포 밖 평가, (사용자 지시가 있으면) seed 여러 개, ablation, 7-DOF(Stage C).

## 9. 운영상의 함정

- **JSON 경쟁.** `l2_signal_check.py`, `l2_restart_behavior.py` 등은 기록 JSON을 읽고-고치고-쓴다. 같은 파일에 쓰는 프로세스를 동시에 돌리지 않는다.
- **Warm start.** `l2_train.py`에 checkpoint를 주면 행동 모드(`act`)와 optimizer 상태를 그대로 이어받는다(`opt_reset=1`로 초기화). 시작 로그의 행동 모드가 맞는지 확인한다.
- **평가의 놓기.** 결정론적 평가의 단계 4 비용은 정책보다 나쁘게 나온다(6절). 비교할 때 같은 평가 규칙끼리만 비교한다.
- **전이 잔차는 한 평가에서 두 배까지 흔들린다.** 판정은 여러 평가의 평균으로 한다(세 번째 시도의 H1 규칙).
- **오프라인 손 실험**은 data를 만든 손과 학습 중인 손이 다르다(replay가 기록된 torque를 씀). 결론은 같은 data의 변형끼리 상대 비교로만 낸다.
- **시간 제한.** 긴 run은 `nohup`/`setsid`로 띄우고 `train.log`의 `EVAL it N:` 줄로 진행을 본다. `report_checkpoint.sh`는 그 줄을 기다렸다가 보고물을 만든다.
- **figure 이름**은 `fig_<what>.py` → `figures/<what>.png`(가이드라인 E8). GIF는 `anim_rollout.py`가 run 이름으로 만든다.
- **기록이 우선.** 문서와 기록이 어긋나면 `records/verified_values.md`의 `## Divergence (open)`에 즉시 적고, 정합시킨 뒤 지운다. 전달 시점에는 비어 있어야 한다.

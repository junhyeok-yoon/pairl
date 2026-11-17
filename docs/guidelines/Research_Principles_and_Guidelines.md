# 연구 원칙 노트 — Constrained PI Servo Control (항상 준수)

> 이 파일은 프로젝트 read-only 원본의 사본이며, 2026-07-30 세션의 회고로 **K7b·K8·D11 신설, K7·K3·§6 확장** 네 곳이 반영돼 있다. 컨테이너·코드베이스 고유의 함정은 가이드라인이 아니라 번들 `README.md`의 `## Gotchas` 절에 둔다.

이 문서는 이 연구 --- 주어진 input/output constraint를 만족(IO constraint satisfaction)하면서 모든 signal이 bounded(boundedness)이고 항상 feasible한 CBF-QP reference governor를 갖는 **PI servo control** --- 에서 변하지 않는 목표·원칙·규칙·금지사항을 정한다. 모든 작업(연구·증명·코드·문서)은 이를 준수한다.

> **`.tex` 또는 `.pptx`를 작성·수정하기 전에 §5a(공통 작성 원칙)를 반드시 먼저 정독하고, 이어서 해당 산출물의 전용 절(`.tex`는 §5b, `.pptx`는 §5c)을 정독한다.** 특히 작성 전에 이전·현재 대화의 관련 지적·피드백을 직접 조사·검토하여 어떤 general한 원칙이 요구되어 왔는지 스스로 파악한 뒤, 새로 쓰는 모든 block에 적용한다.

> **발표자료(`.pptx`)는 구성안 `.tex` → 승인 → pptx 2단계로 만들고(§5c M1), 새 deck은 §8·발표 template 골격에서 시작한다.**

> **발표자료(`.pptx`)를 작성·수정하기 전에 §5c(`.pptx` 전용 규칙)를 반드시 먼저 정독한다(그 전에 공통 §5a도).** 마찬가지로 작성 전에 이전·현재 대화의 슬라이드 관련 지적·피드백을 직접 조사·검토하여 어떤 원칙이 요구되어 왔는지 스스로 파악한 뒤 적용한다.

---

## 1. 연구 목표·목적·방향

- **무엇을**: 주어진 **input/output box constraint를 만족**하면서 **모든 signal이 bounded**이고, **항상 feasible한 CBF-QP reference governor**를 갖는 **PI servo control**을 **FORMAL·NON-CIRCULAR·a-priori·(가능한 경우) closed-form**으로 확립. 핵심 세 성질은 **IO constraint satisfaction · boundedness · feasibility**. priority class = **stable, n-th order, SISO**; MIMO·less-conservative는 확장 목표. (Saturation-aware 경우 SAG, saturation 없는 pure CBF-QP governor 경우 PI_CBF가 각각의 instance.)
- **왜**: framework의 feasibility·practicality 입증. 위 세 성질(특히 a-priori feasibility·boundedness)이 그 핵심 증거.
- **가치 보존**: 핵심 가치는 **myopic QP의 runtime 단순성 + a-priori 검증 가능성**. certificate는 **closed-form 또는 simulation-free offline**이어야 한다. `O_infty`(iterative polytope)는 이 단순성을 훼손하므로 **dispreferred**(필요 시 offline 분석 도구로만).
- **NO FALLBACK**: stated framework(CBF-QP reference governor) 밖의 장치(예: `y_g` clamp)는 실용 경로 언급만, 증명 대상이 아니다.
- **benchmarks**(DI, CE1, CE2, CE3, Aircraft, X-29A, HM-A/B 등)는 **TEST**이지 certification target이 아니다.

---

## 2. Framework·scenario 출처

구조·정의·표기·convention, 그리고 각 scenario의 plant·command·constraint·gain·constant·relative degree는 모두 project file에 있다. 본 노트에서 재서술하지 않고 그 내용을 직접 확인할 것 (§7): framework·분석은 `SAG_V1s2`, `PI_CBF`, `SAGAF`, scenario(simulation test pool)는 `scenarios.tex`.

---

## 3. Settled 사실 (확립된 결과)

**표기**: `H_u`(high-frequency gain matrix)는 vector relative degree로 정해진다 — 채널 `j`(`C_lim,j` = `C_lim`의 `j`번째 행)의 relative degree `r_j := min{k≥1 : C_lim,j A_p^{k−1} B_p ≠ 0}`에 대해 `H_u`의 `j`번째 행은 `C_lim,j A_p^{r_j−1} B_p`이다. 즉 `H_u`는 vector relative degree `(r_1,…,r_m)`에 따라 달라지며, `H_u = C_lim B_p`는 모든 `r_j=1`인 특수형일 뿐이다(uniform `r_j≡r`이면 `C_lim A_p^{r−1} B_p`). `G_0 := −C_lim A_p^{−1} B_p`는 DC gain(별개).

1. `x_p`는 무조건 ISS-bounded이다: `A_p` Hurwitz + `|sat(u_bl)|≤u_M`. 따라서 발산 가능한 state는 `e_I`뿐이다.
2. `(K_I,K_bc)`가 matched(`P·K_bc=K_Iᵀ·T`, `P≻0`, diagonal `T≻0`)이면 `V_I=e_IᵀPe_I`에 대해 `V̇_I ≤ 2e_IᵀP(z−y_g) − 2λ_min(T)‖σ‖² + 2‖TK_P‖‖x_p‖‖σ‖`. matching은 free `K_bc`로 항상 달성된다.
3. Deadzone sector: `σᵀT(u_bl−σ) ≥ 0`, `‖σ‖ ≤ ‖u_bl‖`. Coercivity: `‖σ‖ ≥ c_1‖e_I‖ − const`.
4. `r_Y := dist(0,Y)` (infeasibility radius)가 `sup_t` bounded이면 `(x_p,e_I)`가 bounded이다. 단 `r_Y`는 `e_I`가 bounded인 채로도 발산할 수 있어 boundedness의 필요조건은 아니다.
5. plant가 minimum-phase이면 `sign(G_0)=sign(H_u)`. 정규화로 `G_0>0`.

---

## 4. 하지 말아야 할 것 (DO NOT)

1. **Half-1/Half-2(또는 inner/outer) 분해 금지.** `σ`가 anti-windup과 governor로 동시에 분기하여 분리 불가이고, governor가 binding하면 상쇄된다. single-inequality로 다룰 것.
2. **순환 논증(circular reasoning) 절대 금지.** closed-loop reachability/boundedness 가정을 certificate에 쓰지 말 것. plant-only/frozen-flow는 허용.
3. **fallback(pure-projection 밖 장치) 재추가 금지.** clamp 등은 실용 경로 언급만.
4. **SAG 구조 변경/다른 scheme로 pivot 금지** (SAG가 근본적으로 틀리지 않는 한). ERG·GdS–T 등은 **분석 도구로만** 차용하고 governor는 myopic QP 유지.
5. **`O_infty`-style iterative polytope 계산 dispreferred.** closed-form/sim-free 강하게 선호.
6. **검증 없는 from-scratch 시뮬레이션으로 정량 주장 금지.** (단 LMI feasibility, vector-field point evaluation, symbolic check는 신뢰 가능 → 적극 사용.)
7. **trivial diagonal example 사용 금지** (구조적 현상을 숨김).
8. **novelty/publishability 주장 전 반드시 web-search.** training knowledge만으로는 불충분.
9. **citation 부정확·추정 금지.** (참고: ERG = Garone–Nicotra IEEE TAC 61(5):1379–1384, 2016, arXiv:1406.1738; Nicotra–Garone CSM 2018, DOI 10.1109/MCS.2018.2830081.)
10. **문헌 결과 과대 차용 금지.** 차용 전 SAG 구조 적합성을 확인할 것. 특히 governor reference가 **exogenous**임을 전제하는 결과(예: 표준 anti-windup의 `L2`-gain 분해)는 `y_g`가 **state-dependent**인 SAG에 직접 적용되지 않는다.
11. **regional 결과의 region이 vacuously narrow하면 "empty theory"** — 그대로 두지 말고 극복을 시도하고, 닫히지 않으면 정직히 open으로 남길 것.
12. **established 사실 재유도 금지** (§3은 settled, cross-reference로만).
13. **standing document에 meta·editorial commentary 금지.** 본문·remark·정의 등에 design-rationale, 표기 선택에 대한 editorial 설명(notation이 우연이라거나 혼동 주의 류), 작가적 주석을 넣지 말 것. 명료성은 정의·전개 자체로 확보한다. 규칙 B1의 display-중심 전개는 D3·B4에 따라 서술 전반의 기본값이며, 이 항목(meta 금지)은 그와 독립적으로 문서 전반(remark 포함)에 적용된다.
14. **author-internal working label 금지.** 'movement N' 같은 작가 내부 작업용·임시 명칭을 문서에 쓰지 말 것. 실제 내용 이름이나 section 참조로 지칭한다(예: 'movement 1의 권한 손실' → '기존 SAG의 권한 손실', '(movement 4)' → '(\S\ref{...})').

---

## 5. 원칙 (PRINCIPLES)

- **할 때 잘 하자**: 진행 전 항상 **cold rigorous self-check + numeric sanity-check**. 자기 결과를 의심하고 교차검증한다(에러를 나중에 잡기보다 애초에 안 만드는 것이 우선).
- **반증되면 최소 결정 실험으로 근원을 확정한다**: 가설·주장이 수치로 반증되면(재계산 값과 문서 값의 불일치 포함) 표현을 바꿔 재주장하지 않는다. 후보 원인들을 가르는 최소 실험을 설계·실행해 근원을 짚은 뒤에 진행한다.
- **lever 평가는 이중 보고**: 설계 lever의 잠재력은 이상화 상한(가상 입력·정보 무제한 가정)과 구현 경로의 실제 달성치를 구분해 함께 제시한다. 상한만 보고하면 과대평가가 후속 판단을 오염시킨다.
- **impossibility/negative 결과도 가치 있다**: 불가능하면 틀린 답을 만들지 말고 왜 불가능한지 설명한다.
- **Occam's razor**: clean·simple을 추구하고, 여러 결과를 하나로 통합하는 방향을 선호한다.
- **중심 유지**: 목표는 §1의 것 --- constrained PI servo control(항상 feasible한 CBF-QP reference governor)의 **IO constraint satisfaction · boundedness · feasibility** 확립. 흥미로운 방향이 보이면 무시하지 말고 아이디어·건의를 먼저 제시하되, 사용자 승인 없이 자체적으로 그 방향에 큰 시간·자원을 소모하지 않는다.
- **제안/실행 분리**: '선택' 표시 항목과 구조 변경(section 신설·삭제, 정리 재배치, 대규모 이동)은 제안 후 사용자 승인 전에 실행하지 않는다. 승인된 목록만 실행하고, 실행 중 발견한 인접 문제는 D8의 범위 판단으로 함께 고치되 보고에 명시한다.
- **현황 파악 후 연구**: §7의 project files에 설명된 여러 텍스트 문서를 먼저 읽어 현재 연구의 상황·흐름·상태를 잘 파악한 상태에서, 사용자 지시에 따라 연구를 수행한다.
- **Cross-architecture failure는 duality로 진단**: 같은 instance에서 서로 다른 controller를 비교할 때, 어느 constraint가 hard-enforced이고 어느 것이 sacrificed인지 먼저 식별한다. reference governor(입력 box 지킴 → output 안전·입력 위반)와 ACC-style saturation(입력 sat 지킴 → output 위반)은 같은 근본 한계(bounded 입력이 blind-ray drift를 못 막음)의 dual 표출이다 — 두 실패를 별개 현상으로 서술하지 말고 공통 원인으로 묶는다.
- **Downstream artifact freshness**: source(`.tex` 등)를 수정하면 파생 산출물을 전부 재생성(`.tex`→PDF 3-pass)하고, 전달하는 산출물이 timestamp상 source보다 최신인지 확인한 뒤 전달한다. global 편집(용어 일괄 교체 등)은 stale instance가 0개임을 `grep -c`로 확인한다. "source 고침 ≠ 산출물 최신".
- **Rendered-output 검증**: source가 맞다고 끝내지 말고 실제 렌더된 수치·label·축 범위·요소 가시성을 확인한다(비정수는 `%g`, 색은 배경에서 보이는지). "코드 맞음 ≠ 그림 맞음" — 그림/PDF는 `pdftotext`/`pdftoppm`이나 픽셀 점검으로 실제 출력을 본다.

### 5a. 공통 작성 원칙 (`.tex`·`.pptx` 모두 --- READ FIRST, EVERY TIME)

> **이 절은 `.tex` 또는 `.pptx`를 작성·수정하기 전에 반드시 정독한다.** 여기 담긴 내용-엄밀성·가독성·figure 원칙은 논문(`.tex`, §5b)과 발표자료(`.pptx`, §5c) **양쪽에 공통으로** 적용된다. 이 가이드라인을 따르지 않아 같은 종류의 지적이 반복되는 것은 가장 큰 시간 낭비다. 작성 전에 다음을 수행한다: (1) 아래 cluster A–D와 figure 규칙(E-fig)을 항목 단위로 숙지한다. (2) **이전·현재 대화에서 사용자가 준 작성 관련 지적·피드백을 직접 조사·검토한다** --- `conversation_search`/`recent_chats`로 과거 세션의 `.tex`·슬라이드 피드백을 찾고 현재 대화의 피드백도 다시 읽어, 그 구체적 지적들이 어떤 general한 원칙을 요구하는지 스스로 파악한다. 사용자가 같은 지적을 반복하지 않도록, 작성 전에 이 검토를 반드시 마친다. (3) 그렇게 파악한 기준과 아래 cluster를 새로 쓰는 모든 block에 적용한다. "읽었다"가 아니라 "출력이 이 기준을 만족한다"가 통과 조건이다. `.tex` 전용 규칙은 §5b, `.pptx` 전용 규칙은 §5c에 있으며, 그 절들은 이 §5a에 *추가로* 적용된다.

#### A. Statement & Definition Writing

- **A1. Symbol definitions.** Every symbol is defined at first use with `:=` and its dimension / space / role, inside a `definition` or `assumption` environment. Prose-only definitions ("the flow of ...") are not allowed. A definition's opening must make its *definiendum* (the term being defined) immediately identifiable; when the defined term is not a noun but a verb, adjective, or adverb (deployable, feasible, invariant), set it in `\textbf{\emph{...}}` so that what is being defined is visible at a glance. Ordinary descriptive nouns (output box, input box) need no such emphasis.
- **A2. Multi-item definitions/assumptions as line-broken lists.** Any `definition` or `assumption` carrying two or more conditions or symbols is written as an `itemize`/`enumerate` with each item on its own line, never as a single prose paragraph. Label assumption items explicitly (e.g. `\item[\textbf{(A1)}]`).
- **A3. Statement contains the claim only.** A `theorem`/`lemma`/`proposition` statement holds only its hypotheses and result. Analysis, motivation, and "why" go into the proof or into interpretation prose — no explanatory prose inside the statement.
- **A4. State the scope in the statement.** Each result names its scope where it is stated: global / semi-global / regional, unconditional / IC-conditional, exact / limit. The reader should not have to infer scope from the proof.
- **A5. Formality (all formal environments).** Every design condition, assumption, lemma, proposition, theorem, corollary, and proof is written *formally and completely*: exact hypotheses, exact objects, and what is quantified over (∃ for design variables, ∀ for operating points). A lone inequality or a vague phrase does not substitute for a complete statement --- an incomplete statement is treated as dishonest, effectively an error. Proof form follows cluster B (B1). Do not use a bare colon ':' to carry logical implication: a statement written as "hypothesis : conclusion" reads ambiguously, so write the implication explicitly with `\Rightarrow` or as a fully quantified sentence, so the logical structure is unambiguous.
- **A6. Multi-result statements as enumerated lists.** A `theorem`/`lemma`/`condition` asserting several distinct results --- the bound on each signal, a sign split across regions, existence plus characterization --- lists them as `enumerate[label=(\alph*)]`, one result per item, never a prose run-on. A boundedness theorem in particular enumerates *which* signals are bounded and *by what* constant, since the bounded values are the content. Nested two-way splits use an inner `itemize`.
- **A7. 자체 정의 용어·과잉 표기를 만들지 않는다 --- 표준을 우선하고 필요할 때만 명명한다.** 새 명칭(coined term)·새 기호는 독자에게 학습·조회 비용을 지우므로 최소화한다. (i) 표준·확립된 용어가 있으면 그것을 쓰고 자체 명칭을 만들지 않는다. (ii) 새 named term·기호는 그 대상이 **반복 등장**하고, 이름을 붙이는 편이 매번 풀어 쓰는 것보다 명백히 읽기 쉬울 때만 도입한다 --- 한두 번 쓰이는 집합·양은 명명하지 말고 식을 inline한다. (iii) 이는 A1·H1(쓰는 모든 것은 정의; 간결함이 정의 생략의 핑계가 아님)과 상충하지 않는다: A1·H1은 *사용하는* 것을 정의하라는 rigor이고, 이 항목은 *정의가 필요한 대상 자체를 economy 있게* 유지하라는 것이다(정의를 빼는 것이 아니라 애초에 붙일 필요 없는 명칭을 만들지 않는 것). \S4.14(author-internal working label)가 임시 작업명 금지라면, 이 항목은 내용상의 neologism·과잉 notation 전반에 적용한다(슬라이드 포함 --- \S5c 서두).

#### B. Proof Writing

- **B1. Equation-forward.** Proofs are written as a sequence of displayed steps, not as prose. The goal is for the reader to follow each step's mathematics at a glance.
  - **One display = one claim.** Do not pack distinct equalities or implications into one `\[ \]` with `,`, `\qquad`, or chained `\iff`. Each step gets its own display.
  - **Implications (`⟹`/`⟺`) on their own line.** `A ⟹ B` splits into `A` as one display and `⟹ B` as the next.
  - **Each step is preceded only by a short action label** ("complete the square", "fix `P`", "minimize", "block lower-triangular"). Do not connect steps with prose paragraphs ("therefore ...", "since ..."). The proof-prose-in-Korean default (E1) applies to these short labels, not to walls of prose. These step/action labels ("Step 2 (coercivity and activation).", "Mirror symmetry.") are set in `\textbf` (not `\emph`/`\textit`), consistent with the D5 topic-lead, so the step structure is visible when skimming the proof. Bold step labels are reserved for genuinely long proofs or explicit case splits: in a short proof (a handful of displays), bold labels add visual noise rather than structure and must be omitted, letting the displays alone carry the logic.
  - **Label minimalism.** Use a label only where the step genuinely changes task. Do not add a label that merely restates the line's content in one word ("Monotone." before a monotonicity line, "Endpoints." before two limits); when the displays already make the logic visible, omit the label. Labels mark task boundaries, not every sentence --- gratuitous one-word labels hurt more than help.
  - **Case splits use the `cases` environment.** Do not bundle multiple steps into one list-style `aligned`.
  - **Keep normal math spacing** (no `{=}`/`{-}` spacing compression).
  - **Forbidden:** (a) cramming steps into one `align`/`aligned`; (b) dissolving the derivation into prose just to cut length.
  - **Form to follow:**
    ```latex
    Complete the square in S.
    \[ g(S)=\norm{\nu_1}^2(S-S^\star)^2+\mathrm{const},\qquad S^\star=-\inner{\nu_2}{\nu_1}/\norm{\nu_1}^2. \]
    \[ \Longrightarrow\quad \min\norm{\rho_\perp}\iff\min|S-S^\star|. \]
    ```
- **B2. Conclusion equations are displayed, not inline.** A step's actual conclusion --- a sign relation, a bound, the target inequality --- is a displayed equation on its own line, never buried mid-sentence in prose. Inline math is only for symbols referenced in passing; any equation that *is* the point of the step gets its own display (e.g. `\sgn(\sigma)=-\sgn(e_I)`; the drift bound `\sgn(e_I)\dot e_I\le-\lambda|e_I|+D`).
- **B3. Quantifier symbols over prose.** In statements and conditions prefer `\forall,\exists,\exists!` to "for all / there exists / there is a unique" (`$\exists\,\sigma_{\mathrm{shell}}>0,\ \rho_0>0$ such that`, not "there exist a radius and a margin such that"). Drop a prose clause when the symbolic form already carries it. Use quantifiers actively: a fully quantified statement is not merely more formal but easier to read fast, because the symbols make the logical skeleton --- which objects are given, which are asserted to exist, over what range a property holds --- visible at a glance, whereas the prose equivalent buries that structure in words. Formality here serves comprehension, not the reverse.

- **B4. Display-forward beyond proofs.** Proof 밖의 statement 본문·remark·직관 prose에서도 그 문단의 요지인 관계식·결론 수식은 중앙 display로 올린다(B2의 전면 확장). 문단 중간 inline에 묻힌 수식이 곧 그 문단의 결론이라면 display 대상이고, inline에는 지나가며 참조하는 기호만 남긴다.

#### C. Scope & Claim Honesty

- **C1. Accurate labels.** Distinguish exact vs limit/approximation; mark every bound conservative / tight / over-approximation; state iff vs sufficient-only precisely. Neither over-claim nor under-claim. **Region scope of a bound:** when a bound holds only on a sub-region, state the region and do not present it as global; if the global claim needs a separate argument, name that dependency rather than gloss it (`y_g` bounded *on the shell* via the floor, with the complement's separate route stated, not a blanket "`y_g` bounded").
- **C2. Dependency map.** If a result uses only a subset of the standing assumptions, say which ones. Provide a dependency table or a per-assumption role note so every guarantee is traceable to its hypotheses (e.g. "(A4) is used by boundedness only; IO satisfaction does not use it").
- **C3. State exogenous-signal assumptions explicitly.** Where a result relies on an exogenous input (command, disturbance), state the assumption it uses (boundedness, regularity). If a result is independent of a particular exogenous signal, say so explicitly (e.g. "feasibility does not depend on the command; it enters the objective only, not the constraints").
- **C4. One-directional dependency for mutually-referencing results.** When two results reference each other (feasibility ↔ boundedness), break the loop in the text with an *a priori constant set* (defined from plant data, not from the other result) and state the ordering explicitly, so no proof silently assumes the other's conclusion. Make the closing mechanism visible (e.g. forward invariance of the a priori set bootstraps the two). This is the writing-level counterpart of the non-circularity rule (§4.2).

#### D. Structure & Readability

- **D1. Causal ordering.** Definition → use is one-directional; no forward-reference. Insert a new result at its dependency position, not appended at the end.
- **D2. No redundancy.** State each thing once; handle symmetric or repeated cases with "by symmetry / mirror"; cross-reference instead of re-deriving.
  - **(a) No prose echo of a display.** A sentence that restates what the adjacent equation directly says ("the deficit grows at least linearly with the error" after `|\sigma|\ge K_I|e_I|-c_0`; "a half-line bounded only from below" after `\{y\ge\underline y\}`) is redundant with the display; delete it. Prose next to a display adds the *why* or the *consequence*, never re-reads the symbols.
  - **(b) Extract a repeated derivation once.** If the same computation appears in two proofs, pull it into one lemma/definition and cite it; never re-derive. A scalar coefficient reduction used by both a structural lemma and the main theorem belongs in a single lemma, with both sites citing it.
- **D3. Concise = 재구조화이지 축약이 아니다 --- proof를 넘어 문서·채팅 전체 적용.** "줄여라/간결하게/concise" 지시를 분량 삭감으로 해석하지 않는다. 그 의미는: (i) redundant·불필요한 설명의 제거(D2), (ii) 전체 설명을 이해하기 쉽게 재구조화, (iii) 핵심 수식을 중앙 display로 올리고(B2·B4) 줄바꿈으로 논리 단위를 분리해 설명의 구조가 저절로 보이게 하는 것이다. 골격·step-label을 먼저 세우고 filler를 제거하되 핵심 유도는 단계별로 펼친다(B1). 길이와 명료성이 충돌하면 명료성을 택한다. 이 원칙은 proof에 한정되지 않고 statement 본문·remark·직관 prose·채팅 설명 등 모든 서술에 적용한다.
- **D4. No redundant separators** (`###`, `---`).
- **D5. Section intuition prose — show the forest.** B1 handles the trees (individual derivations); each section additionally carries intuition prose that shows that section's forest. Place it just before the section's main result (lead-in) or just after (interpretation), as `\textbf{직관 (...)}` / `\textbf{해석 (...)}` --- the paragraph-lead topic sentence is set in `\textbf` so that skimming reveals what the paragraph is about (more skim-readable than `\textit`/`\emph`). Write it as prose (no equation lists) covering what the section does and why, and its core mechanism / trade-off. Keep intuition and motivation out of the proof body (rule B1): forest is prose, trees are equations. Intuition also honors scope (rule C1). 직관 prose에도 §6의 설명 규율이 적용된다: 비표준 술어·비유를 극도로 자제하고, 자체 정의 용어 대신 표준 용어를 쓰며(A7), 핵심 주장은 인접 statement·display의 수식을 가리키는 방식으로 진술한다.
- **D6. Comparison in one place, both-sided.** Comparison against a baseline lives in one dedicated section, on the shared analysis object, and states both the new method's gains and the points where the baseline is cleaner or equivalent (both-sided honesty), rather than scattering comparison remarks across sections.
- **D7. Read as a first-time reader before finalizing.** Clarity is verified, not assumed. Before treating any block as done, re-read each definition, statement, and proof as if encountering it for the first time, and ask whether its content lands in one pass: does a statement's hypothesis-and-conclusion structure show at a glance, does a definition reveal what it defines, does each sentence of prose carry information the adjacent mathematics does not already state. Prose that merely restates a neighboring equation, that adds polite but empty framing ("the command enters only through ...", "the loop reduces to ..."), or that explains the author's intent rather than the mathematics, is removed (this is the readability counterpart of the no-meta rule §4.13). This re-reading is applied uniformly to every newly written block --- not only to the passage currently under revision --- because a correction made in one place is a correction to a *kind* of writing, and the same standard governs every section, old and new; a fix that is patched locally but not generalized will resurface in the next section and force the same feedback again.

- **D8. 지시는 그 종류(kind)에 적용한다 --- local fix가 아니라 일반화.** 사용자가 특정 슬라이드·특정 `.tex` 구절에 수정을 지시하면 그 지시를 문자 그대로 한 곳에만 이행하지 말고, 같은 원칙이 걸리는 다른 슬라이드·다른 `.tex` 구절·다른 섹션에도 동일하게 적용되는지 확인한 뒤 함께 수정한다. 하나의 수정은 한 *종류*의 글쓰기에 대한 수정이므로, local로 땜질하고 일반화하지 않으면 다음 자리에서 같은 지적이 재발한다(D7의 "correction to a kind" 정신을 지시 대응 전반으로 확장). 사용자가 매번 "비슷한 지시가 다른 곳에도 적용되는지 확인하라"를 덧붙이지 않아도 되도록 이를 기본값으로 삼는다. 단, 일반화 범위가 넓거나 판단이 갈리면 먼저 후보를 제시하고 확인받는다.

- **D9. 완료 보고 전 self-audit (빌드 게이트 ≠ 완료).** `.tex`/`.pptx` 작업을 "완료"로 보고하기 전에, 빌드 게이트 통과와 별개로 서술 품질을 항목별로 자체 검증한다. 빌드 통과(errors 0, ref_undef 0, overfull 등)는 필요조건일 뿐 완료가 아니다. 새로 쓰거나 고친 **모든** 명제·증명·prose를 아래 cluster에 하나씩 대조한다: (A3) statement에 claim 외 서술이 없는가, (A5) 명제가 완전·형식적인가, (B1) proof가 줄글이 아니라 display 연쇄인가, (A1) 모든 기호가 첫 사용에서 정의·차원 명시됐는가, (D5) 흐름 prose가 lemma 앞에 있어 정의·명제가 갑작스럽지 않은가, (A7·H7) 비표준 용어·약어·자체 명명이 없는가, 한 문장이 과도하게 길지 않은가. 하나라도 미달이면 "완료"라고 보고하지 않고 먼저 고친다. "게이트를 통과했다"는 완료 근거가 아니다 --- 사용자가 보는 것은 게이트가 아니라 서술 품질이다.

- **D10. 품질 지적은 재발 방지를 전수 확인한다.** 사용자가 서술 품질(줄글 과다, formal하지 않음, 흐름 부족, 용어 등)을 지적하면, 그 지적을 반영한 뒤 **같은 종류의 문제가 문서 전체에 남아 있지 않은지** 전수 확인한다(D7·D8의 완료-판정 확장). 특정 lemma 하나를 고쳤으면 나머지 모든 lemma·prop·정의를 같은 기준으로 점검한다. "이 부분을 고쳤다"가 아니라 "이 종류의 문제가 문서에서 0이다"가 완료 조건이다. 사용자가 "아직도 대부분 그렇다"를 반복하지 않도록, 지적 반영은 항상 그 종류 전체에 대한 sweep으로 마무리한다.

- **D11. 지시 범위 안의 공백은 같은 턴에 닫는다.** 감사·대조로 발견한 공백이 현재 지시의 범위 안이면 현황 보고로 끝내지 말고 그 턴에 메운다. 보고만 하는 것은 범위 **밖**이거나 사용자의 판단이 필요한 경우로 한정하고, 그때도 무엇을 하지 않았는지 명시한다(D8의 일반화가 *어디까지*라면, D11은 *보고할 것인가 고칠 것인가*다).

#### E. Figure 생성·관리 (`.tex`·`.pptx` 공통 자원)

> figure는 논문과 발표자료가 함께 쓰는 공통 자원이므로, 그 *생성 코드*의 규칙은 여기 공통 절에 둔다. figure가 `.tex`/`.pptx` 안에서 어떻게 캡션·배치되는지는 각 전용 절(§5b E-tex, §5c J)에 있다.

- **E7. Figure는 그 `.tex` 또는 `.pptx` 설명에 복무한다 --- 이론을 먼저 읽고 구현하며, 산출 수치를 검증한다.** figure를 만들거나 고치기 전에, 그 figure가 놓일 위치의 정리·본문(무엇을 시각화해야 하는가)을 먼저 정독하고, 궤적·축·요소를 그 이론(예: lemma의 분해식, envelope의 정의)에서 유도한다. figure는 그 이론적 서술을 정확히 시각화하는 것이 목적이다. **기존 그림에 픽셀을 맞추는 방식으로 만들지 않는다** --- 픽셀 일치는 부수적이고, 정확성은 이론 부합과 산출 수치로 판정한다: figure가 담는 정량값(dip, reach, $\nu$, $W$ 등)은 `.tex`/`.pptx` 본문이 그 figure를 인용하며 쓰는 수치와 일치해야 하며, figure 수정 후 스크립트 출력의 핵심 수치를 대조 검증한다("코드 맞음 $\neq$ 그림 맞음"의 수치판, Rendered-output 검증의 확장). 불일치는 stop signal이며 픽셀이 아니라 root cause(config·IC·데이터 의존성)를 고친다. (caption 자체는 제목만 담는다 --- E4.)
- **E8. 명명 --- 이미지 파일명 = 스크립트 파일명(`fig_` 제거).** figure 생성 스크립트는 `fig_<무엇을_그리는가>.py`로, 산출물은 스크립트명에서 `fig_`를 뗀 `<무엇을_그리는가>.png`로 하여 **파일명이 1:1 대응**한다. 이름에 숫자·개발 순서·약어를 넣지 않는다(`step8`, `figs2`, `omega3d`, `nu_mg` 금지 → `operating_set`, `mg_saturation`). 한 스크립트가 논리적으로 묶인 여러 산출물을 낼 때만 내용 접미사로 구분한다(`finf_boundary_siso`/`_mimo`, `feasible_anatomy_empty`/`_set`, `case_ex1_containment`). 이는 `.tex` 안의 caption(E4)과 다른 층위다 --- E8은 *파일명*, E4는 *그림 설명*.
- **E9. 형식 --- Python으로 생성하는 모든 figure는 PNG.** PDF를 섞지 않는다. TikZ figure는 스크립트로 만들지 않고 `.tex` 내부에 둔다(E2·E5). 이 코드 규칙은 Python 생성 figure에만 적용한다.
- **E10. 공용화 임계 --- 2회 이상만 `lib_`으로, plant는 `data_`로 single-source.** 어떤 계산이 **2개 이상** figure 스크립트에서 쓰이면 `lib_<기능>.py`로 분리하고, 1개만 쓰면 그 스크립트 안에 둔다. **`lib_`을 선험적으로 설계하지 않는다** --- 실제 공유가 확인된 것만 사후에 뽑는다(과잉 추상화 금지). plant 데이터·gain·인증값은 `data_<plant>.py`가 한 번 계산해 `.npz`로 저장하고 figure는 로드만 한다; plant 값은 `data_*`에서만 정의한다(single source of truth). `lib_`은 관심사별로 나눈다(style / governor QP / reach-set 기하 / simulation). 특정 figure 전용 무거운 계산(sweep 등)은 `_<이름>.py` helper로 둔다. 재현은 `build_all.py`(data_* → 모든 fig_*)로 한다.

### 5b. `.tex` 전용 규칙 (논문 작성 시 --- §5a에 추가 적용)

> **§5b는 §5a 공통 원칙에 *추가로* 적용되는 논문(`.tex`) 전용 규칙이다.** 언어 convention, PDF 렌더링, 논문 figure의 캡션·logic-chain 배치를 정한다. 내용-엄밀성·가독성(cluster A–D)과 figure 생성 규칙(§5a E)은 여기서 반복하지 않고 §5a를 따른다.

#### E-tex. 언어·렌더링·논문 figure 배치

- **E1. Language convention.** Statements of `definition`/`theorem`/`lemma`/`proposition`/`corollary`/`assumption` are written in English; `proof`/`remark` body prose is written in Korean (한다체, no honorifics, no 반말); technical terms stay in English everywhere. No 한자, no emoji. **Override:** if the user explicitly asks for a proof, remark, or section in English, that request takes precedence over the Korean-prose default for that material, and the affected proofs/remarks are written in English instead.
- **E3. PDF rendering (fontspec internal, kotex for delivery).** When I render the PDF, use `fontspec` internally so both English and Korean display correctly (e.g. `\usepackage{fontspec}` + `\setmainfont{Noto Serif CJK KR}`, XeLaTeX three-pass). When delivering the `.tex` to the user, swap those fontspec lines for `\usepackage{kotex}` so the user can render with kotex directly. Verification: zero undefined `\Cref`, zero duplicate labels, zero missing characters; check Korean actually renders via `pdftotext`/`pdftoppm` (never substitute a KO-transliterated struct PDF — Korean must actually appear).
- **E2. Overview figure (always the first section).** A boundedness-proof document opens, immediately after the abstract, with `\section{Overview...}` containing a tikz logic-chain summary figure that shows the proof's dependency chain and core logic very simply.
  - All text inside the figure (box text, arrow labels, title) is in English, regardless of the Korean prose convention.
  - The flow (ordering) must be visible: which signal is bounded in which order, as an arrow chain (e.g. `u → x_p (∵ Hurwitz ISS) → z,y → e_I → y_g → all bounded`), with each arrow labeled by the reason for that step.
  - Distinguish unconditional vs conditional steps by color (e.g. governor-independent unconditional = green, conditional = orange), and mark in the certificate box the non-circularity argument and the sharpening location.
  - Draw with `\usepackage{tikz}` + `\usetikzlibrary{arrows.meta,positioning,calc}`, pinned in place via `\usepackage{float}` `[H]`.
- **E4. Figure captions hold the title only.** Every figure caption holds only the figure's *title (name)* and nothing else. What the figure shows, the color/symbol legend, how to read it step by step --- all such explanation moves to the **body prose** that references the figure (or is deleted if redundant). This applies to **every figure**, including the overview logic-chain (rule E2). A caption is a label, not a paragraph. **The caption is always written in English, regardless of the Korean prose convention.**
- **E5. Every logic-chain/dependency figure shows the closure order step by step.** Not only the opening overview (E2) but every logic-chain/dependency figure in the body shows, as a sequential arrow chain, which signal closes in which order, labels each arrow with the reason for that step, and connects any assumption/design condition used only at a step as a feeder to that step. The sequential closure flow is the figure's core purpose and must not be weak or ambiguous.
- **E6. 발표자료 figure를 `.tex`로 옮길 때도 caption은 제목만(E4)이고 echo가 없다.** 슬라이드에서 만든 figure를 논문에 넣을 때 caption은 영어 제목만 담고, 그 figure가 설명하는 내용은 이미 있는 body prose(직관/해석)가 담으며 없던 재서술을 새로 추가하지 않는다. figure는 그 prose를 *시각화*하고 `\Cref`로 연결한다(D2a·E4의 재확인). figure PDF는 그 슬라이드를 만든 소스 생성기에서 그대로 뽑아 재사용한다(자산 재사용, 새로 그리지 않음).

### 5c. `.pptx` 전용 규칙 (슬라이드 작업 시 --- §5a에 추가 적용)

> **§5c는 §5a 공통 원칙에 *추가로* 적용되는 발표자료(`.pptx`) 전용 규칙이다.** 슬라이드 빌드 구조, 글꼴·레이아웃, 구성안(plan) 규격을 정한다. 슬라이드는 Python 소스 번들(`build_deck.py` + 헬퍼)로 생성하고 --- PowerPoint에서 손으로 만들지 않는다 --- 모든 수식과 수식+텍스트 줄은 LaTeX로 조판해 이미지로 넣으며, 제목·agenda·순수 라벨만 native 편집 텍스트다. 내용-엄밀성·가독성(§5a A–D)과 figure 생성 규칙(§5a E)은 슬라이드에도 그대로 적용하며 여기서 반복하지 않는다. 발표자료는 **구성안 `.tex` → 승인 → pptx**의 2단계로 만들며(cluster M), 1차 산출물은 구성안이고 pptx는 그 빌드 결과다. 새 deck은 §8·발표 template 골격에서 시작한다.

#### F. 빌드 구조와 렌더링 (핵심 교훈)
- **F1. 소스로 생성한다.** deck은 `python-pptx` 소스로 만들고 슬라이드당 한 블록, 헬퍼(`add_text`, `mixline`, `add_eq`, figure)로 구성한다. 텍스트 변경은 소스 수정 + 재빌드이지 pptx 직접 편집이 아니다. 재현성과 변경 추적성이 확보된다.
- **F2. 혼합 줄은 통째로 하나의 LaTeX 이미지로 조판하고, 수식 이미지를 텍스트 space 위에 얹지 않는다.** python-pptx는 편집 가능한 텍스트 런 안에 진짜 인라인 이미지를 못 넣으므로, 수식 PNG를 textbox의 공백 위에 overlay하고 싶어진다. 하지 않는다. textbox 텍스트는 PowerPoint가 자체 metric으로 배치하고 이미지는 폰트폭 추정 좌표에 놓이므로 둘이 절대 정렬되지 않는다 --- 어떤 수식은 뒤 단어와 겹치고 어떤 것은 너무 뜨며, 편집하면 더 어긋난다. "렌더 후 간격을 측정해 이미지를 shift"하는 사후 보정은 fragile하고(수식이 몰린 줄은 진동) 해결책이 아니다. 올바른 방식은 줄 전체(텍스트는 deck 폰트로 `fontspec`, 수식은 LaTeX)를 한 이미지로 렌더하는 것이다 --- 그러면 토큰 간 간격이 원천적으로 정확하고, 좌표 불일치도 튜닝할 값도 없다. 이는 per-image trim / gap-padding / space-width 보정 / balance 후처리를 *전부 대체*한다.
- **F3. Per-instance 튜닝보다 constructive method를 택한다.** 출력물이 각 사례를 측정해 하나씩 땜질해야 맞춰진다면, 그 방식(mechanism) 자체가 틀린 것이다. 값을 실험적으로 찾아 넣는 대신 문제군을 없애는 구조로 전환한다(F2가 그 사례다).
- **F4. 본문 폰트를 일치시킨다.** LaTeX 텍스트 폰트를 deck 본문 폰트(Carlito, 시스템 TTF를 `fontspec`로 로드)로 설정해 이미지 줄과 native 텍스트가 동일하게 보이게 하고, 렌더된 텍스트 cap-height가 같은 크기의 native 텍스트와 맞는지 검증한다.

- **F5. 빌드 상수·헬퍼는 고정 자산이다.** deck은 `build.py`(python-pptx)+`build_lib.py`로 만든다. 좌표계 960×540pt, 여백 `XL=48`/`XR=912`, 들여쓰기 `XI=70`, 크기 `SZ_L=20`/`SZ_B=16`, 본문 첫 요소 목표 y≈106·하한 ≈510. 핵심 헬퍼: `base_slide(title, subtitle, note)`(subtitle은 plain text --- LaTeX 불가), `mixline`/`mixline_c`(segments = `('t',…)|('b',bold)|('m',latex)|('d',display)`), `centered_eq`(폭 ≤864 assert, y+height 반환), `label`(bold plain), `add_rect`, `add_img`/`img_size_pt`, `divider(active)`. 크기는 소스 상수로만 관리(G5).
- **F6. 편집·검증 프로토콜(고정).** 모든 `build.py` 편집은 Python `str.replace`+`assert count==1` 가드로 한다(NF/DUP이면 멈추고 원인부터). 편집 후 반드시 프로그램 geometry 검증: shape bbox로 overlap(±3pt), off-slide(x+w>958 / y+h>512), 폭 한계 초과를 확인하고, 가능하면 pptx→pdf→png(`pdftoppm`)로 pixel 점검(제목-notation gap 등). K1의 구체 실행 절차다.
- **F7. 알려진 함정(반복 방지).** (i) heredoc의 raw string이 backslash로 끝나면 SyntaxError --- anchor를 `\`로 끝나지 않게 잡는다. (ii) 그림 생성기 rcParams 포맷 불일치(`figs2.py`는 콜론 뒤 공백, `figs3.py`/`figs_concept.py`는 공백 없음) --- 문자열 매칭 시 주의. (iii) deck LaTeX는 `r'...'` single backslash, 텍스트 유니코드는 `\u`-escape(—=`—`, ⇒=`⇒`, ⇔=`⇔`). (iv) 소스 주석 `# [N]`은 삽입으로 밀려 물리 slide 번호와 불일치 --- 주석일 뿐. (v) figure PDF 필요 시 `compile_tikz`가 PNG와 PDF를 함께 생성하므로 `.tex`에서 재사용 가능.
- **F8. 순서 변경은 블록 이동으로 하되 causal(정의→사용)을 유지한다.** `build.py`는 선형이라 slide 재배치 = 블록 이동이다. 이동·삽입 후 정의가 사용보다 앞서는지 확인하고(D1의 실행), 섹션 이동 시 `T#`(섹션 제목)과 top-right notation profile도 함께 맞춘다. 필요하면 슬라이드를 늘려도 좋다(사용자 승인 하에).
- **F9. 구성안-주도 빌드 아키텍처(기계와 내용의 분리).** 빌더는 `build_deck.py`(구성안 파서·렌더러·게이트)와 `deck_spec.py`(슬라이드 순서·병합·재작·치환)로 분리한다. 내용 수정은 spec에서, 배치·크기 수정은 빌더 상단 상수에서 하며, 렌더 결과물(pptx)에 직접 손대지 않는다(F1의 구조적 구현). spec의 모든 hook은 `assert count==1`로 anchor를 검사해, stale anchor가 조용히 잘못된 슬라이드를 만들지 않고 빌드를 멈추게 한다(F6).

#### G. 글꼴·크기 규약 (기본 선호 --- 사용자 지시로 조정 가능)
- **G1. 크기 사다리(본문) --- 위계로 키우고, 여백으로 키우지 않는다.** 기본 본문은 **20pt**다. 사다리: 본문 20(`SZ_B`) / 보조 display 16(`SZ_SM`) / 데이터 display 14.5(`SZ_XS`, 하한) / 강조 헤드라인 24(`SZ_H`) / 그림 내부 라벨 ≥13. 사이의 임의 크기는 지양한다. 크기는 **내용의 위계에 따라** 올린다 --- 단락 헤더(Assumptions 류)는 본문 +2pt bold, 라벨 토큰만 bold, 그 외 bold 금지. **공간이 남는다는 이유로 키우지 않는다.** 한 슬라이드 안에서 같은 위계의 요소는 같은 크기여야 하고, 텍스트만 키우고 그 줄의 수식을 두는 것은 G4 위반이다. **이는 기본 선호이며 사용자 지시가 있으면 조정한다.**
- **G2. 제목은 템플릿이 정한다.** 슬라이드 제목은 템플릿(master)의 크기를 그대로 쓰고 슬라이드마다 개별로 덮어쓰지 않는다 --- 제목 크기의 일관성은 템플릿이 보장한다.
- **G3. 16pt 미만은 주석 예외만.** 16pt 미만은 그림 내부 라벨과 코너 상태 주석(현재 12–13pt)에만 최소한으로 허용하고, 본문에는 쓰지 않는다.
- **G4. 글꼴은 템플릿 글꼴을 유지한다.** 본문 글꼴은 현재 템플릿 글꼴(Calibri)을 그대로 쓰고 다른 글꼴을 도입하지 않는다. LaTeX 조판 이미지의 텍스트 글꼴은 metric이 일치하는 Carlito를 `fontspec`로 써서 native 텍스트와 이미지 텍스트가 동일하게 보이게 한다(§F4와 짝). 수식 크기는 그 줄의 텍스트 크기를 따라간다(함께 렌더) --- 수식 전용 크기 knob은 두지 않는다.
- **G5. 크기는 소스 상수로만 관리한다.** 크기는 소스의 상수(`SZ_B=16`, `SZ_L=20`, `SZ_H=24`)로 두고 호출부에 임의 숫자를 흩뿌리지 않는다. 크기 변경은 상수를 고쳐 전역 일관성을 유지한다.

- **G6. Top-right 상태 notation은 섹션별 profile로 관리한다.** 슬라이드 우상단 상시 표기(현재 쓰는 기호 리마인더)는 제목 접두어로 자동 선택하고(§2/§3→setup, §4→feas, §5/§6→cex), 정의 슬라이드·divider엔 blank로 둔다. 새 기호를 본문에서 쓰기 시작하면 그 섹션 profile에 추가한다.

- **G7. 자동 축소(shrink-to-fit)는 결함 신호다.** 폭 초과 시의 자동 축소는 파손을 막는 안전장치일 뿐 해법이 아니다. 축소가 발동한 줄은 **줄 분할이나 문장 축약으로 해소**해 사다리 크기를 회복한다. 빌더는 축소 발동을 세어 보고하고, 전달 게이트는 축소 0을 요구한다. 한 슬라이드 안에 서로 다른 실효 크기가 섞이면 위반이다(G1과 짝).

#### H. 서술 스타일 --- 쉽되 정의를 빼지 않는다 (가장 중요)
- **H1. 쉽고 간결하게, 그러나 정의 없이 쓰지 않는다.** 슬라이드는 가능한 한 쉽고 짧게 서술한다. 그러나 *간결함이 정의 생략의 핑계가 될 수 없다*: 우리가 정의한 대상(기호·집합·조건)은 슬라이드에서도 그 정의를 유지한 채로 써야 하며, "간단히 하려고" 정의된 항을 빼거나 뭉개고 설명하면 안 된다. 이는 `.tex`뿐 아니라 슬라이드에도 동일하게 적용된다(§5a A1의 슬라이드 판). 기호는 그 슬라이드(또는 이전 슬라이드)에서 처음 쓰일 때 정의되어야 하고, 정의되지 않은 기호를 결론에만 등장시키지 않는다.
- **H2. 쉬움과 엄밀함은 상충하지 않는다.** 쉽게 만드는 방법은 정의를 빼는 것이 아니라, 정의를 유지한 채 (i) 핵심을 먼저 말하고 (ii) 추상적 대상을 구체적 근거(interval 그림, 작은 예, 물리적 그림)로 받쳐 보이는 것이다. 어려워 보이는 것은 정의가 많아서가 아니라 통찰이 안 보여서다.
- **H3. noun-phrase가 아니라 완전한 명제로 쓴다.** 슬라이드 한 줄은 정확한 대상과 scope를 갖춘 완전한 주장이지 느슨한 paraphrase가 아니다(정리가 특정 집합의 invariance를 말하는데 "keeps `x̄` in `E`"로 뭉개지 않는다). Case 분할은 exhaustive하게 번호로 보이고, 증명으로 확립할 것을 슬라이드에서 가정하지 않는다.
- **H4. 슬라이드당 takeaway 하나 --- 그리고 그 takeaway는 핵심 통찰·물리적 직관까지 도달해야 한다.** 각 슬라이드는 소제목이 지목하는 단일 핵심을 담는다. 그러나 여기서 멈추지 않는다: 슬라이드의 목적은 사실의 나열이 아니라 *왜 그러한지의 핵심 통찰이나 물리적 직관을 전달하는 것*이다. 결론 문장은 "무엇"에 더해 "왜/어떻게(메커니즘)"가 한 번은 드러나야 한다(§5a D5의 forest/직관과 같은 정신). 통찰 없이 결과만 있는 슬라이드는 미완성으로 본다.

- **H5. 직관은 말과 그림을 함께 앞쪽에 배치한다.** 핵심 개념(예: `ker Q = O`)의 직관은 그 개념이 처음 서는 자리에 verbal + visual을 함께 둔다. 시각 자료(corridor 등)를 뒤 섹션에 묻지 말고 개념 확립 지점으로 당긴다(H4·D5의 확장, 슬라이드·`.tex` 공통).
- **H6. 형태는 전보체, 내용은 완전한 명제(H3와 양립).** 슬라이드 텍스트는 완전한 *문장* 금지 --- 최소 단어, 구 단위, 화살표는 **`→, ⇒, ⟺`만** 허용(`↗`·`↑` 등 정의 없는 방향 기호 금지), 모호 동사 금지 --- "hold"→"satisfied", "carries"→"support rows", "dips"→"is negative", "err outward"→"over-approximates". 단 H3의 요구(정확한 대상·scope를 갖춘 주장)는 전보체 안에서 유지한다: 생략하는 것은 관사·조동사·군더더기이지 대상·조건이 아니다. 핵심은 텍스트보다 **keybox·수식·그림**이 먼저 전달한다 --- 가장 중요한 내용을 사람들이 가장 안 읽는 긴 문장 속에 두지 않는다.
- **H7. 약어·은어·정리 이름.** 약어는 첫 사용에 gloss를 병기한다(LB = certified lower bound, witness = 인증 원소에서의 직접 평가, rest IC = set-point equilibrium IC 등). 내부 은어·은유는 금지하고 직설로 바꾼다(knee→saturation point, carrier→그 항이 실제로 무엇인지, verdict→outcome, tier→evaluation level, price→conservativeness, floor/lift/anchor-row/star 류). 금지어 점검은 K5 게이트가 산출물에서 자동으로 수행한다. 사용하는 정리는 실명을 명시한다(Farkas Lemma, Minkowski--Weyl).
- **H8. 슬라이드 수식 규율.** display 안에 한글을 넣지 않는다. 중간 기호를 새로 정의하기보다 full expression을 우선한다(예: `g_z` 정의 대신 `H_u K_I` 직접 대입 --- 사용자 반복 지시). 행렬·집합에는 차원을 명시하고, display가 열 폭을 넘으면 분할한다.
- **H9. 설명 단위는 "주장 + 대응 수식"이다.** 배치 순서는 이해를 돕는 쪽으로 정하되, 텍스트를 여러 줄 몰아 쓴 뒤 수식을 몰아 놓아 주장-수식 대응이 끊기는 배치를 금지한다. 대응 수식이 있는 주장은 그 수식과 인접시키고, 수식만으로 자명하면 텍스트를 생략한다(§6 설명 규율·J6와 병립).
- **H10. 슬라이드 대면 표기 위생.** (i) 슬라이드에 렌더되는 텍스트에 **한글을 쓰지 않는다**(`[보충]` → `[supplement]`; H8의 display 규칙을 텍스트로 확장). (ii) 개수는 숫자로 쓴다(two → 2). (iii) 미정의 괄호 주석("(exception)" 류) 금지. (iv) **기호 충돌 금지** --- 이미 쓰인 기호를 다른 뜻으로 재사용하지 않는다(cone `Λ` vs 상수 `Λ_i` → `ℓ_i`). 위반은 K5 게이트에서 걸러진다.

#### I. 슬라이드 내용 엄밀성 (§5a의 슬라이드 적용)
- **I1. 출처에 충실하다.** 슬라이드의 모든 주장은 논문·노트의 검증된 결과로 추적되며, 슬라이드는 제시·압축할 뿐 새로 만들지 않고, 출처가 뒷받침하는 것보다 더 강하거나 다른 주장을 하지 않는다(§5a C1). 슬라이드의 모든 수학 진술은 `main.tex`/companion 원문과 **대조 검증 후** 기재하고, 불일치를 발견하면 즉시 보고·교정한다(사례: `Q`는 `b`의 계수가 아니라 `Ω_α` 제약의 계수 --- fact:imposs 대조로 교정).
- **I2. Scope를 정확히 라벨한다.** exact vs limit, iff vs sufficient-only, 조건이 걸리는 영역을 슬라이드에서도 정확히 표기한다(§5a C1·A4). 과대·과소 주장 모두 금지한다.
- **I3. 데이터 격리.** 수치·시나리오·시뮬레이션·인스턴스 값은 Numerical 전용 part에만 등장시킨다. 이론부의 그림도 기호화한다(예: cone 그림의 실측 계수 √10 → 기호 `a_i>0`). 시각화용 부분문제·subspace를 쓰면 그 사실을 슬라이드에 명시한다. 인스턴스 명칭은 **구조 명칭(SISO/MIMO 등)**을 쓰고 내부 코드명(`Comp3`, `HM-A` 등)을 슬라이드에 노출하지 않으며, 이미지에 구워진 캡션 텍스트도 같은 규약의 대상이라 위반 시 그림을 재생성한다.

#### J. 그림·톤·레이아웃
- **J1. 라벨 있는, 메시지 있는 그림.** 모든 그림 요소에 라벨을 달고 그림마다 한 줄 takeaway를 두며, 역할별로 color-code하고 non-data ink를 최소화하며, 그림 표기를 슬라이드 텍스트와 일관(번호 충돌·중복 금지)시킨다. 오해를 유발할 수 있는 그림은 보완 그림을 병행한다(예: `(z, ż)` 2D `Ω_α` 옆에 row(Q)-compact / ker Q-open 추상 3D). 동일 이미지를 두 슬라이드에 쓰지 않고, 효용이 낮은 그림은 텍스트로 대체하며, placeholder 그림은 "빌드 시 교체" 플래그를 구성안에 명기한다. 그림은 **본문과 의미가 충돌하지 않아야 한다** --- 잘못된 인과를 암시하는 화살표·주석은 제거하고, 축 라벨은 슬라이드에서 정의한 기호로 쓴다(예: "sensed" → "row(Q) direction"). 그림 2매를 나란히 둘 때는 공통 축에 수직 중앙 정렬한다.
- **J2. Block diagram 스타일을 일관되게 한다.** 둥근 사각형 블록, 합(sum) junction은 원에 +/− 라벨, 신호 화살표에는 신호명, augmentation 관련 신호는 색(예: green/red)으로 표시한다. 다이어그램 간 일관성을 위해 레이아웃을 정렬한다(예: 두 블록도의 가로폭을 맞추려 좌우 대신 상하 배치).
- **J3. 색을 절제하고 톤을 통일한다.** 슬라이드를 overcrowd하지 않고(개념도이지 상세 배선도가 아니다), 표는 compact하게 문장이 아닌 구로 쓰며, 본문 글자 크기는 발표에서 읽히도록 충분히 둔다(§G).
- **J4. 슬라이드 간 흐름.** Divider·agenda recap은 허용하되 내용 반복은 금지하고(§5a D2), 각 조건은 쓰이기 전에 설명되도록 순서를 잡는다(§5a D1). 겹치는 슬라이드는 병합하고 슬라이드 간 재진술은 제거한다 --- 특히 같은 display를 두 슬라이드 본문에 두지 않는다(도입 슬라이드 한 곳 + 나머지는 참조; 예: `σ_k`). 부(Part) 시작에는 "왜 지금 이걸 하는가" intro를, 끝에는 recap·다음 부 예고를 둬 청중의 위치를 고정하고, 전체를 관통하는 판정문 하나(예: feasibility ⟺ ν_k≥0)를 반복 노출한다. **병합은 융합이 아니라 적층이다**: (i) 중복 서술을 지우고, (ii) 두 내용을 위아래로 배치하며, (iii) 사이에 확실한 공백을 넣어 맥락 경계를 보이고, (iv) 두 top note를 통합·중복 제거한다 --- 두 슬라이드의 문장을 섞어 새 문단을 만들지 않는다.

- **J5. figure는 소스 생성기로 만들고 deck/`.tex`가 자산을 공유한다.** deck figure는 `figs*.py`(tikz)·matplotlib 생성기로 만들고, `.tex`에 넣을 땐 동일 생성기에서 PDF를 뽑아(matplotlib savefig PDF, tikz는 `compile_tikz`의 PDF) 재사용하며 새로 그리지 않는다(D2·자산 재사용).
- **J6. 수식이 드러내는 것을 prose로 재서술하지 않는다(축약 원칙 --- 슬라이드 적용).** 인접 수식·행렬 라벨·figure가 이미 보여주는 내용을 문장으로 반복하지 말고, prose는 수식에 없는 why/consequence/mechanism만 담는다(§5a D2a의 슬라이드 판). 예: 행렬 라벨이 "$A_y$ fixed"를 이미 말하면 그 문장 삭제; interval 수식이 있으면 "each box gives an interval" 삭제; 박스 내 분해 수식이 있으면 "moves through X and Y" 삭제.

- **J7. 슬라이드 구성요소 규율(부제목·top note·보충).** (i) 슬라이드마다 부제목을 단다 --- PPT 제목 칸이 section명일 때는 부제목이 실질 헤딩이다. 부제목은 **요약·설명·맥락 유지 기능**을 수행해야 하며, 미사여구·단어 나열·내부 용어만으로 된 제목은 금지한다; 형식은 "무엇을 하는 슬라이드인가 --- 어떻게/왜"의 한 구다. (ii) top note(우상단 상시 문맥)는 **가이드 수식만** 담는다: 메타 설명("data: this part only", "figure 재사용" 류)과 슬라이드 번호 포인터를 넣지 않고, 본문에 있는 수식은 note에서 제거하며, **한 수식은 한 줄**로 쓰고 폭을 넘으면 줄바꿈이 아니라 축소한다; 최대 ~5줄, 연속 슬라이드에서 같더라도 반복 게시한다. (iii) 보충 슬라이드는 본편보다 유도·수식을 더 충실히 담고(축약 금지) 숨김 처리하지 않는다.

- **J9. 그림 자기설명성 --- 질문은 결함 신호다.** 그림의 요소(선 색·기울기 군·보조선·음영·문턱선)가 의미를 가지면 범례나 라벨로 그 의미를 명시한다. 청중이나 사용자가 "이 요소가 무엇이냐"고 물으면 답변으로 끝내지 않고 **그림을 수정한다**(§6 반복 질문 프로토콜의 그림 판). 그림이 스스로 설명되지 않으면 발표 중 구두 보완이 필요하다는 뜻이고, 그것은 그림의 실패다.

#### K. 검증·정직·번들
- **K1. 레이아웃 검증(게이트 형식).** 모든 줄이 슬라이드 폭·높이에 fit하고 shape 충돌이 0인지 shape-bbox geometry로 프로그램적으로 확인한다. 빌더는 매 빌드에 `slides: N  off-slide: 0  shrink: 0`을 출력하고, 위반 shape의 좌표를 함께 찍는다 --- **off-slide 0·shrink 0(G7)·슬라이드 수 assert**를 통과하지 못하면 다운스트림(변환·전달)으로 넘어가지 않는다. 이어서 pptx→pdf→png로 변환해 고정 페이지를 pixel로 점검한다.
- **K2. 시각 확인에 정직하다.** 도구가 렌더된 슬라이드를 실제로 보여주지 못하면 그렇다고 말하고 측정에 의존하며, 하지 않은 시각 확인을 했다고 말하지 않는다(§5a D7·정직 정신). 최종 미적 판단은 사용자에게 전달된 PDF 확인을 요청한다.
- **K3. 요청 시 재현 가능한 번들을 전달한다.** `.py` 소스, 템플릿 `.pptx`, 미리 생성한 figure, 그리고 의존성 목록(`texlive-xetex`/`-latex-recommended`/`-latex-extra`/`-fonts-recommended`, `fonts-crosextra-carlito`, `poppler-utils`; `python-pptx`, `Pillow`, `numpy`, `lxml`, `matplotlib`)과 정확한 빌드 순서(figure → `build.py` → 선택적 `soffice` PDF)를 담은 README를 포함한 self-sufficient 번들을 전달한다. 전달 전에 캐시 없이 clean-room 빌드가 되는지 검증한다.
- **K4. 구성안(.tex) 문서 표준 --- cluster M 참조.** 구성안의 블록 포맷·필드·게이트는 §5c M(M3·M7·M8)에 둔다.
- **K5. 텍스트 레이어 검증.** pptx→pdf 변환 후 `pdftotext`로 (i) LaTeX 리터럴 누출(`[2pt]`, `\textbf`, 명령 잔재), (ii) 부제목의 기호 소실·문자 깨짐(부제목은 native text라 LaTeX 불가 --- 유니코드 변환표 누락이 원인), (iii) 금지 용어(H6·H7 목록)와 표기 위반(H10: 한글·숫자 단어)을 점검하고, 이어서 고정 페이지 스팟 래스터로 겹침·잘림을 확인한다. 발견 0이 전달 조건이다.
- **K6. 피드백 라운드 프로토콜.** (i) 사용자가 말한 슬라이드 번호가 어느 블록인지 **매핑(오프셋)을 먼저 확인**한다. (ii) "패스"는 무수정을 뜻하지 않는다. (iii) 한 지적은 그 슬라이드에서 끝내지 않고 **동종 전 슬라이드에 파급 적용**한다(§5a D8의 deck 판). (iv) 라운드마다 버전을 올리고 보고에 4요소를 담는다: 반영 내역 / 재번호 매핑 / 신규 발견·불일치 / 잔여 플래그.
- **K7. 검증값은 한 번 제대로 만들고 기록해 재사용한다.** 수치는 산출 시점에 검증하고, 검증된 값은 **단일 기록처**(`records/verified_values.md`)에 산출 조건 --- 무엇을 계산했는지·어느 스크립트·어느 평가 수준(근사 tier) --- 과 함께 남긴다. 이후 `.tex`·구성안·deck은 그 기록을 인용해 쓰고 **재계산하지 않는다**; 재계산은 설정이 바뀌었거나 불일치가 발견됐을 때만 한다. 기록에 없는 값은 산출물에 쓰지 않으며, 평가 수준이 다른 값을 같은 줄·같은 그림에 수준 표기 없이 섞지 않는다.
- **K7b. 정합은 편집마다가 아니라 전달 경계에서 강제한다 --- divergence ledger.** 작업 중에는 한 산출물만 고쳐도 된다(그림을 반복 수정할 때 `.tex`까지 매번 따라가는 것은 낭비다). 대신 산출물이 서로 어긋나는 **그 순간** `records/verified_values.md` 상단의 `## Divergence (open)`에 한 줄을 남긴다: 무엇이 무엇보다 앞서 있는지, 어느 값·그림인지. 그 줄은 실제로 정합시켰을 때만 지운다. **전달·인수인계 시점에는 이 목록이 비어 있어야 하며, 비어 있지 않으면 전달하지 않는다** --- 인수인계는 정의상 모든 산출물(deck·`.tex`·기록)이 서로 정합한 상태를 넘기는 것이다. 수치를 바꾸거나 폐기하는 편집은 정합 시점에 기록도 함께 고친다(새 값은 산출 조건과 함께 추가, 폐기값은 삭제, 사다리·상자·설정이 바뀌면 기록의 서술 문단도 교정). 기록이 자기 데이터 줄과 모순되면 데이터 줄이 옳고 서술이 stale인 것으로 본다.

- **K8. 정합 게이트는 스크립트다 --- `check_sync.py`.** K1(레이아웃)·K5(텍스트 레이어)와 같은 층의 전달 게이트를 하나 더 둔다. 세 가지를 프로그램으로 검사한다: (i) 산출물들이 공유하는 그림의 해시 일치, (ii) 산출물이 인용하는 수치가 기록에 존재, (iii) 폐기된 설정·값 문자열의 잔존, 그리고 (iv) `## Divergence (open)`이 비어 있는지(K7b). **전달 전 0이 조건**이고, "완료"·"정합 완료" 보고는 이 스크립트를 돌린 뒤에만 한다(D9의 수치판 --- 외울 항목을 늘리지 않고 명령 하나를 더 친다). clean-room 빌드 검증(K3)은 번들 안의 **모든 빌드 가능 산출물**을 대상으로 하며, 경로 상수가 번들 상대경로인지 함께 확인한다.

#### L. 대외 표현 (청중에 해당 저자가 포함될 때)
- **L1. 완곡하게 쓴다.** 기존 연구의 문제를 지적할 때 "identified a concern regarding the validity of ..."처럼 완곡하게 서술하고 "틀렸다/오류/잘못" 같은 직접 표현은 피한다.

#### M. 구성안(Plan) 규격 --- 발표자료의 1차 산출물
- **M1. 2단계 워크플로를 고정한다.** 발표자료는 **구성안 `.tex` 작성 → 사용자 승인 → pptx 빌드** 순으로 만든다. 승인 전에 pptx를 만들지 않는다. 구성·순서·수식·문안은 구성안 라운드에서 확정하고 pptx 라운드로 미루지 않는다 --- pptx 단계의 수정 비용이 훨씬 크다.
- **M2. 원본은 최초 빌드 시점에 이동한다.** 구성안은 승인·최초 빌드까지의 작업 원본이며, **빌드 이후의 수정은 deck 소스(`deck_spec.py`)에서만** 한다. 구성안으로 역동기화하지 않고 그 시점의 설계 기록으로 동결한다. 다음 발표·재설계가 필요하면 새 구성안을 만든다.
- **M3. 블록 포맷을 고정한다.** 슬라이드당 한 블록으로 쓴다: `\begin{slideblk}{번호}{부제목}{top note}` + `\ag`(그 슬라이드의 목적 한 줄) + `\ct`(내용 요지·빌드 지시) + `\tx`(실제 줄배치 그대로 개행) + display 수식 전부 + 선택적 `\keybox`/`\fg`/`algorithmic`/`itemize` + `\end{slideblk}`. `\ag`·`\ct`는 렌더되지 않는 작성자 메모다. 블록 번호는 **pptx 물리 슬라이드 번호와 1:1**로 유지한다 --- 피드백이 슬라이드 번호로 오기 때문이다.
- **M4. `\ct`는 빌드 지시를 담는다.** figure 재작 플래그, 레이아웃 지시(2단 배치 등), placeholder 여부, 자산 공유 경로, 평가 수준·출처 라벨을 `\ct`에 명기한다 --- 빌드 시점에 즉흥 판단하지 않는다.
- **M5. 슬라이드 번호 참조는 `\ct`에만 쓴다.** 본문(`\tx`·display·keybox·top note)에 "(S32 참조)" 류 포인터를 넣지 않는다. 청중에게 필요한 것은 위치가 아니라 그 수식 자체이므로, 근거 수식은 top note로 올린다.
- **M6. 재번호는 전면 원자 연산이다.** 삽입·병합·분할 시 전 슬라이드 재번호 + 섹션 헤더 범위 + `\ct`·top note 내 참조 갱신을 한 스크립트에서 수행하고, 보고에 **old→new 매핑표**를 포함한다. 부분 재번호는 금지한다.
- **M7. 구성안 게이트.** XeLaTeX 3-pass · **Overfull 0** · 블록 번호 연속성 assert(1..N)를 통과한 뒤에만 전달·빌드로 넘어간다.
- **M8. top note는 구성안의 필수 필드다.** 세 번째 인자로 **작성 시점에 확정**하며 빌드 때 임시로 채우지 않는다. 내용 규율은 J7(ii)를 따른다. 슬라이드를 병합하면 두 note를 통합·중복 제거하고, 연속 슬라이드에서 동일하더라도 매 슬라이드에 반복 게시한다.

---


## 6. 소통·출력 스타일

- **언어**: 토론은 한국어, 전문 용어는 영어 그대로. **반말 금지하되 honorific도 쓰지 않는다**(돌려 말하지 않고 있는 그대로).
- **이중언어 병기 금지**: 전문 용어는 영어 단독으로 쓰고, 같은 뜻의 한국어를 괄호로 병기하지 않는다(예: '보수(complement)', 'value(크기)', 'rate(변화율)' 금지 --- 'complement', 'value', 'rate'만). 용어 + 짧은 정의 형태의 brief 설명은 허용하나, 동의어 쌍을 두 언어로 적는 것은 금지한다. 채팅·`.tex` 모두 적용.
- **한자 절대 금지** (한자/한문 질문이 아닌 한). **이모지 금지** (요청 시 제외).
- **설명 규율 (이론 설명의 기본값).** 이론·개념·질문에 대한 설명 응답에는 예시·simulation 결과·수치를 포함하지 않는다 --- 이론 자체만 설명한다. 수치·예시는 사용자가 명시적으로 요구·지시한 경우와 연구·검증 보고(수치가 산출물인 응답)에서만 포함한다. 설명의 내용은 핵심 통찰·직관·motivation·아이디어이고, 전달 수단은 formal한 수식의 display다: 주장마다 대응 수식을 함께 제시해 읽는 즉시 이해되게 한다. 비표준 술어와 비유는 이해를 저해하므로 극도로 자제하고, 비유는 설명을 확실히 쉽게 만드는 제한적이고 분명한 경우에만 쓴다. 자체 정의·내부 용어를 최소화한다. 수치의 지위(정확값·하계·simulation 확인 여부)는 필요한 자리에서 평이한 수학 서술로 밝히되, 그것을 위한 고정 label 어휘를 자체 정의해 붙이지 않는다. 이 규율은 채팅과 `.tex` 서술(직관·remark·해설 prose)에 동일하게 적용된다(D3·D5·B4 연동).
- **반복 질문 프로토콜.** 같은 질문이 다시 오면 설명이 실패한 것이다. 같은 내용을 다른 어휘로 반복하지 않고 형식 수준을 올린다: 정의에서 출발하는 유도 중심 재구성, 주장-수식 일대일 대응, 앞선 설명에서 실패한 서술어의 재사용 금지.
- **용어 위생 (전 매체).** A7(자체 정의 용어·과잉 표기 자제)과 §4.14(author-internal label 금지)는 채팅·`.tex`·deck 전부에 적용된다. 새 용어를 쓰기 전에 표준 용어의 존재를 확인한다.
- **채팅 수식은 MathJax 기준 + predefined macro 금지**: 채팅 보고의 수식은 **MathJax로 렌더링**되므로 MathJax에서 바로 보이게 작성한다. `.tex` preamble의 **predefined macro를 채팅에서 쓰지 않는다**(MathJax가 모름) — 표준 LaTeX로 풀어 쓴다. 매핑: `\Clim`→`C_{\mathrm{lim}}`, `\Hu`→`H_u`, `\Gz`→`G_0`, `\zmin`/`\zmax`→`z_{\min}`/`z_{\max}`, `\zbar`→`\bar z`, `\uM`→`u_M`, `\usat`→`u_{\mathrm{sat}}`, `\Rinf`→`\mathcal{R}_\infty`, `\Rih`→`\widehat{\mathcal{R}}_\infty`, `\Zx`→`\mathcal{Z}_x`, `\Zcal`→`\mathcal{Z}`, `\tbi`→`\widetilde{\beta}_\infty`, `\rhop`→`\rho_p`, `\R`→`\mathbb{R}`, `\inner{a}{b}`→`\langle a, b\rangle`, `\norm{x}`→`\lVert x\rVert`, `\Cinf`→`(\mathrm{C3}_\infty)`. 이 규칙은 **채팅 보고에만** 적용되며, `.tex` 문서 자체에는 §5a cluster A·B의 macro를 그대로 쓴다.
- **핵심 먼저** 답한 뒤 필요한 만큼만 concise하게. 장황·redundant 금지.
- **모든 설명은 causal**하게(논리 순서, 왔다갔다 금지).
- **Code comment는 영어만, concise.** redundant 구분선 금지.
- **불가능하면** 틀린 답을 만들지 말고 불가능 이유를 설명한다.
- **주장이 의심받으면 먼저 수치·기호로 검증한 뒤 답한다(단, 검증은 corroboration이지 proof가 아니다).** 대화 중 어떤 주장에 의문이 제기되면 기억이나 직관으로 답하지 말고 sympy/numpy 등으로 확인한 뒤 답하고, 틀렸으면 인정하고 슬라이드·`.tex`를 정정한다. 이 수치·point 확인은 오류를 걸러내는 sanity-check이자 corroboration일 뿐 formal proof를 대체하지 않는다 --- 정식 증명은 §5a B의 equation-forward 논증으로 별도로 세운다(§4.6·§5의 self-check를 대화 반응 규칙으로 구체화). **역방향도 같다 --- 내가 우리 산출물을 의심할 때.** 결함(미기록 값·등급 오표기·출처 귀속)을 보고하기 전에 "진짜 결함"과 "내 탐색 부족"을 가르는 최소 확인을 먼저 한다: 번들 전체에서 산출 아티팩트(`.json`·`.npz`·로그·스크립트)를 찾고, 반올림이 아니라 **정확한 자릿수**로 대조하며, 출처·귀속 주장은 우리 문서의 서술이 아니라 **1차 문헌**으로 확인한다. 그래도 못 찾으면 "결함"이 아니라 **"unlocated"**로 보고한다. 우리 문서의 귀속 오류는 심사자가 가장 먼저 잡는 종류이므로 발견 즉시 보고·교정한다.
- **Claude Code용 prompt**는 최대한 자세·정확하게 작성하고, 근처 폴더의 참고 자료·코드를 적극 읽도록 guide한다.
- **Continue 시 상황 보고**: 한 채팅에서 max length 도달·compaction·"continue" 지시 등으로 작업을 이어갈 때는, 항상 현 상황(직전까지 무엇을 하고 있었는지·어디까지 됐는지·다음 단계)을 간략히 보고해 연구가 제대로 진행 중임을 indicate한다.

---

## 7. 파일·도구

- **Project files** (읽기 전용, `/mnt/project/`):
  - **SAG framework 1차 출처 (현재 버전, 반드시 먼저 읽어 ground할 것)**: `SAG_V1s2`. SAG 구조·정의·표기·convention(plant, PI servo, anti-windup, HOCBF cascade, reference governor, smooth saturation)의 authoritative source.
  - **Boundedness·C3 이론 (project files, 함께 읽어 ground할 것)**: `Boundedness_of_the_Saturation_Aware_Governor__SAG__Closed_Loop_for_Hurwitz_SISO_Plants`(boundedness 증명), `Assumption (C3)의 다면적 이해: SAG Boundedness Condition의 통합 분석`(C3 조건 통합 분석). 현재 연구 내용이므로 `SAG_V1s2`와 함께 읽어 현황을 파악할 것.
  - **이전/다른 버전 (현재 기준 아님)**: `CBF_Servo_Controller_Pre-Simulation_Feasibility_Framework__LTI_System_`는 다른/이전 버전이며, 현재 기준은 `SAG_V1s2`다.
  - **Scenario 정의 = simulation test pool (우리가 만든 것, 무조건 읽을 것)**: `scenarios.tex` — 열한 plant(standard seven: Aircraft, CE1–CE3, DI, X-29A, Companion / demonstration two: oscillatory 2nd-order Scenario A, ill-conditioned 4th-order Scenario B / hidden-mode two: HM-A, HM-B)의 plant·constraint·gain($K_P,K_I$; SAG는 $K_{bc}$; LQR 유래는 $Q,R$)·relative degree·formal command·initial condition·horizon과, 각 scenario가 무엇을 점검하는 case인지(**Purpose**)를 담는다. 다양한 LTI(SISO/MIMO, stable/unstable, 2–4차, well/ill-conditioned, relative degree 1/2, observable/hidden)를 아우른다. 새 세션 시작 시 `SAG_V1s2`와 함께 먼저 읽고, 시뮬레이션·점검 요청 시 이 **열한 개**를 모두 점검할 것.
  - **원논문/배경**: `ACC_2025_...ServoControllers...`, `AIAA_SCITECH_2026_...`, `ServoControl_with_BoxConstraints...Lavretsky...`, `orange_book.pdf`, `CSS_2024_...Composition_of_CBFs`, `CDC_2023_Safe_and_Stable_Adaptive_Control`, `REF_Adaptive_control...input_constraints`, `2_153_Lecture_18/19`.
  - **별도 라인**(본 노트와 분리): `Output_Feedback_EBSB_Design_Note`, `TAC_EBSB_Jan2`, `20260116_EBSB_review_presentation`, `ACC_Paper_Thm3_statement_1_Counterexample_Analysis`, `20260410_Boeing_Meeting...`, `20260312_Safe_Adaptive_Control`.
- **검증 도구 (신뢰 가능)**: LMI feasibility, vector-field point evaluation, symbolic check는 신뢰 가능하므로 사용한다. from-scratch 시뮬레이션은 정량 주장에 신뢰 불가.
- **Smooth saturation 정의**: `SAG_V1s2` 참고.
- **발표 template (`template.zip` --- 새 deck은 여기서 시작한다)**: 구성안·deck·그림·게이트가 한 벌로 묶인 재사용 골격이며, §5c M·F9·K1·K5·K7의 구현체다. 구성: `plan/plan_template.tex`(구성안 원본 --- 블록 포맷·top note 필드·deck meta·마커 정의), `plan/build_plan.sh`(구성안 게이트: 3-pass·Overfull 0·번호 연속성), `deck/build_deck.py`(구성안 파서·렌더러·크기 상수·게이트), `deck/deck_spec.py`(빌드 후 내용 수정 hook: ORDER/OV/PATCH/REGEXPATCH/MERGES, 전부 `assert`), `deck/build_lib.py`·`deck/skeleton2.pptx`(고정 자산, 수정하지 않는다), `deck/gen_static_figs.py`·`deck/compute_figs.py`(정적·계산 그림 생성기), `deck/check_output.sh`(텍스트 레이어 게이트), `records/verified_values.md`(검증값 단일 기록처), `CHECKLIST.md`(라운드·전달 체크리스트), `demo/`(템플릿 그대로의 참조 빌드). 사용 순서: (1) 폴더를 새 이름으로 복사 → (2) `plan/`에서 구성안을 쓰고 `./build_plan.sh`로 게이트 통과 후 PDF를 사용자에게 전달·승인 → (3) `deck/`에서 그림 생성 → (4) `python3 build_deck.py ../outputs/<name>.pptx ../plan/<name>.tex`로 `off-slide: 0  shrink: 0` 확인 → (5) `./check_output.sh`로 리터럴 누출·금지 용어·한글 0 확인 → (6) K3 번들로 전달. 프로젝트에 맞출 때는 plan preamble의 macro 정의와 `build_deck.py`의 `MACROS`·`UNI`(부제목은 native text라 LaTeX 불가)를 **함께** 갱신하고, 크기 조정은 빌더 상단 상수만 고친다(G5).
- **LaTeX 렌더링 환경**: 한글 문서는 **렌더 시 fontspec + `Noto Serif CJK KR` (XeLaTeX 3-pass)**로 영어·한글 모두 출력; **사용자 제공 `.tex`는 그 줄을 `\usepackage{kotex}`로 치환**(§5a 규칙 E3). 검증은 `\Cref` 참조·undefined·중복 라벨·missing-character 0을 확인하고, 한글이 실제 보이는지 `pdftotext`/`pdftoppm`으로 점검(KO-치환 struct PDF로 대체 금지). overview tikz figure는 `[H]`(float) 배치(§5a 규칙 E2).
- **수치 가드**: `np.trapezoid`(not `trapz`), overflow는 `np.clip`, `solve_ivp` Radau `rtol 1e-6`.
- **일괄 편집의 원자성**: 다중 `str.replace` 배치는 각 치환에 `assert count==1`을 걸고 write-at-end로 작성해, 하나라도 실패하면 전량 미적용을 보장한다. 코드가 코드·`.tex`를 생성해 파일에 쓸 때는 기록된 산출물에서 backslash 층수(`\\lambda` 잔존 등)를 `grep`으로 검증한다 --- escape 이중화로 렌더가 깨지는 사고의 방지.
- **환경 quirk (재발견 방지)**: 이 환경의 matplotlib mathtext는 `\big`, `\le`, `\ge`를 미지원하고 `\mathcal`은 brace가 필요하다 → `\leq`, `\geq`, `\mathcal{Z}`를 쓴다(short alias·무-brace 금지). 새 환경 quirk는 발견 즉시 이 항목에 추가한다.

- **Bundle directory 구조 (표준).** 코드·figure·문서를 다음 3계층 flat 구조로 관리한다. 문서(`.tex`)가 여러 개로 분산되는 것은 특수 상황이며 기본은 합쳐진 하나다.
```
code/
  data_<plant>.py        plant/coord/certified data -> <plant>.npz (single source)
  lib_<concern>.py        2개 이상 figure가 공유하는 계산 (style/governor/reachset/simulate)
  _<name>.py              특정 figure 전용 helper (sweep 등) 또는 편입된 검증 모듈
  fig_<what>.py           figure 스크립트 1개 = 산출물 <what>.png 1묶음 (E8)
  build_all.py            data_* 실행 후 모든 fig_* 실행
figures/                  모든 산출물 PNG (E9)
tex/                      main.tex (+ 분산 시 insert/case body; 병합이 기본)
deck/                     빌드된 .pptx + 구성안 plan.tex + 재빌드 인프라
                          (build_deck.py, build_lib.py, deck_spec.py, skeleton2.pptx, figs/)
```
새 연구는 이 구조를 담은 `bundle_template`에서 시작한다(§8).

- **Bundle 제공 --- 총망라 + 재현 확인 후 전달.** 사용자가 bundle을 요청하면 다음을 **모두** 담고 `README.md`에 구조·재현법·figure↔script↔사용위치 매핑 표를 포함한다: **code**(`data_*`/`lib_*`/`fig_*`/helper/`build_all.py`), **figures**(전 PNG), **tex**(모든 `.tex` 소스), **deck**(빌드된 `.pptx` 산출물 + **구성안 `plan.tex`** + 재빌드 인프라 `build_deck.py`·`build_lib.py`·`deck_spec.py`·`compute_figs.py`·`skeleton2.pptx`·임베드 figs). **deck은 소스만이 아니라 산출물 `.pptx`와 구성안 `.tex`을 함께 넣는다.** bundle zip 파일명은 `YYYYMMDD_<이름>.zip` 형식으로, 맨 앞에 전달 날짜(연월일 8자리)를 붙인다(예: `20260725_PI_CBF_full_bundle.zip`). 이 날짜 접두사는 **bundle zip 파일명에만** 적용하고, 그 안의 코드·figure·문서 등 다른 파일명에는 붙이지 않는다. 전달 전 `build_all.py`가 전 figure 스크립트를 통과하는지, deck이 재빌드되는지, 그리고 재생성 산출물의 핵심 수치가 문서 인용 수치와 일치하는지 확인한다(E7) --- 불일치가 있으면 전달하지 않고 먼저 수정한다.

## 8. 새 연구·새 문서 개시 (bundle_template 기반)

새 주제를 시작하거나 사용자가 이전 산출물을 주며 "새로 시작"을 지시하면, `bundle_template`(빈 골격 + 기본 preamble + principles + README)의 **재사용 가능한 infrastructure만 보존**하고 그 위에서 새로 작성한다. 내용(정리·본문·슬라이드·figure)은 이전 것을 재활용하지 않는다(§5a "새 tex/pptx를 bundle infrastructure로 시작"의 실행 절차).

- **개시 순서.** (1) `bundle_template`를 새 프로젝트명으로 복사한다. (2) `Research_Principles_and_Guidelines.md`를 먼저 정독하고, 특히 §5a(공통)와 해당 산출물의 전용 절(`.tex`는 §5b, `.pptx`는 §5c)을 이 연구에 맞게 확인한다. (3) `tex/main.tex`의 preamble·theorem 환경은 그대로 두고, `\title`·project-specific macro만 채운다. (4) `code/`에서 `data_example.py`를 `data_<plant>.py`로 복제해 이 연구의 plant를 정의하고(E10), figure가 필요해지면 `fig_example.py`를 `fig_<내용>.py`로 복제한다(E8). (5) deck이 필요하면 `deck/plan_template.tex`를 구성안으로 쓰고 `build_deck.py`로 빌드한다(§5c). (6) `README.md`의 구조·빌드법 표를 이 연구의 figure·문서로 갱신한다.

- **보존 대상 (고정 자산, 새로 만들지 않음).** tex: preamble(package, 렌더 시 `fontspec`+Noto Serif CJK KR / 전달 시 `kotex`)·theorem 환경·generic macro. code: `lib_style.py`·`build_all.py`·data/lib/fig 3계층 구조와 명명 규칙(E8·E10). deck: `build_lib.py`·`skeleton2.pptx`·`build_deck.py`·`deck_spec.py`·구성안 포맷(§5c F5·F9). principles·README 골격.

- **비우는 대상 (새로 작성).** 모든 정리·정의·증명·본문·abstract, 모든 슬라이드 내용, 모든 figure. template의 `main.tex`는 빈 Overview·Setting section만, `plan_template.tex`는 빈 첫 슬라이드만 남긴 상태다 --- build가 실제로 도는 최소 골격이며 내용은 채워 넣는다.

- **plant·인증값의 출처.** plant·gain·scenario는 project files(§7)에서 ground하고, 새로 정의하는 값은 `data_<plant>.py`에 두어 single source로 관리한다. 정량 주장은 검증 후에만 쓴다(§4·6, E7).

# Project instructions (read first)

- Read HANDOFF.md in full before any work: research goal and invariants, the user's standing rules, the system, the current state, the port steps and the next steps.
- Writing and verification rules: docs/Research_Principles_and_Guidelines.md. Sections 5, 5a, 5b and 6, rules K7, K7b and K8 of 5c, and the rendering, batch-edit and bundle rules of 7 apply; the rest of 5c applies only to slides; sections 1-4 belong to another project and do not apply.
- Reply in Korean (no 반말, polite -습니다 style), technical terms in English, no emoji, no Chinese characters, key point first, concise and causal.
- Code comments in concise English, no separator lines.
- Never change the design (cost, observation, action, network, loss, hyperparameters, start distribution, evaluation rule) or start a training run without the user's approval. Pre-register every experiment in records/<name>_plan.md with its sha256 prefix in records/<name>_plan.hash before running it.
- One seed only unless the user says otherwise ("앞으로 내 별도 지시 없이는 seed별 학습은 절대 하지마. 한 seed만 할 것. 시간 낭비임."). Small pilots are fine.
- Report numbers exactly as in the records. Training reports include data plots and rollout GIFs; give interim reports during long runs.
- Before reporting a document change as done: ./build_tex.sh spec (all gate counts 0), copy build/spec.pdf to spec.pdf, and cd code && python3 check_sync.py must print "sync findings: 0".
- Do not invent citations; check novelty claims with a web search. Do not send the user's personal information to external services.

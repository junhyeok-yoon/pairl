# Grasp design study (descriptive, not registered; no training): proposal for after attempt 4

Data: records/l2_grasp_design.json (eval/diagnostics/l2_grasp_design.py; this PC's CPU, parallel launcher; the 256 original-start evaluation episodes of
pilot O, deterministic and fully sampled, for l2_pilotO_NH at 200 and l2_pilotO_C at 200; 256 near-object starts built in the analysis only) and
records/l2_grasp_failure.json. Heights are the tertiles of the grasp-point height (below 0.121 m, 0.121-0.248 m, above 0.248 m).

## (a) Why grasps of high objects fail
NH 200, deterministic, by height tertile (low / middle / high):
- grasp 0.235 / 0.174 / 0.082; median minimum distance in phase 1 0.0185 / 0.0200 / 0.0317 m; decisions within 2 cm 547 / 299 / 127.
- within 10 cm: tip speed 0.043 / 0.055 / 0.063 m/s; joint speed 0.398 / 0.422 / 0.422 rad/s; hand error |u_ad - u*| 0.877 / 0.776 / 0.677 N m;
  gravity-load ratio (max |u*_i| / ubar_i) 0.124 / 0.130 / 0.124; joint-box gauge 0.702 / 0.680 / 0.663; stiffness about (195, 84, 32) in all bins;
  commanded displacement |dp| 0.024 m in all bins.
- within 2 cm: joint speed 0.383 / 0.288 / 0.282 rad/s, slow share 0.027 / 0.054 / 0.055.
C 200: grasp 0.035 / 0.035 / 0.000; minimum distance 0.0294 / 0.0408 / 0.0452 m.
Reading: high objects fail because the tip does not get within 2 cm as often (the minimum distance is larger and there are fewer decisions inside the ball),
not because of speed (inside the ball high objects are slower), the hand (its error is smaller there), the gravity load (equal across bins) or the joint
box (the gauge is lower, i.e. further from the limits, for high objects). The commanded displacement is the same small size (2.4 cm) in all bins while
the tip moves faster near high objects (0.063 against 0.043 m/s), so the aim is less precise there. It is a learning problem of the final approach.

## (b) Candidate 1: phase-1 speed term near the object
Rate in phase 1 (object not held): c = |p - o|/0.1 + kappa s(d) min(max(0, |dq|_inf - 0.1)/0.1, 3), s(d) = max(0, 1 - d/0.05), d = |p - o|; zero once held.
Cost audit (as for revisions 14 and 15):
- Target state unchanged: at the grasp point at rest (d = 0, |dq|_inf <= 0.1) the added term is 0; after the grasp it is 0, so phases 2-4 are unaffected.
- Dominance of approaching over hovering: at the edge of the ring (d = 0.05) the rate is 0.5 at any speed. Inside the ring at any speed the rate is at most
  d/0.1 + 3 kappa (1 - d/0.05), which is at most 0.5 for every d in [0, 0.05] if and only if kappa <= 1/6. With kappa = 1/6 being closer is never more
  costly than hovering at 5 cm, whatever the speed; slow approach is strictly cheaper. kappa = 0.25, 0.5, 1.0 violate this (fast entry costs up to 0.75,
  1.5, 3.0 against 0.5 at the edge): at the current inside speed (about 0.4 rad/s, which saturates the cap) they would reward staying out of the ball,
  the same kind of defect as the release incentive of revision 13. Proposal: kappa = 1/6.
- Size on the visited states (NH 200, fully sampled): the term is active on 42.3% of the free phase-1 decisions (C 200: 20.1%); at kappa = 0.25 it adds
  11.6% of the phase-1 distance cost (0.173 per episode); at kappa = 1/6 about 7.8% (proportional).
- Expected learning signal: today slowing down is rewarded only through the grasp event, which needs within 2 cm, slow and closed at once; inside the
  ball only 2.7-5.5% of decisions are slow, so the signal is sparse. The speed term gives a dense, graded signal for slowing on every decision within
  5 cm (17.0 per original-start episode for NH, 8.1 for C, fully sampled), i.e. about 8,700 decisions per 1024-episode batch (512 original starts; the
  restarts begin after phase 1), while the grasp event occurs in about 15% of the original starts (about 78 per batch).

## (b) Candidate 2: phase-1 start states near the object
A share of the original starts begins phase 1 at rest with the tip 2-8 cm from the grasp point (posture by best-margin inverse kinematics inside the box,
above the block); the cost and every other part unchanged; evaluation stays on the original starts.
Audit:
- Feasibility: all 256 constructed starts lie inside the joint box (margin min 0.139, median 0.619).
- Cost: unchanged, so no new incentive; only the start distribution changes (like the restart share of pilot R). The phase-1 cost of these episodes is
  small (NH 0.873, C 1.223 per episode), so they add little variance to the batch return; their advantages concern the final approach only.
- Current brains from these starts (deterministic): grasp NH 0.219, C 0.051; decisions within 2 cm per episode NH 4.1, C 1.4; slow share within 2 cm
  NH 0.053, C 0.028. Starting close raises exposure only modestly (NH 3.8 -> 4.1 decisions within 2 cm per episode) and does not by itself make the arm
  slow down.
- Expected learning signal: more final-approach states per batch and more grasp events, but the slowing still has to be learned from the sparse grasp
  event. It complements candidate 1 rather than replacing it.

## Proposal (for after attempt 4; each a design change needing approval and pre-registration)
1. A short pilot from the attempt-4 checkpoint: control vs the speed term with kappa = 1/6 (cost revision 17), paired evaluation on the original starts;
   decision: grasp at the end of phase 1 and total cost (bootstrap CI), with the audit above re-run on the pilot's states.
2. If the speed term passes, a second arm adding near-object starts (for example a quarter of the original starts), decided the same way.
Not proposed: a larger action range or joint box (no structural limit was found).

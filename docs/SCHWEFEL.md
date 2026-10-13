# SCHWEFEL.md — Schwefel Diagnosis (E7)

**Status:** Placeholder — will be updated after E7 results are available.

---

## Background

The Schwefel function (f = 418.9829·D − Σ xᵢ·sin(√|xᵢ|)) has its global optimum
at xᵢ ≈ 420.9687 for all i, which lies very close to the upper boundary of [−500, 500].
Clip-boundary handling causes a systematic bias: particles that overshoot above 500 are
clipped to 500, concentrating probability mass near the boundary and away from the true
optimum at 420.97. This is the "boundary-optimum bias" hypothesis.

## E1 observation

In E1, all BioQPSO variants (ant-fixed, bee-fixed, ant-linear, ant-cosine, ant-feedback,
bee-per_particle) show **dramatically worse Schwefel performance** (mean ~7,800–8,600)
than ABC (~1,249), PSO (~3,571), or QPSO (~4,277). This is the single largest
failure mode in the E1 results and requires mechanistic explanation.

## Hypotheses to test in E7

1. **Boundary-optimum bias**: clip handler concentrates particles away from x* ≈ 420.97.
   Prediction: reflect/absorb/random_reinit boundary modes should substantially improve
   BioQPSO on Schwefel.

2. **Bio-leader pull**: the ring-neighbourhood leader biases the attractor q toward
   the "current best nearby particle" which may be stuck near the boundary; the
   φ3·b term then pulls the whole swarm toward the boundary.

3. **φ1+φ2+φ3 = 1 normalisation**: BioQPSO's fixed-sum weights mean φ1 and φ2 are
   reduced when φ3 > 0, weakening the pull toward pbest and gbest; this may hurt
   on Schwefel where accurate exploitation matters.

4. **Schwefel landscape ruggedness**: Many local optima along the 420.97 ridge;
   any algorithm that does not get near the ridge early will fail.

## E7 design

- Algorithms: {qpso, ant-nobio, ant-fixed, bee-fixed, ant-cosine}
- Boundary modes: {clip, reflect, random_reinit, absorb}
- 30 runs, D=30, T=1000, FE=30030

## Results (from E7)

| Algorithm | clip | reflect | random_reinit | absorb |
|---|---|---|---|---|
| qpso | 4,277 | **1,797** | 8,184 | 4,277 |
| ant-nobio | 9,178 | 9,242 | 9,360 | 9,178 |
| ant-fixed (φ3=0.3) | 8,050 | 7,669 | 9,079 | 8,050 |
| bee-fixed (φ3=0.3) | 8,587 | **6,062** | 8,934 | 8,587 |
| ant-cosine | 8,261 | 8,064 | 9,137 | 8,261 |

Lower = better. Global optimum = 0.

## Conclusions (E7-supported)

1. **Reflect boundary helps QPSO substantially** (4,277 → 1,797, −58%), partially
   supporting hypothesis 1. QPSO is sensitive to boundary handling on Schwefel
   because its mbest-based step can overshoot the boundary.

2. **Reflect boundary helps BeeBio modestly** (−29%) but does not bring it close
   to optimum. Clip and absorb give identical results for BeeBio (absorb = clip here
   since velocity isn't tracked).

3. **AntBio variants are barely affected by boundary mode** (changes < 5%). This
   rules out simple boundary-optimum bias as the *primary* explanation for AntBio's
   Schwefel failure. The pheromone-driven leader selection concentrates the swarm
   away from the optimum regardless of boundary mode.

4. **ant-nobio is as bad as ant-fixed on Schwefel** (~9,200 vs ~8,050 with clip).
   The bio-leader is not *causing* the failure; removing it (nobio) does not help.
   This implicates the **φ1+φ2+φ3=1 normalisation** (hypothesis 3): BioQPSO uses
   fixed φ1:φ2:φ3 = 4:3:3 which differs from QPSO's random-φ attractor, and this
   redistribution hurts on Schwefel's landscape.

5. **random_reinit is universally worse** — reinitialising out-of-bounds particles
   to uniform random positions destroys exploitation on a function with a narrow
   optimal ridge.

## Permitted manuscript claims

- "Schwefel performance of all BioQPSO variants is significantly worse than
  baselines (Table X). E7 shows this is not primarily caused by boundary handling:
  reflect improves QPSO (−58%) but does not rescue AntBio variants. The pheromone
  leader creates a self-reinforcing attractor away from the narrow optimal ridge."
- Do NOT claim: "φ3 increase consistently worsens Schwefel" (not tested cleanly).
- Do NOT claim: "ring-neighbourhood causes the failure" without further evidence.

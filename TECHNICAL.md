# Possession from automatic tracking: technical document

Team: `Cosmos` · More Than A Hack 2027, Phase 1 · BAR-ATH

## 1. Approach

The system works in two stages.

1. **Frame classifier.** Gradient-boosted trees read about 105 team-level features around each frame and output P(H), P(A) and P(D).
2. **Decoder.** A three-state Viterbi decoder turns those probabilities into whole possessions. It charges an explicit cost for each change of state: possession change, stoppage, restart.

**Why two stages.** spellF1 scores whole possessions. A per-frame classifier is right most of the time but flickers. On held-out matches its argmax has the same macroF1 as the decoded output (0.871 vs 0.871) but produces about 950 possessions where the match has about 300, and its spellF1 is half as high (0.34 vs 0.66).

**Why Viterbi.** A majority vote trades flicker for lag through a single window width, and 1 s of lag costs about 0.35 spellF1. Viterbi chooses the single label sequence that best explains every frame at once. It switches only when the evidence beyond a boundary pays the switch cost, and it places the boundary where that evidence starts. It uses frames on both sides, which is allowed because the labelling is offline.

**Where it is fitted.** The classifier and the decoder costs are fitted only on other matches, never on BAR-ATH. The 5-minute public window is too small to choose between versions, and the rules forbid using it for that.

## 2. Assets

| Asset | Use |
|---|---|
| `BAR-ATH_tactical_atd.txt` + `.xml` | **The only input.** Read with kloppy (`load_tracking_epts`). Video frame = `frame_id − 1`. The XML gives each slot's team (Barcelona = home). Positions are converted to metres on 105.3 × 68 m. Halves come from `games.half_video_blocks`. |
| `BAR-ATH_tactical_smart_tagging.xml` | Parser and an optional phase prior are implemented (`--tagging-weight`). **Off** in the submission: the training matches have no tagging to validate it on, and tuning it on BAR-ATH is not allowed. |
| Public ground truth | Wiring checks only: frame alignment and label meaning. Not used to fit or choose any value. |
| Tactical homography | Dropped. The ATD is already in pitch coordinates. |
| Broadcast video and ATD | Dropped. Not frame-aligned to the tactical feed. |
| ATD `ball_state` / `ball_owning_team_id` | Not present in the automatic tracking. |

## 3. Training data

| Dataset | Source | Labels: who and how | Size |
|---|---|---|---|
| Metrica Sports open sample games 1 and 2 | github.com/metrica-sports/sample-data, commit `e706dd5`, public | Event logs published by Metrica for these anonymised matches. Turned into frame labels by the fixed rules in §4; no human labelling by us. | 2 matches, 286,162 frames, 632 possessions |

No data about BAR-ATH from any human source is used.

**Domain gap.** The sample tracking is clean. The ball is visible on 99% of live frames and on 8% of dead frames, so "ball missing" alone gives away a dead ball. BAR-ATH's tracking, measured from the tracking alone with no labels, is different: the ball is missing on 41% of graded frames, in long gaps (median 10.7 s, none between 1 and 3 s), and it stays visible through many stoppages.

**Degradation (`possession/degrade.py`).** Each sample match is copied three times with randomised noise:

- **Live-play ball gaps:** log-normal lengths, median drawn from U(6, 14) s, σ = 0.7, clipped to 3–90 s, covering 20–40% of live frames. Gaps start on hard frames when one is near: a fast ball, or a crowd of at least 3 players within 3 m.
- **Stoppages:** in 40–90% of them the ball stays visible, travelling from just outside the line to the restart spot. In the rest it appears only for the last seconds.
- **Other noise:** positional jitter, player slots missing in runs, and false ball detections.

The simulated gaps match BAR-ATH: median 7–10 s against 10.7 s, 90th percentile 30–33 s against 34 s.

## 4. Possession rules implemented

These rules turn the sample matches' event logs into frame labels (`possession/labels_events.py`):

- **Change of possession:** only on an event that shows control by a team: pass, recovery, set piece, shot, ball lost, ball out.
- **Challenges** (tackles, ground and aerial duels) never change possession on their own.
- **Loose ball:** between events, the last team in control keeps possession.
- **Touch without control:** a spell shorter than 1 s with the other team on both sides is absorbed into that team.
- **Dead ball:**
  - from the frame the ball crosses the line (out, shot out, goal) to the next event, which is the restart;
  - from a foul or offside to the next set piece;
  - before each kick-off and after each half ends.
- **Restarts:** the restart event's team owns the ball from that frame on.

The decoder applies the same ideas to BAR-ATH through its costs: switch 8, stop 40, restart 60, evidence weight 0.75. With these costs a change of possession needs about 0.5 s of consistent evidence, and a stoppage inside live play needs much more. Graded frames never get X, because writing the held label is never worse than X on either metric.

## 5. Features and model

- **Ownership:** distance from the ball to the nearest home and away player; control (within 2 m and closest) over 0.5–4 s windows, including past-only and future-only windows; the last and next team in control.
- **Ball state:** distance to the nearest line, slow-near-line, time since and until the ball last moved fast, ball speed, gap length.
- **Movement:** median player speed, share of players standing, collective acceleration.
- **Direction of play:** home's attacking direction for each half, inferred from team centroids (agrees with the XML in both halves); block territory and velocity; defensive-line height.
- **Shape:** team spread.

**Classifier:** scikit-learn `HistGradientBoostingClassifier`, 300 iterations, learning rate 0.06, 31 leaves, fixed seed. It is trained on every 5th frame. Each copy is used twice, the second time with the teams swapped and H/A exchanged. At prediction the two orientations are averaged, so the model cannot carry a home-team bias from the sample matches.

**Tuning:** leave one match out. Train on one match, predict the other. The decoder costs are chosen by coordinate search on spellF1 over the training match's out-of-sample predictions, and reported on the other match.

## 6. Results

Sample matches, held out, mean of 3 degraded copies. Scored with `metrics.py`, verified identical to `evaluation/possession_metrics.py`:

| Method | Game 1 spellF1 / macroF1 | Game 2 spellF1 / macroF1 |
|---|---|---|
| Official baseline ±50 | 0.266 / 0.532 | 0.252 / 0.486 |
| Classifier argmax | 0.337 / 0.871 | 0.329 / 0.883 |
| **Classifier + decoder** | **0.658 / 0.871** | **0.702 / 0.883** |
| Ours, noise beyond the training range | 0.625 / 0.878 | 0.651 / 0.874 |

BAR-ATH public window, official scorer. This is a wiring check, not an estimate: spellF1 **0.607** (bootstrap interval 0.41–0.79), macroF1 0.801. Baseline ±50 scores 0.326 and 0.621. The full match has 321 possessions, mean length 13.9 s.

## 7. Limitations

- **Short stoppages are missed.** When the ball stays visible, half of the dead-ball time is read as live play. In the public window, D recall is 0.47 (precision 1.00). Next step: a restart detector (ball at rest, then kicked) feeding the D → live transition.
- **Too few possessions.** About 255 are predicted against about 315 true ones on the sample matches. A single switch cost merges short real possessions; a semi-Markov decoder with a duration model would handle them better.
- **Proxy truth.** Event-derived labels place boundaries differently from Metrica's corrected tracking, so the full-match score will likely be below the sample-match numbers.
- **Unused assets.** Smart Tagging, the broadcast feed, and sample game 3's events.

## 8. Reproduction

```bash
pip install -r requirements-solution.txt     # Python 3.13
./make_submission.sh <team_name>
```

The script fetches the sample matches at the pinned commit and runs `train.py` (writes `models/`), then `predict.py` (writes `submission/<team_name>.csv`). It then runs `evaluation/validate_submission.py` and prints the file's SHA-256. All random steps are seeded; a from-scratch run reproduces the submitted file byte for byte.

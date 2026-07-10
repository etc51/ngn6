# NGN6 strategy audit - 2026-07-10

## Scope

This audit covers every completed VPS paper trade in `data/paper_events.jsonl` and joins it to
the nearest accepted decision, live order-book snapshot, sampled market path, recent trades,
and look-ahead-safe completed 1m/5m/15m candles fetched from T-Invest.

- Runtime/source commit under analysis: `c4991f593a2b7ec4bef2857003f3a966fccb5767`.
- Trade period: 2026-07-03 12:51 MSK through 2026-07-06 16:16 MSK.
- Completed position lifecycles: 182 opens and 182 closes; no unmatched server events.
- Detailed rows: `reports/strategy_audit_2026-07-10/paper_trade_forensics.csv`.
- Machine summary: `reports/strategy_audit_2026-07-10/strategy_audit_summary.json`.
- Caveat: these historical events predate commit attribution, so their exact deployed revision
  cannot be proven event by event. The P&L and market-data joins are exact; code-cause findings
  are verified against the current repository and against observable trade behavior.

## Result

| Metric | Recorded paper | Bid/ask + configured slippage |
| --- | ---: | ---: |
| Trades | 182 | 182 |
| Gross P&L | -2,678.63 RUB | -2,678.63 RUB |
| Commission charged | -7,589.90 RUB | -7,589.90 RUB |
| Additional spread/slippage | 0.00 RUB | -12,024.96 RUB |
| Net P&L | **-10,268.53 RUB** | **-22,293.49 RUB** |
| Profit factor | 0.099 | 0.024 |
| Expectancy per trade | -56.42 RUB | -122.49 RUB |
| Winning trades | 17 (9.34%) | 3 (1.65%) |
| Maximum drawdown | 10,268.53 RUB | 22,293.49 RUB |

All four trading dates were negative. Long trades lost 5,126.90 RUB; shorts lost 5,141.62 RUB.
The state reconciliation is exact: gross P&L minus both-side commission equals paper-state P&L.

## Every Trade

The CSV contains one row per completed position and 250+ scalar fields. It includes timestamps,
side, lots, entry/exit/stop, model target and component scores, all recorded feature values,
top-1/3/5/10 book pressure, microprice, walls, raw and tick-rule trade flow, candle indicators,
MFE/MAE, execution costs, session/candle freshness, result flags, and commit attribution.

The most important aggregate facts from those rows are:

- 83 trades were gross winners, but commission converted 66 of them into net losers.
- Median commission break-even was 2.59 ticks; median realistic break-even was 6.64 ticks.
- Median MFE was only 0.50 tick. The strategy usually did not create enough excursion to pay costs.
- Median holding time was 60.82 seconds; 89 trades lasted under 60 seconds.
- 161 exits were opposite-ML flips and lost 6,321.37 RUB. Of them, 116 occurred in under two
  minutes and lost 4,425.02 RUB.
- 99 entries occurred within 60 seconds of the previous close and lost 6,772.58 RUB.
- 157 entries occurred less than 45 minutes after a losing trade and lost 9,237.84 RUB.
- The longest loss streak was 43 trades.

## Root Causes

### 1. Stop evaluation uses price history from before entry

The runtime evaluates the entire current 15m candle high/low after opening a position. That candle
contains extrema formed before the entry. Thirteen of sixteen hard stops closed in about five
seconds, and all thirteen stop levels had already been crossed by completed 1m candles before the
entry. Those thirteen trades lost 2,729.92 RUB. Two repeated bursts reopened and stopped the same
direction every ten seconds.

Required correction: maintain post-entry high/low or evaluate stops from ticks/bars whose timestamp
is strictly greater than `opened_at`. Never apply an earlier candle extreme to a new position.

### 2. Exit cadence is incompatible with entry horizon

Three ML confirmations are three five-second runtime cycles, not three completed bars. The bot can
enter on a multi-minute forecast and exit 10-15 seconds later on normal probability noise. This
creates churn whose typical movement is smaller than round-trip cost.

Required correction: express hysteresis in completed bars and position state. Except for a hard
stop, require at least two completed 1m bars before a flip exit; feed holding time, unrealized R,
post-entry MFE/MAE, stop distance, and session time into the exit policy.

### 3. Costs dominate an already negative gross signal

Gross trading was already negative by 2,678.63 RUB. Commission added 7,589.90 RUB of loss. Paper
fills occur at midpoint/signal price and omit spread, queue position, and slippage, so paper P&L is
optimistic. Charging executable bid/ask and the configured 4 bps slippage per side reduces the
result to -22,293.49 RUB.

Required correction: paper opens at ask for long and bid for short, closes on the opposite side,
and charges measured/assumed slippage. An entry must have predicted net MFE comfortably above the
full cost. With this sample, a gross target below roughly 14-15 ticks cannot cover median cost plus
the configured eight-net-tick safety margin.

### 4. Microstructure inputs are incomplete and not predictive

All 182 entries recorded every exchange trade side as `unknown`; directional flow, signed volume,
and trade pressure therefore collapsed to neutral values. A tick-rule reconstruction was possible,
but filtering on it remained negative. Book pressure aligned by at least 0.18 produced 18 trades,
zero winners, PF 0.0, and -759.52 RUB. Single-snapshot pressure is not an edge in this sample.

Required correction: fix SDK enum mapping, record availability/staleness explicitly, and do not
substitute missing flow with a meaningful zero. Validate candle-only, matched-microstructure, and
availability-aware models as separate OOS ablations. Use pressure persistence over multiple live
snapshots, not one snapshot.

### 5. Session and freshness controls were not enforced historically

Thirty-five entries occurred outside the configured 10:00-23:45 MSK session. Their 1m candle
context was over an hour old; they lost 1,460.61 RUB. Current configuration declares session,
daily-loss, consecutive-loss, hard-stop, and time-stop limits, but several are not applied by the
runtime path.

Required correction: enforce the exchange calendar and session gate before signal generation,
then enforce and persist cooldown/risk lock state. The current 45-minute loss cooldown would have
selected only 22 observed trades and reduced loss to -1,089.02 RUB (PF 0.160), but it still would
not create positive expectancy. A three-loss daily lock would reduce the observed loss to
-763.33 RUB (PF 0.345), also still negative.

### 6. Confidence is uncalibrated

Raising the historical score threshold does not repair the strategy. At 0.62, 28 trades lost
1,206.38 RUB (PF 0.270). At 0.68, all seven trades lost 616.97 RUB (PF 0.0). Every tested ADX,
spread, volatility, time, and side-aligned book-pressure bucket was negative. A broad in-sample
single-feature threshold scan with at least 20 trades and three dates found no positive subset.

Required correction: select thresholds by calibrated expected net R on a temporal calibration
window, not by accuracy or a fixed probability. Report Brier score, log loss, ECE, and reliability
by side/regime.

## ML And Validation Defects

The current system is fail-closed, but it is not yet a valid self-learning loop.

- Runtime decisions default to `label_matured=false`; no outcome-maturity updater exists. If later
  accepted, the current decision-label path clones the bot action instead of learning realized
  outcome.
- Configured triple-barrier labeling is not the production training path; training primarily uses
  fixed-horizon terminal close labels.
- Daily oracle selects ex-post opportunities and copies labels across intervals, creating
  post-outcome contamination risk; `latest_oracle_labels.csv` overwrites history.
- Candidate promotion trains on the full set and replays the already-trained model across folds.
  It does not retrain each purged fold, so this is not honest walk-forward OOS.
- Backtest and runtime policies differ: backtest omits several 5m, exhaustion, microstructure,
  liquidity, and exit-hysteresis gates.
- Runtime and training construct 5m/15m bars differently; runtime can include incomplete bars and
  duplicate direct and aggregated bars. This is material train/serve skew.
- Shadow evaluates directional argmax and oracle score rather than exact deployable gates and
  executable fills; candidate and dataset hashes are not frozen.
- Promotion metadata in the training report and serialized artifact can diverge.
- `paper_state.json` retained its first commit hash while later marks updated the timestamp. The
  audit change replaces stale attribution on every write and adds a regression test.

Current candidate status on 2026-07-10: schema/features v2 and all four heads are present, but only
4,739 entry examples versus 5,000 required; status `rejected`, promotion score -0.1104, and zero OOS
trades in its promotion metrics. Current shadow has 213 matched labels, PF 0.327, average -1.0735%,
and only three days versus ten required. Lack of 261 examples is not the main blocker; measured
expectancy is negative.

## Strategy V2 To Test

This is a research specification, not a claim of positive expectancy.

1. Regime layer: use only completed 15m and 5m bars from one shared train/serve transformer. Test
   trend and mean-reversion experts separately. Do not deploy an ADX rule just because it is common;
   the present sample shows no positive ADX bucket.
2. Opportunity layer: train cost-aware triple-barrier labels from one causal feature snapshot at
   event start. Barriers and horizon are ATR-scaled. Predict `P(trade)` and expected net MFE/MAE.
3. Direction layer: compare direct three-class and two-stage models. Calibrate each head on a later
   temporal window and restore real class priors after balancing.
4. Execution layer: require live known-side flow, one-tick spread where possible, persistent aligned
   microprice/book pressure across three snapshots, sufficient executable depth, and predicted gross
   excursion of roughly 14-15 ticks or more. Start with one lot.
5. Exit layer: stops use only post-entry observations; no flip exit before two completed 1m bars;
   exit-head confirmations are bar-based; implement the configured 12-bar time stop and persist all
   state across restarts.
6. Risk layer: 45-minute cooldown after a loss, 10 minutes after another exit, daily loss cap,
   three-loss lock, and two-hard-stop lock must be executable code with tests, not configuration only.

## Acceptance Gates

Do not promote a strategy until all of these pass on a frozen dataset/model hash:

- Causal labels, no-look-ahead tests, identical train/serve features, and exact runtime-policy replay.
- Eight purged walk-forward folds with retraining inside each fold and untouched final test data.
- At least 150 OOS trades and 20 per fold; total PF >= 1.25; median fold PF >= 1.15.
- Positive-fold share >= 70%; expectancy >= 0.08R; average net >= 8 ticks.
- Total OOS drawdown <= 8% and single-fold drawdown <= 5%.
- Bootstrap 95% lower confidence bound for net expectancy above zero.
- At least ten shadow days with 50+ matched mature deployable signals, PF >= 1.15, realistic fills,
  and drawdown <= 3%.

The correct target is robust out-of-sample expectancy after all costs, not the maximum backtest
number. On the currently available trade sample, no positive strategy regime has been demonstrated.

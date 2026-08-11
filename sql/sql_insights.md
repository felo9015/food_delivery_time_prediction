# SQL Insights

## Purpose

This document extends the analysis of the 5 required questions (`sql/sql_queries.sql`) with additional business insights that surfaced while exploring the dataset, and records the key methodological decisions behind Questions 3 and 5 — where a threshold given or implied by the prompt was replaced with a data-driven one — so a reviewer does not have to reconstruct them from the exploration notebook.

## Additional Business Insights

### 1. Does weather condition affect delivery time — and does it show up in customer ratings?

**Business question:** How much slower are deliveries under adverse weather, and do customers penalize the platform's ratings for it?

**Why it matters:** If delivery time degrades meaningfully under rain or storms but ratings do not reflect that, two different but related actions are worth considering: adjusting the ETA shown to customers during bad weather (so a slower-than-usual delivery is not perceived as underperformance), and checking whether couriers are being fairly evaluated for delays that are outside their control.

```sql
SELECT
    weather_condition,
    AVG(delivery_time_min) AS avg_delivery_time_min,
    AVG(delivery_rating) AS avg_delivery_rating,
    COUNT(*) AS n_deliveries
FROM deliveries
GROUP BY weather_condition
ORDER BY avg_delivery_time_min DESC;
```

On the synthetic dataset, this shows a clear gradient in delivery time — Stormy (~55 min) > Rainy (~48 min) > Cloudy (~41 min) > Clear (~39 min) — while `avg_delivery_rating` stays flat at roughly 4.27-4.28 across all four conditions. Delivery time is weather-sensitive; ratings do not visibly move with it. That gap is itself worth investigating on the real dataset: it could mean customers are already weather-aware, or it could mean rating data is not capturing a real service issue.

### 2. Does order size affect delivery time?

**Business question:** Do orders with more items take meaningfully longer to deliver?

**Why it matters:** If larger orders are a real driver of delay, that argues for dedicated prep/dispatch handling for bigger orders, or for factoring order size into the ETA shown to the customer. If they are not, effort spent optimizing around order size would be better spent elsewhere (distance, traffic, weather — the factors already shown to matter more).

```sql
SELECT
    CASE
        WHEN o.items_count <= 2 THEN '1-2 items'
        WHEN o.items_count <= 4 THEN '3-4 items'
        ELSE '5+ items'
    END AS order_size_bucket,
    AVG(d.delivery_time_min) AS avg_delivery_time_min,
    AVG(o.order_value) AS avg_order_value,
    COUNT(*) AS n_orders
FROM deliveries d
JOIN orders o ON d.delivery_id = o.delivery_id
GROUP BY order_size_bucket
ORDER BY avg_delivery_time_min DESC;
```

On the synthetic dataset, `avg_delivery_time_min` is essentially flat across buckets (~42 min regardless of order size), even though `avg_order_value` scales with item count as expected. This is a useful negative result: it suggests delivery time here is driven by logistics factors (distance, traffic, weather) rather than order size, which would redirect optimization effort away from prep-time-per-item toward routing and dispatch.

## Methodology Notes

**A data-driven minimum-deliveries threshold, not a fixed number — applied consistently in Question 3 and Question 5.** Both questions need to decide how much data is "enough" before trusting a number computed from it (an average in Question 3, a regression slope in Question 5). The prompt gives Question 3 a fixed cutoff ("at least 50 deliveries") and gives Question 5 no cutoff at all. Rather than keeping an arbitrary given number in one place and inventing another arbitrary number in the other, the same rule is used for both: couriers are ranked by delivery volume, and the threshold is set to the smallest delivery count found among the top 80% of couriers by volume (equivalently, the 20th percentile of the per-courier delivery-count distribution). This makes the threshold adjustable to whatever data is actually available, instead of depending on numbers that only happen to be reasonable for this dataset's specific size. In Question 3 it is computed directly in SQL with window functions (`ROW_NUMBER() OVER (...)` plus `COUNT(*) OVER ()`); in Question 5 it is computed in Python (pandas), since Python is already required there for the regression itself.

On this synthetic dataset, both questions converge on the same number — 123 deliveries out of 15 couriers — since both start from the same courier/delivery counts. The two questions land on different outcomes with that threshold, though, which is itself informative. In Question 3, the stricter threshold happens to produce the exact same top-10 list a fixed `>= 50` would have: courier `7` (119 deliveries) was already the slowest of the couriers passing the looser threshold and would have been trimmed by `ORDER BY` + `LIMIT` regardless, so excluding it earlier via `HAVING` changes nothing about the final business answer. In Question 5, the same threshold matters more: it excludes that same courier `7` — the one deliberately built with a real increasing trend, at 119 deliveries, just under the 123 cutoff — so the final significant-trend list comes back empty. Both outcomes are reported as-is, not adjusted to force a particular result; see `sql_exploration.ipynb`, Questions 3 and 5, for the full reasoning and the side-by-side comparison against the fixed thresholds. The trade-off worth flagging for a production version of this analysis: a principled, reproducible threshold is more defensible than a hand-picked number, but it can be conservative enough to filter out a borderline-but-genuine signal along with the noise it is designed to exclude — which argues for treating "just below the cutoff" couriers as candidates for closer monitoring rather than as confirmed non-signals.

**Individual-delivery regression, not monthly aggregation, for Question 5.** The trend analysis fits a linear regression per courier using every individual delivery (`delivery_time_min` against days since that courier's first delivery), rather than first collapsing deliveries into monthly averages. Aggregating to ~6-7 monthly points per courier before fitting a line would have thrown away most of the sample, made the fit far more sensitive to a single noisy month, and produced p-values that are not trustworthy on so few points. Fitting on every delivery keeps the full sample size (100+ deliveries for most couriers) and lets the significance test (`scipy.stats.linregress`'s `pvalue`) mean what it is supposed to mean.

## GenAI Usage Disclosure

**Tool used:** Claude Code, Anthropic's coding assistant.

**What it was used for in Part I:** drafting and iterating SQL syntax for all 5 queries; explaining the semantics of specific clauses (`JOIN` ON-conditions and their failure modes, `WHERE` vs. `HAVING`, multi-column `GROUP BY`, window functions) while building each query incrementally in the exploration notebook; and prototyping the trend-detection methodology for Question 5 and the volume-threshold methodology later reused for Question 3 — including the initial window-function (`LAG()`) approach for Question 5, its replacement with a per-delivery `scipy.stats.linregress` regression, the SQL window-function (`ROW_NUMBER()` / `COUNT(*) OVER ()`) implementation of the same percentile-based threshold for Question 3, and the design of the synthetic dataset used to validate every query end to end.

**How the output was validated and modified:** every query was run against the synthetic dataset in `sql_exploration.ipynb` and checked against deliberately seeded edge cases (e.g., an area with a resolved historical problem vs. one with a recent spike for Question 1; an inactive high-volume courier and a low-volume new hire for Question 3) before being accepted — no query in `sql_queries.sql` was taken on faith. Explanatory text was reviewed cell by cell until the underlying logic was fully understood, not just copied. Three methodology choices were deliberately overridden from what was first suggested, based on independent judgment: the minimum-deliveries threshold for Question 5 was changed from an initial fixed value to the data-driven percentile-based calculation described above, once it was clear a fixed number would not generalize to different datasets; that same data-driven threshold was then applied to Question 3 in place of the fixed "50" given in the prompt, for methodological consistency across the two questions that share this problem; and `scipy.stats.linregress` was chosen over `numpy.polyfit` specifically to obtain a p-value, since the business question is about statistical significance, not just the sign of a slope.

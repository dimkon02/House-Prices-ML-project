## Understanding correlation — worked example

**Data:** 5 houses, Size (sq ft) vs Price ($)

| House | Size | Price |
|---|---|---|
| A | 800 | 100,000 |
| B | 1000 | 130,000 |
| C | 1200 | 160,000 |
| D | 1400 | 190,000 |
| E | 1600 | 220,000 |

**Step 1 — averages:** Size avg = 1200, Price avg = 160,000

**Step 2 — deviations from average (value - avg), for each column:**

| House | Size dev | Price dev |
|---|---|---|
| A | -400 | -60,000 |
| B | -200 | -30,000 |
| C | 0 | 0 |
| D | +200 | +30,000 |
| E | +400 | +60,000 |

**Part A — numerator: do the two columns move together?**
Multiply Size dev x Price dev, row by row, then sum:
(-400)(-60000) + (-200)(-30000) + 0 + (200)(30000) + (400)(60000) = 60,000,000
-> Every product is positive because Size and Price are always on the *same side*
of their own average together (both above, or both below).

**Part B — denominator: how spread out is each column on its own?**
Square each column's deviations *separately* (never mixing Size with Price), sum each:
- Size: 400² + 200² + 0² + 200² + 400² = 400,000
- Price: 60000² + 30000² + 0² + 30000² + 60000² = 9,000,000,000
Multiply the two totals, take the square root:
sqrt(400,000 x 9,000,000,000) = 60,000,000

**Final step — divide A by B:**
60,000,000 / 60,000,000 = 1.0 -> perfect positive correlation

**Key distinction:**
- Part A pairs the two columns together (X dev x Y dev, same row) -> captures the relationship
- Part B looks at each column in isolation (X squared alone, Y squared alone) -> 
  captures spread, used only to rescale Part A into the -1 to +1 range

This is exactly what `.corr()` does automatically for every column pair in a dataframe.

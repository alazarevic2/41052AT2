# Individual Reflection: Anna – LPT

Worst case

LPT’s worst case is Graham’s family: two jobs of each size (2m-1, 2m-2, \ldots, m+1), followed by (m+1) jobs of size (m). For two machines, the smallest example is (3,3,2,2,2). LPT places the two 3s on separate machines, then assigns the 2s, finishing at 7. The optimum groups the 3s together and finishes at 6. Generally, LPT finishes at (4m-1) against an optimum of (3m), achieving the tight ratio (4/3-1/(3m)). I implemented a generator for this family and confirmed in E9 that the ratio matched the bound for every (m) from 2 to 10, using exact fractions to avoid rounding errors. Random inputs came nowhere near this bound: across 1,600 instances in E3, the largest error was 9.43%.

Hardest bug

My first implementation used sorted(jobs), which sorts shortest first. The code still produced valid schedules, but it was effectively list scheduling rather than LPT. Hand-worked tests exposed the issue because the makespans differed from my expected results. Adding reverse=True fixed it. The bug was difficult to spot because the output was not obviously invalid, just worse. I also replaced scanning every machine ((O(nm))) with a min-heap ((O(n\log m))). Using (load, index) pairs preserved the original lowest-index tie-breaking, so both implementations passed the same tests.

What I would do differently

I would implement brute-force validation earlier, checking that LPT never exceeds its approximation guarantee on small instances. I planned this initially but skipped it. I would also test the shared timing helper more thoroughly: it incorrectly recorded a crashed worker as a timeout. Finally, after E8 showed that execution order distorted my results, I would randomise run order from the beginning.

What I now understand

I thought the comparison  would reveal that DP is slow but optimal, while LPT is fast but approximate. My experiments revealed a more  trade-offs that i wasnt expecting. 

LPT was faster on every instance but was suboptimal more often than I expected, producing optimal schedules in fewer than half the instances from three jobs per machine onwards. However, its errors were usually small, often just one unit. 

I also learned that DP’s cost depends not only on the number of jobs, but on the number of machines and how often different assignments produce reusable load states. 

This changed how I view approximation algorithms, the important question is not simply whether an algorithm is optimal, but whether the improvement justifies the computational cost.

In my experiments, LPT’s speed and bounded error made it a practical alternative to DP, particularly as the number of machines increased.
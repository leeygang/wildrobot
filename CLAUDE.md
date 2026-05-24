You are the expert in the field of robotics and machine learning. You are responsible for the quality of the direction, design and code.

Shell Commands:
1. NEVER prepend `cd /Users/ygli/projects/wildrobot && ` to Bash commands when the working directory is already `/Users/ygli/projects/wildrobot`. Run commands directly using paths relative to the project root (e.g. `uv run pytest training/tests/test_a.py -q 2>&1 | tail -5`, NOT `cd /Users/ygli/projects/wildrobot && uv run pytest ...`). The `cd && ...` prefix is redundant, bloats permission rules, and `cd <dir> && git ...` triggers an unconditional security advisory in Claude Code. This rule has no exceptions — applies to git, pytest, python scripts, rm/ls, everything.

Code Change:
1. always commit to a local git commit when make the code change, if it is a fix on existing **unpushed** commit, you can use `git commit --amend` to amend the commit.
2. ToddlerBot's code can be found in local ~/projects/toddlerbot do not need to read from internet github.

Design:
1. ALWAYS research the public paper or projects to avoid reinventing the wheel. When discuss the solution, always provide the reference. Prefer to stick with the highest confident solution.
2. make the metrics accurate as the first step when analyzing the data.
3. keep the design simple
4. current design and implementations follows ToddlerBot(projects/ToddlerBot), if we have our own design or implementation, we should have explicit rationale. When design/implementation has to do with robot size, please reflect to WR's size ratio with TB, speed, and timing also needs to normalize.

Result Analysis:
1. When analyze training result, always check with the code from both wildrobot and Toddlerbot as source of truth instead of only reading the doc.
2. Need to find the solid evidence instead of guess.
3. Align with ToddlerBot when analyze and solve the problem unless explicit reason not.
4. the analysis result and proposal has to include ToddlerBot comparision, code confirm, the result and proposal should be explicit instead of hand waving.

Result:
1. Update the training result to CHANGELOG.md after confirming with me, if the CHANGELOG.md file is too big, you can create a new CHANGELOG.md file and link it in README.md.

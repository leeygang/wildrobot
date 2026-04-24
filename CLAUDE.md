You are the expert in the field of robotics and machine learning. You are responsible for the quality of the direction, design and code.
Code Change:
1. always commit to a local git commit when make the code change, if it is a fix on existing **unpushed** commit, you can use `git commit --amend` to amend the commit.

Design:
1. ALWAYS research the public paper or projects to avoid reinventing the wheel. When discuss the solution, always provide the reference. Prefer to stick with the highest confident solution.
2. make the metrics accurate as the first step when analyzing the data.
3. keep the design simple
4. current design and implementations follows ToddlerBot(projects/ToddlerBot), if we have our own design or implementation, we should have explicit rationale.

Result Analysis:
1. When analyze training result, always check with the code as source of truth instead of only reading the doc.
2. Need to find the solid evidence instead of guess.
3. Align with ToddlerBot when analyze and solve the problem unless explicit reason not.

Result:
1. Update the training result to CHANGELOG.md, if the CHANGELOG.md file is too big, you can create a new CHANGELOG.md file and link it in README.md.

# CSEN 166 — Term Team Project for Social Impact (TTP4SI)

**Team:** A2  
**Challenge:** Ardook
**Team members:** Seerat Kakar, Jason Wu, Michael Tiburcio
**Repository:** https://github.com/SCU-CSEN166/ArdookA2
**Last updated:** Oct 28, 2026

> **How to use this README:** Maintain this single file throughout the project. Complete the Week 5 project plan first, then update the appropriate weekly section before each milestone. Be specific and honest: describe what you attempted, what worked, what did not work, and what evidence supports your conclusions. For every week, document each member's actual contribution. If a task was not completed, explain the blocker and next step. Do not include private or sensitive data in this public README.

## Remember - Best Practices

### Data access and privacy

[Explain where authorized data can be accessed. Do not commit personal information, private WhatsApp exports, API keys, credentials, or restricted grant/application records. Use synthetic or anonymized examples in the README.]

### Known limitations

[Keep a current list of what the system cannot reliably do.]

## Week 5 — Scope: Project Plan

**Deliverable during Week 6 Lab:** A project plan (write in this README file) describing your problem formulation, proposed approach, each team member's responsibilities, proposed baseline models with Hugging Face links, proposed evaluation metrics, proposed data with links, and goals for each week.

### 1. Problem formulation and social impact

- **Problem:** Nonprofits spend a lot of time figuring out which grants they can actually apply for. Finding grant listings isn’t the hard part. The hard part is reading each one closely enough to know whether they’re eligible, whether it’s worth their time, and when it’s due. We’re building a system that takes a nonprofit’s profile and a list of grants, determines eligibility and fit, and produces recommendations grounded in evidence from the original grant sources.
- **Intended users:** The person at a small nonprofit who handles grants.
- **Inputs and outputs:** The system takes in a nonprofit's profile and a list of grants, and it returns a ranked list where each grant is marked eligible, not eligible, or unsure, along with a short reason and quotes from the source that back it up.
- **Success criteria:** A grant writer should be able to trust our shortlist without re-reading every grant page. That means the system rarely marks a grant eligible when it isn't, and it says "unsure" when the information is missing instead of guessing.
- **Constraints and risks:** The main risk is the AI making things up, like eligibility rules or deadlines that aren't actually in the grant. The grant data is also messy, and some links may be outdated. Telling an org they're eligible when they're not is worse than missing a grant, since they could waste days on an application.

### 2. Proposed technical approach

- **Pipeline:** This pipeline takes a nonprofit's profile and a spreadsheet of grant opportunities, removes all the opportunities which will fail known eligibility requirements, and ranks the remaining grants via semantic relevance. In this case, each recommendation will include a score (how good the rec is), the eligibility checks, and additional evidence. We use an LLM to generate a brief explanation of the grant fit for a nonprofit.
- **Baseline(s):** To evaluate our pipeline, we'll compare it to a simple keyword-overlap matcher with deterministic eligibility checks. Grants that don't pass requirements like having expired deadlines or incompatible organization types will be excluded when providing recommendations. In addition, if there's missing information in any grant, that'll be marked as unverified. For grants that pass these deterministic checks, they will be ranked by the proportion of nonprofit mission/description keywords which can be found in the grant's own description and requirements. Thus, this would be a good baseline for measuring whether semantic similarity does improve recommendation relevance.
- **Proposed models/tools:**
  - [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — Open source model that converts nonprofit descriptions and grant requirements into embeddings for ranking opportunities using cosine similarity. Selected as a lightweight model suited to semantic search.
  - [FLAN-T5-small](https://huggingface.co/google/flan-t5-small) — Open source model that generates optional explanations of potential fit using retrieved grant evidence. Its explanations will require evaluation and it does not determine eligibility or scores.
- **Alternative approach:** Given that our current pipeline fails, we will try to train our own ML model in order to generate accurate recommendations for any given nonprofit. Currently, we're leveraging similarity search and RAG to see if this implementation can suffice for making sure that nonprofits have the most precise and accurate recommendations for their specific needs.

### 3. Proposed data

| Dataset/source | Link or access instructions                                                                                                        | Purpose    | Size / languages                  | Privacy or usage considerations |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------- | --------------------------------- | ------------------------------- |
| [Ardook Data]  | [URL](https://docs.google.com/spreadsheets/d/1o_9ft5_eBtUpgBe9rj8TeUIXbJkzKGT7QQRXFxfDqdA/edit?gid=0#gid=0) or other instructions] | [Training] | [reference labels for evaluation] | [Notes]                         |

- **Data preparation:** [Cleaning, anonymization, annotation, splitting, deduplication, etc.]
- **Data limitations:** [What is missing, noisy, or uncertain?]

### 4. Proposed evaluation

| Metric / test                    | What it measures    | How it will be calculated | Target / success criterion |
| -------------------------------- | ------------------- | ------------------------- | -------------------------- |
| Precision@5 | How many of the top 5 recommendations are grants the nonprofit should actually get | Count the top 5 grants that are labeled a match for the profile, divided by 5, then average over all test profiles | At least 0.60. We measured 0.43 on the synthetic set with MiniLM, so this is the number we most need to improve |
| Recall@5 | Whether the good grants show up in the shortlist | Matches found in the top 5 divided by the number of matches that exist (capped at 5), averaged over profiles | At least 0.70. We measured 0.55 on the synthetic set |
| False-eligible rate | How often we recommend a grant the nonprofit can't apply for | Recommended grants that fail a known rule (entity type, region, revenue) divided by all recommended grants | 0. This matters most, since a wrong "eligible" wastes the most time. We measured 0 on the synthetic set |
| Evidence match | Whether every quote and deadline we show is really in the grant source | Check that each evidence string appears word for word in the source row of the spreadsheet | 100% |
| Unknown handling | Whether missing information is marked unverified instead of guessed | For test profiles with a missing attribute (revenue left blank, for example), count the grants whose check comes back "unknown" instead of pass or fail | 100% |

- **Evaluation examples:** We have two sets and keep them separate. The first is a synthetic set made to look like the Ardook spreadsheet: 150 fake grants, 200 fake nonprofit profiles, and a label for every pair (match, ineligible, or unrelated). Its labels come from the same eligibility rules the pipeline runs, so it can test the plumbing and the ranking, but it can't tell us whether the rules themselves are right. The second is the real 37-grant tracker. For that one, each of us will write a few nonprofit profiles and mark by hand which grants fit after reading the official source. We use the synthetic set while building and the real tracker for the numbers we report.
- **Human verification:** The three of us will check a sample of recommendations against the official grant page, looking at whether the eligibility status, deadline, and quotes are correct. We'll also read the generated explanations next to the evidence they were given and mark any sentence that isn't supported. If we can reach a grant writer at Ardook or a nonprofit, we'll ask them to review the final shortlist for a few profiles.
- **Failure cases to watch:** (1) The generated explanation says something that isn't in the grant, since FLAN-T5-small is a small model. It never decides eligibility, but a reader could still believe it. (2) Most requirements are free text that we can't turn into rules, so many grants will stay "needs verification" and the shortlist may look less certain than it is. (3) The tracker has old deadlines and links, so a grant can look open when it is closed. (4) The synthetic data is cleaner than the real data, so scores on it will look better than real performance. (5) Similarity can rank a grant high because of shared words in the program name and not because the mission fits.

### 5. Team responsibilities

| Member   | Primary responsibility                   | Specific Weekly contribution                                                                                                                                                      | Planned next task                                                                                                  |
| -------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Jason Wu | Build the initial RAG pipeline prototype | Completed the initial prototype, including grant data ingestion and cleaning, deterministic eligibility checks, semantic similarity scoring, and optional generated explanations. | Begin productionizing the system by scaling the pipeline to process requests concurrently for multiple nonprofits. |

---

## Week 6 — Build: Working Baseline

**Deliverable during Week 7 Lab:** A runnable baseline, setup/run instructions, and 1–2 example inputs and outputs. Update this section with what worked, what failed, and individual contributions.

### Implementation status

- **What runs end-to-end now?** [Describe exactly.]
- **How to run it:** [Commands, dependencies, configuration; refer to Setup below.]
- **Model(s) and data actually used:** [Names, versions, links, and any changes from the plan.]
- **Example 1:** Input: [non-sensitive example] → Output: [result]
- **Example 2:** Input: [non-sensitive example] → Output: [result]

### Attempts and outcomes

| Attempt / change | Worked? (Yes / Partly / No) | Evidence (output, metric, link) | What we learned / next step |
| ---------------- | --------------------------- | ------------------------------- | --------------------------- |
| [Attempt]        | [Status]                    | [Evidence]                      | [Learning]                  |

### Individual contributions during that week

| Member | What I personally implemented/tested | Result (worked / partly / failed) | Evidence (commit, PR, experiment) | Blocker / next step |
| ------ | ------------------------------------ | --------------------------------- | --------------------------------- | ------------------- |
| [Name] | [Specific contribution]              | [Status]                          | [Link]                            | [Next step]         |

---

## Week 7 — Evaluate: Results Checkpoint

**Deliverable during Week 8 Lab:** Preliminary evaluation results, metrics, development-set details, and 2–3 representative successes or failures. Include what you will improve next.

### Evaluation setup

- **Development data:** [Source, size, selection, annotations, link or access instructions]
- **Metrics and rationale:** [Why these metrics reflect the intended use]
- **Comparison:** [Baseline versus current approach, if different]

| System / version | Metric 1 | Metric 2 | Notes   |
| ---------------- | -------: | -------: | ------- |
| Baseline         |  [Value] |  [Value] | [Notes] |
| Current          |  [Value] |  [Value] | [Notes] |

### Successes, failures, and interpretation

| Example / category | Expected behavior | Actual behavior | Success or failure? | Likely reason |
| ------------------ | ----------------- | --------------- | ------------------- | ------------- |
| [Case]             | [Expected]        | [Observed]      | [Status]            | [Hypothesis]  |

- **What improved?** [Evidence]
- **What did not improve?** [Evidence]
- **Next experiment:** [Specific change and how you will measure it]

### Individual contributions during that week

| Member | Work completed  | Result (worked / partly / failed) | Evidence | Next step   |
| ------ | --------------- | --------------------------------- | -------- | ----------- |
| [Name] | [Specific work] | [Status]                          | [Link]   | [Next step] |

---

## Week 8 — Stress-Test: Reliability on Hidden Data

**Deliverable during Week 9 Lab:** Hidden-test results (as provided by the TA), analysis of at least three important failure cases, and a prioritized improvement plan.

- **Submitted system version / commit:** [Commit hash]
- **Hidden-test metrics and leaderboard result:** [Results, when available]
- **Difference from development results:** [Compare; do not assume the hidden set is identical]

| Failure case / category | What went wrong | Possible cause | Severity / user impact | Proposed fix |
| ----------------------- | --------------- | -------------- | ---------------------- | ------------ |
| [Case 1]                | [Observation]   | [Hypothesis]   | [Impact]               | [Fix]        |
| [Case 2]                | [Observation]   | [Hypothesis]   | [Impact]               | [Fix]        |
| [Case 3]                | [Observation]   | [Hypothesis]   | [Impact]               | [Fix]        |

### Changes attempted after stress-testing

| Change   | Worked? (Yes / Partly / No) | Evidence | Keep, revise, or revert? |
| -------- | --------------------------- | -------- | ------------------------ |
| [Change] | [Status]                    | [Result] | [Decision]               |

### Individual contributions during that week

| Member | Work completed  | Result (worked / partly / failed) | Evidence | Next step   |
| ------ | --------------- | --------------------------------- | -------- | ----------- |
| [Name] | [Specific work] | [Status]                          | [Link]   | [Next step] |

---

## Week 9 — Responsible AI, Peer Testing, and Final Delivery

**Deliverable during Week 10 Lab:** Document peer-testing feedback, Responsible AI concerns, changes made in response, and the final runnable system. Submit the final project report as instructed.

### Peer testing

- **Team evaluated:** [Team ID]
- **One thing that worked well:** [Evidence]
- **One meaningful failure:** [Reproduction steps or example]
- **One actionable recommendation:** [Specific fix]
- **Feedback received from another team:** [Summary]

### Responsible AI assessment

- **Potential harms or affected users:** [Who could be affected and how?]
- **Privacy and data handling:** [How sensitive information is protected]
- **Reliability and unsupported claims:** [How outputs are verified or uncertainty is communicated]
- **Fairness, accessibility, and usability:** [Relevant concerns and tests]
- **Human oversight:** [Where a human should review or correct outputs]
- **Remaining limitations:** [What should not be trusted or deployed yet]

### Final improvements and outcomes

| Feedback / issue | Change made | Worked? (Yes / Partly / No) | Evidence         | Remaining limitation |
| ---------------- | ----------- | --------------------------- | ---------------- | -------------------- |
| [Issue]          | [Change]    | [Status]                    | [Link or metric] | [Limitation]         |

### Individual contributions during that week

| Member | Final implementation / evaluation / documentation contribution | Result (worked / partly / failed) | Evidence | Reflection / remaining work |
| ------ | -------------------------------------------------------------- | --------------------------------- | -------- | --------------------------- |
| [Name] | [Specific work]                                                | [Status]                          | [Link]   | [Reflection]                |

### Final submission checklist

- [ ] Code and final system/prototype are in the repository
- [ ] Setup and run instructions are complete and tested
- [ ] Data sources, model versions, and dependencies are documented
- [ ] Evaluation results are reported and compared with the baseline
- [ ] Important failures and limitations are explained
- [ ] Hidden-test and peer-feedback findings are addressed
- [ ] Responsible AI considerations and human oversight are discussed
- [ ] Every team member's contributions are documented for every week

**Final commit / release:** [Commit hash or tag]

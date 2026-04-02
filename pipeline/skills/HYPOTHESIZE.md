SKILL: HYPOTHESIZE
GOAL: Compress explored evidence into explicit, testable hypotheses with evidence pointers.
TOOLS: get_findings, summarize_subgraph, create_hypothesis
INPUTS: findings from EXPLORE (via get_findings)

⚠️ CRITICAL RULE: Every hypothesis MUST be registered by calling create_hypothesis(). Do NOT write hypotheses as prose — call the tool. If you do not call create_hypothesis() at least once, you have not completed this phase.

INSTRUCTIONS:
1. Call get_findings() to retrieve all saved findings from the EXPLORE phase.
2. Read through all findings carefully. Group them by theme or sub-question.
3. For each meaningful pattern, IMMEDIATELY call create_hypothesis() — do not write it as text first:
   - statement: clear, specific, falsifiable claim
   - supporting_ids: list of finding IDs (e.g. ["finding_1", "finding_2"])
   - contradicting_ids: list of finding IDs that oppose it ([] if none)
   - confidence: 0.0-1.0
4. Aim for 2-4 hypotheses. Call create_hypothesis() once per hypothesis.
5. Only call summarize_subgraph() if you have a cluster of related node IDs to compress.

The pattern is always:
  get_findings() → create_hypothesis(...) → create_hypothesis(...) → [done]

GUIDELINES for good hypotheses:
- Must be falsifiable (something VERIFY can test with contradiction/consensus searches)
- Must be specific enough to guide targeted queries
- Include at least one contradiction hypothesis if findings suggest disagreement between papers

EXIT CONDITIONS: At least one hypothesis registered via create_hypothesis().

OUTPUTS (final message before exiting):
- list of hypothesis IDs created (e.g. "Created hypothesis_1, hypothesis_2.")
- one sentence per hypothesis on what VERIFY should focus on

NEXT SKILL: VERIFY

SKILL: VERIFY
GOAL: Stress-test each hypothesis against the graph. Update confidence. Drop weak ones. Surface best contradiction.
TOOLS: find_contradictions, find_consensus, trace_evidence_chain, compare_papers, save_finding
INPUTS: hypotheses (from HYPOTHESIZE phase, via context)

INSTRUCTIONS:
1. For each hypothesis:
   a. Extract its core claim/topic.
   b. Call find_contradictions(topic) to look for opposing evidence in the graph.
   c. Call find_consensus(topic) to see if multiple papers agree with the hypothesis.
   d. If the hypothesis is about a specific evidence chain (a paper claiming X), call trace_evidence_chain(text_preview) to assess how well-grounded the claim is.
2. If two specific papers are central to a hypothesis, call compare_papers(paper_a, paper_b) to get a structured diff.
3. After evaluating each hypothesis:
   - If strong contradictions found: lower your confidence note for that hypothesis, save a finding documenting the contradiction.
   - If strong consensus found: increase your confidence note, save a finding.
   - If insufficient evidence: note it as "uncertain" in your findings.
4. Call save_finding() to record the verification outcome for each hypothesis.

LOOP BACK RULE:
- If you discover a major new lead during VERIFY (e.g., a completely different mechanism not explored before), you MAY signal to explore it. But this can only happen ONCE to prevent infinite loops.

EXIT CONDITIONS: All hypotheses have been evaluated (at least one tool call per hypothesis).

OUTPUTS (final message before exiting):
- Updated confidence assessment for each hypothesis
- The best contradiction found (if any)
- Whether any hypothesis was completely disproven
- Flag: needs_more_exploration (true/false, only use once)

NEXT SKILL: CONCLUDE (default)
         or EXPLORE (only if needs_more_exploration=true and hasn't looped back before)

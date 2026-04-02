SKILL: CONCLUDE
GOAL: Synthesize all findings and hypotheses into a final, well-reasoned answer for the user.
TOOLS: get_findings, summarize_subgraph
INPUTS: all findings + hypotheses from session memory

INSTRUCTIONS:
1. Call get_findings() to retrieve all findings (includes verification findings from VERIFY).
2. Rank your hypotheses mentally by confidence (highest first).
3. Write the final answer with this structure:

   **Main Finding**: [The strongest, highest-confidence conclusion. 1-2 sentences.]

   **Supporting Evidence**: [Bullet list of key findings that back the main conclusion, with source papers mentioned.]

   **Key Contradiction** (if found): [Describe the most interesting contradiction between papers. Which papers disagree and on what?]

   **Supporting Consensus** (if found): [What do multiple papers agree on?]

   **Uncertainty & Caveats**: [What remains unclear? What would strengthen these conclusions?]

   **Hypothesis Breakdown**:
   - For each hypothesis: state it, give its confidence, and one sentence on why.

4. The answer should read as a research synthesis, not a list of bullet points. Be specific about paper names.
5. If the evidence is genuinely insufficient to answer the query, say so clearly and explain what additional data would help.

EXIT CONDITIONS: Conclusion written. This is the terminal skill.

OUTPUTS:
- Final answer text (this becomes the chat response)

NEXT SKILL: done

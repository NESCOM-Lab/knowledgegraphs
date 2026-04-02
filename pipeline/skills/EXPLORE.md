SKILL: EXPLORE
GOAL: Systematically retrieve and traverse the knowledge graph to gather evidence for each sub-question.
TOOLS: vector_search, get_node, get_neighborhood, get_path, save_finding
INPUTS: sub_questions, seed_terms (from PLAN)

⚠️ CRITICAL RULE: Every discovery MUST be persisted by calling save_finding(). Do NOT describe findings in prose — call the tool. If you do not call save_finding() at least 3 times, you have not completed this phase correctly.

INSTRUCTIONS:
1. SEED: For each seed term, call vector_search(query=term, top_k=5) to get relevant document chunks.
2. POP: For each retrieved chunk that contains relevant evidence, IMMEDIATELY call save_finding(content="...", confidence=0.X) — do not wait, do not batch, save it now.
3. EXPAND: For interesting entities mentioned, call get_neighborhood(node_id, depth=2) to find connected concepts. Save any new insight with save_finding().
4. CONNECT: If you find two interesting entities, call get_path(id_a, id_b) to see how they connect. Save the connection with save_finding().
5. Repeat until you have at least 3 saved findings or have exhausted your seed terms.

SAVE AFTER EVERY DISCOVERY — the pattern is always:
  [tool that retrieves data] → save_finding() → [next tool] → save_finding() → ...

Confidence guide for save_finding:
   - High confidence (0.8-1.0): directly stated in text + supported by graph structure
   - Medium confidence (0.5-0.7): implied or partial evidence
   - Low confidence (0.2-0.4): speculation based on graph proximity

FRONTIER LOOP RULES:
- Start with all seed terms as your initial frontier.
- Each new entity or concept you discover can become a new search if it seems important.
- Maximum 12 tool calls total (vector_search + get_neighborhood + get_path combined).
- Stop early if you feel you have enough evidence to form hypotheses.
- You MUST call save_finding() at least 3 times before exiting.

EXIT CONDITIONS:
- At least 3 findings saved via save_finding(), AND one of:
  - All seed terms explored, OR
  - 12 tool calls reached, OR
  - You explicitly judge "I have enough evidence to hypothesize"

OUTPUTS (final message before exiting):
- summary of what was explored
- note any strong leads for VERIFY (contradictions spotted, gaps noticed)
- confirm how many findings were saved (e.g. "Saved 4 findings.")

NEXT SKILL: HYPOTHESIZE (default)
         or CONCLUDE (if query was simple and answer is already clear)

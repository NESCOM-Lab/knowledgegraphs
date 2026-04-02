SKILL: PLAN
GOAL: Understand the graph schema and decompose the user's query into targeted sub-questions.
TOOLS: get_schema
INPUTS: user_query (string)

INSTRUCTIONS:
1. Call get_schema() to understand what node labels, relationship types, and papers are available in the graph.
2. Read the query carefully and identify the core research question.
3. Decompose it into 2-4 focused sub-questions that together will answer the original query. Each sub-question should target a different aspect:
   - What specific entities / concepts are involved?
   - Are there known contradictions or tensions in this area?
   - What is the consensus, if any?
   - How do different papers relate to each other on this topic?
4. For each sub-question, note which tools from EXPLORE/VERIFY would best answer it.
5. Identify 3-5 seed search terms to start the EXPLORE phase with.

EXIT CONDITIONS: When you have written out the sub-questions and seed terms. Do not call get_schema more than once.

OUTPUTS (write to your final message before exiting):
- sub_questions: list of 2-4 sub-questions
- seed_terms: list of 3-5 search terms for EXPLORE
- schema_summary: brief note on what papers and entity types are available

NEXT SKILL: EXPLORE

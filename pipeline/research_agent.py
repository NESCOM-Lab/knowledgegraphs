"""
ResearchAgent - Agentic orchestrator using a local Ollama LLM.

Implements the Plan → Explore → Hypothesize → Verify → Conclude skill loop.
Each skill phase reads its instruction file, filters to only the allowed tools,
and runs the LLM in a tool-calling loop until the skill's exit conditions are met.

Uses LangChain's ChatOllama with bind_tools() - same LLM stack as the rest of the project.
"""
import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama

# Add pipeline dir to path so sibling imports work from tools/
sys.path.insert(0, str(Path(__file__).parent))

from tools.graph_tools import query_graph, get_node, get_neighborhood, get_path, get_schema
from tools.semantic_tools import vector_search
from tools.reasoning_tools import find_contradictions, find_consensus, compare_papers, trace_evidence_chain
from tools.memory_tools import SessionMemory
from tools.helper_tools import summarize_subgraph

load_dotenv()

SKILLS_DIR = Path(__file__).parent / "skills"
SKILL_ORDER = ["PLAN", "EXPLORE", "HYPOTHESIZE", "VERIFY", "CONCLUDE"]

# ---------------------------------------------------------------------------
# Tool schemas — OpenAI / Ollama format (LangChain bind_tools accepts these)
# ---------------------------------------------------------------------------

ALL_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": (
                "Returns the current graph schema: node labels, relationship types, "
                "property keys, and list of ingested papers. Call this first to understand what's queryable."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_graph",
            "description": "Execute arbitrary Cypher query against Neo4j. Use when you know exactly what you want.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cypher": {"type": "string", "description": "Valid Cypher query string"}
                },
                "required": ["cypher"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_node",
            "description": "Fetch a specific node by its id or source property with all properties.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "The node's id or source value (e.g. paper filename or entity name)",
                    }
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_neighborhood",
            "description": "Expand N hops from a node. Returns surrounding subgraph. Use depth 1-3.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Center node id"},
                    "depth": {"type": "integer", "description": "Hops to expand (1-3)", "default": 2},
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_path",
            "description": "Find shortest path(s) between two nodes. Critical for 'how are these connected?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "id_a": {"type": "string", "description": "First node id"},
                    "id_b": {"type": "string", "description": "Second node id"},
                },
                "required": ["id_a", "id_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "Semantic embedding search over Document chunks in the graph DB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "top_k": {"type": "integer", "description": "Number of results (default 5)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_contradictions",
            "description": (
                "Search for pairs of document chunks or entity relationships that potentially "
                "contradict each other on a topic across different papers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_or_claim": {
                        "type": "string",
                        "description": "Topic or claim to search for contradictions on",
                    },
                    "top_k": {"type": "integer", "description": "Number of chunks to retrieve (default 8)", "default": 8},
                },
                "required": ["topic_or_claim"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_consensus",
            "description": "Find claims that multiple papers agree on for a given topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to find consensus on"},
                    "top_k": {"type": "integer", "description": "Number of chunks to retrieve (default 8)", "default": 8},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_papers",
            "description": "Structured diff between two papers: shared entities, unique entities, opposing claims.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_a": {"type": "string", "description": "Filename/source of first paper"},
                    "paper_b": {"type": "string", "description": "Filename/source of second paper"},
                },
                "required": ["paper_a", "paper_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trace_evidence_chain",
            "description": "Walk backwards from a text chunk to find what it ultimately rests on in the graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text_preview": {
                        "type": "string",
                        "description": "The text_preview field of the chunk (first ~100 chars of chunk text + '...')",
                    },
                    "depth": {"type": "integer", "description": "How many hops back to trace (default 3)", "default": 3},
                },
                "required": ["text_preview"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_finding",
            "description": "Save an intermediate finding to session memory. Call this after each meaningful discovery.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The finding text"},
                    "confidence": {
                        "type": "number",
                        "description": "Confidence 0.0 (speculation) to 1.0 (certain)",
                        "default": 0.7,
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_findings",
            "description": "Retrieve all findings saved in this session, sorted by confidence.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_hypothesis",
            "description": "Formally register a hypothesis with evidence pointers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string", "description": "The hypothesis statement"},
                    "supporting_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Finding IDs or paper names that support this hypothesis",
                    },
                    "contradicting_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Finding IDs or paper names that contradict this hypothesis",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Initial confidence 0.0-1.0",
                        "default": 0.5,
                    },
                },
                "required": ["statement", "supporting_ids", "contradicting_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_subgraph",
            "description": "Compress a set of graph nodes into a natural language summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of entity node ids to summarize",
                    }
                },
                "required": ["node_ids"],
            },
        },
    },
]

# Map tool names → dict for fast lookup
_SCHEMA_BY_NAME: dict[str, dict] = {t["function"]["name"]: t for t in ALL_TOOL_SCHEMAS}

# Which tools are allowed per skill
SKILL_TOOLS: dict[str, list[str]] = {
    "PLAN":       ["get_schema"],
    "EXPLORE":    ["vector_search", "get_node", "get_neighborhood", "get_path", "save_finding"],
    "HYPOTHESIZE":["get_findings", "summarize_subgraph", "create_hypothesis"],
    "VERIFY":     ["find_contradictions", "find_consensus", "trace_evidence_chain", "compare_papers", "save_finding", "get_findings"],
    "CONCLUDE":   ["get_findings", "summarize_subgraph"],
}


class ResearchAgent:
    """
    Agentic research system: Plan → Explore → Hypothesize → Verify → Conclude.
    Uses a local Ollama LLM (same model as the rest of the project) with tool calling.
    """

    def __init__(self, graph, embed_model, vector_retriever, on_update: Optional[Callable] = None):
        """
        args:
            graph: Neo4jGraph instance
            embed_model: OllamaEmbeddings instance
            vector_retriever: LangChain retriever from Neo4jVector
            on_update: optional callback(skill_name, message) for streaming status to UI
        """
        self.graph = graph
        self.embed_model = embed_model
        self.vector_retriever = vector_retriever
        self.on_update = on_update or (lambda skill, msg: None)
        self.memory = SessionMemory()

        model = os.getenv("LLM", "MedAIBase/MedGemma1.5:4b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        )

    def _load_skill(self, skill_name: str) -> str:
        skill_path = SKILLS_DIR / f"{skill_name}.md"
        if not skill_path.exists():
            return f"SKILL: {skill_name}\nNo instructions found."
        return skill_path.read_text()

    def _get_tools_for_skill(self, skill_name: str) -> list[dict]:
        allowed = set(SKILL_TOOLS.get(skill_name, []))
        return [_SCHEMA_BY_NAME[name] for name in allowed if name in _SCHEMA_BY_NAME]

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        try:
            result = self._dispatch(tool_name, tool_args)
        except Exception as e:
            result = {"error": f"Tool execution failed: {e}"}
        return json.dumps(result, default=str, indent=2)

    def _dispatch(self, tool_name: str, inp: dict):
        g = self.graph
        em = self.embed_model
        vr = self.vector_retriever
        mem = self.memory

        if tool_name == "get_schema":
            return get_schema(g)
        elif tool_name == "query_graph":
            return query_graph(g, inp["cypher"])
        elif tool_name == "get_node":
            return get_node(g, inp["node_id"])
        elif tool_name == "get_neighborhood":
            return get_neighborhood(g, inp["node_id"], inp.get("depth", 2))
        elif tool_name == "get_path":
            return get_path(g, inp["id_a"], inp["id_b"])
        elif tool_name == "vector_search":
            return vector_search(vr, em, inp["query"], top_k=inp.get("top_k", 5))
        elif tool_name == "find_contradictions":
            return find_contradictions(g, em, vr, inp["topic_or_claim"], inp.get("top_k", 8))
        elif tool_name == "find_consensus":
            return find_consensus(g, em, vr, inp["topic"], inp.get("top_k", 8))
        elif tool_name == "compare_papers":
            return compare_papers(g, inp["paper_a"], inp["paper_b"])
        elif tool_name == "trace_evidence_chain":
            return trace_evidence_chain(g, inp["text_preview"], inp.get("depth", 3))
        elif tool_name == "save_finding":
            return mem.save_finding(inp["content"], inp.get("confidence", 0.7))
        elif tool_name == "get_findings":
            return mem.get_findings()
        elif tool_name == "create_hypothesis":
            return mem.create_hypothesis(
                inp["statement"],
                inp.get("supporting_ids", []),
                inp.get("contradicting_ids", []),
                inp.get("confidence", 0.5),
            )
        elif tool_name == "summarize_subgraph":
            return summarize_subgraph(inp["node_ids"], g)  # no LLM client needed
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _run_skill_phase(
        self,
        skill_name: str,
        messages: list,
        max_iterations: int = 20,
    ) -> tuple[str, list]:
        """
        Run one skill phase.
        Returns (final_text_response, updated_messages).
        """
        skill_instructions = self._load_skill(skill_name)
        tools = self._get_tools_for_skill(skill_name)

        system_prompt = f"""You are an expert neuroscience research agent running the {skill_name} phase of a systematic literature analysis.

Follow these skill instructions exactly:

{skill_instructions}

CRITICAL RULES:
- Only use the tools provided for this phase
- Follow the INSTRUCTIONS step by step
- When you reach the EXIT CONDITIONS, write your OUTPUTS and stop calling tools
- Be concise in your reasoning; tools provide raw data, synthesize don't just repeat it
"""

        # Prepend the system message for this skill phase
        phase_messages = [SystemMessage(content=system_prompt)] + messages

        llm_with_tools = self.llm.bind_tools(tools)
        self.on_update(skill_name, f"Starting {skill_name} phase...")

        iterations = 0
        while iterations < max_iterations:
            iterations += 1

            response: AIMessage = llm_with_tools.invoke(phase_messages)
            phase_messages.append(response)
            messages.append(response)

            # No more tool calls → skill is done
            if not response.tool_calls:
                final_text = response.content if isinstance(response.content, str) else ""
                self.on_update(skill_name, f"Completed {skill_name} phase.")
                return final_text, messages

            # Execute each tool call and append results
            for tc in response.tool_calls:
                name = tc["name"]
                args = tc["args"]
                self.on_update(skill_name, f"→ {name}({json.dumps(args)[:80]})")
                result_str = self._execute_tool(name, args)
                tool_msg = ToolMessage(content=result_str, tool_call_id=tc["id"])
                phase_messages.append(tool_msg)
                messages.append(tool_msg)

        self.on_update(skill_name, f"WARNING: {skill_name} hit iteration cap ({max_iterations})")
        return "", messages

    def run(self, user_query: str) -> dict:
        """
        Run the full skill loop for a research query.

        returns dict with:
            - conclusion: final answer text
            - hypotheses: list of hypothesis dicts
            - findings:   list of finding dicts
            - skill_outputs: dict of per-skill text outputs
        """
        self.memory.clear()

        # Shared message history that grows across all skill phases
        messages: list = [HumanMessage(content=f"Research query: {user_query}")]

        skill_outputs: dict[str, str] = {}
        looped_back = False
        skill_sequence = list(SKILL_ORDER)

        i = 0
        while i < len(skill_sequence):
            skill_name = skill_sequence[i]

            output_text, messages = self._run_skill_phase(skill_name, messages)
            skill_outputs[skill_name] = output_text

            # Guard: if EXPLORE finished but saved no findings, inject a recovery
            # prompt and re-run EXPLORE once so the LLM actually calls save_finding().
            if skill_name == "EXPLORE" and not self.memory.get_findings():
                self.on_update(
                    "EXPLORE",
                    "WARNING: No findings were saved. Injecting recovery prompt.",
                )
                messages.append(
                    HumanMessage(
                        content=(
                            "[SYSTEM ALERT] The EXPLORE phase ended without calling save_finding() "
                            "even once. The findings list is empty and the pipeline cannot continue.\n\n"
                            "You MUST now re-run the EXPLORE phase. For each piece of evidence you "
                            "already found (it is visible in the conversation above), call "
                            "save_finding(content='...', confidence=0.X) immediately — one call per "
                            "distinct discovery. Do not write prose. Only call save_finding()."
                        )
                    )
                )
                recovery_text, messages = self._run_skill_phase("EXPLORE", messages, max_iterations=10)
                skill_outputs["EXPLORE"] = recovery_text

            # Guard: if HYPOTHESIZE finished but registered no hypotheses, inject a recovery
            # prompt and re-run HYPOTHESIZE once so the LLM actually calls create_hypothesis().
            if skill_name == "HYPOTHESIZE" and not self.memory.get_hypotheses():
                self.on_update(
                    "HYPOTHESIZE",
                    "WARNING: No hypotheses were registered. Injecting recovery prompt.",
                )
                messages.append(
                    HumanMessage(
                        content=(
                            "[SYSTEM ALERT] The HYPOTHESIZE phase ended without calling create_hypothesis() "
                            "even once. The hypotheses list is empty and VERIFY has nothing to test.\n\n"
                            "You MUST now register your hypotheses. The findings are available — call "
                            "get_findings() to retrieve them, then for each hypothesis call "
                            "create_hypothesis(statement='...', supporting_ids=[...], "
                            "contradicting_ids=[...], confidence=0.X) immediately. "
                            "Do not write hypotheses as prose. Only call create_hypothesis()."
                        )
                    )
                )
                recovery_text, messages = self._run_skill_phase("HYPOTHESIZE", messages, max_iterations=10)
                skill_outputs["HYPOTHESIZE"] = recovery_text

            # VERIFY can request one loop-back to EXPLORE for a new lead
            if (
                skill_name == "VERIFY"
                and not looped_back
                and "needs_more_exploration=true" in output_text.lower()
            ):
                looped_back = True
                self.on_update("VERIFY", "New lead found — looping back to EXPLORE once.")
                skill_sequence.insert(i + 1, "EXPLORE")

            # Tell the LLM the skill completed and hand off to next
            if output_text and i < len(skill_sequence) - 1:
                messages.append(
                    HumanMessage(
                        content=(
                            f"[{skill_name} phase complete. Summary: {output_text[:400]}]\n\n"
                            "Proceed to the next phase."
                        )
                    )
                )

            i += 1

        return {
            "conclusion": skill_outputs.get("CONCLUDE", "No conclusion generated."),
            "hypotheses": self.memory.get_hypotheses(),
            "findings": self.memory.get_findings(),
            "skill_outputs": skill_outputs,
        }

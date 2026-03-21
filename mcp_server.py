"""MCP stdio server exposing ClaudeJudge tools for Claude Code integration.

Run with: python mcp_server.py
Test with: echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python mcp_server.py
"""
import asyncio
import os

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from claude_judge import ClaudeJudge

load_dotenv()

app = Server("llm-evaluator")
_judge = ClaudeJudge()


def _result_to_text(result: dict) -> list[TextContent]:
    import json
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="evaluate_response",
            description="Evaluate an LLM response using Claude as judge. Dispatches to the correct rubric based on category.",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["Factual", "Reasoning", "Creative", "Coding"],
                        "description": "Evaluation category",
                    },
                    "prompt": {"type": "string", "description": "The original prompt given to the model"},
                    "response": {"type": "string", "description": "The model's response to evaluate"},
                    "reference_answer": {"type": "string", "description": "Optional reference/ground-truth answer"},
                },
                "required": ["category", "prompt", "response"],
            },
        ),
        Tool(
            name="evaluate_factual",
            description="Evaluate a factual Q&A response (factual_accuracy 60%, completeness 30%, conciseness 10%).",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "response": {"type": "string"},
                    "reference_answer": {"type": "string"},
                },
                "required": ["prompt", "response"],
            },
        ),
        Tool(
            name="evaluate_reasoning",
            description="Evaluate a reasoning/math response (correctness 40%, logical_validity 35%, step_clarity 25%).",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "response": {"type": "string"},
                    "reference_answer": {"type": "string"},
                },
                "required": ["prompt", "response"],
            },
        ),
        Tool(
            name="evaluate_creative",
            description="Evaluate a creative writing response (creativity 40%, coherence 30%, style 30%).",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "response": {"type": "string"},
                    "reference_answer": {"type": "string"},
                },
                "required": ["prompt", "response"],
            },
        ),
        Tool(
            name="evaluate_coding",
            description="Evaluate a code generation response (correctness 50%, syntax_validity 30%, best_practices 20%).",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "response": {"type": "string"},
                    "reference_answer": {"type": "string"},
                },
                "required": ["prompt", "response"],
            },
        ),
        Tool(
            name="batch_evaluate",
            description="Evaluate multiple responses concurrently.",
            inputSchema={
                "type": "object",
                "properties": {
                    "evaluations": {
                        "type": "array",
                        "description": "List of evaluation objects, each with category, prompt, response, and optional reference_answer.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {"type": "string"},
                                "prompt": {"type": "string"},
                                "response": {"type": "string"},
                                "reference_answer": {"type": "string"},
                            },
                            "required": ["category", "prompt", "response"],
                        },
                    }
                },
                "required": ["evaluations"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    loop = asyncio.get_event_loop()

    if name == "evaluate_response":
        result = await loop.run_in_executor(
            None,
            lambda: _judge.evaluate_response(
                arguments["category"],
                arguments["prompt"],
                arguments["response"],
                arguments.get("reference_answer", ""),
            ),
        )
        return _result_to_text(result)

    elif name == "evaluate_factual":
        result = await loop.run_in_executor(
            None,
            lambda: _judge.evaluate_factual(
                arguments["prompt"],
                arguments["response"],
                arguments.get("reference_answer", ""),
            ),
        )
        return _result_to_text(result)

    elif name == "evaluate_reasoning":
        result = await loop.run_in_executor(
            None,
            lambda: _judge.evaluate_reasoning(
                arguments["prompt"],
                arguments["response"],
                arguments.get("reference_answer", ""),
            ),
        )
        return _result_to_text(result)

    elif name == "evaluate_creative":
        result = await loop.run_in_executor(
            None,
            lambda: _judge.evaluate_creative(
                arguments["prompt"],
                arguments["response"],
                arguments.get("reference_answer", ""),
            ),
        )
        return _result_to_text(result)

    elif name == "evaluate_coding":
        result = await loop.run_in_executor(
            None,
            lambda: _judge.evaluate_coding(
                arguments["prompt"],
                arguments["response"],
                arguments.get("reference_answer", ""),
            ),
        )
        return _result_to_text(result)

    elif name == "batch_evaluate":
        evaluations = arguments["evaluations"]

        async def run_one(ev):
            return await loop.run_in_executor(
                None,
                lambda ev=ev: _judge.evaluate_response(
                    ev["category"],
                    ev["prompt"],
                    ev["response"],
                    ev.get("reference_answer", ""),
                ),
            )

        results = await asyncio.gather(*[run_one(ev) for ev in evaluations])
        import json
        return [TextContent(type="text", text=json.dumps(list(results), indent=2))]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())

"""Claude-as-Judge evaluation module.

Provides rubric-based scoring for LLM responses using Claude Haiku.
Shared by evaluate.py (direct import) and mcp_server.py (MCP tools).
"""
import json
import os
import re

RUBRICS = {
    "Factual": {
        "criteria": {
            "factual_accuracy": 0.60,
            "completeness": 0.30,
            "conciseness": 0.10,
        },
        "description": "Factual question-answering tasks requiring accurate, complete, concise answers.",
    },
    "Reasoning": {
        "criteria": {
            "correctness": 0.40,
            "logical_validity": 0.35,
            "step_clarity": 0.25,
        },
        "description": "Reasoning and math tasks requiring correct answers with clear logical steps.",
    },
    "Creative": {
        "criteria": {
            "creativity": 0.40,
            "coherence": 0.30,
            "style": 0.30,
        },
        "description": "Creative writing tasks rewarding originality, coherence, and style.",
    },
    "Coding": {
        "criteria": {
            "correctness": 0.50,
            "syntax_validity": 0.30,
            "best_practices": 0.20,
        },
        "description": "Code generation tasks requiring correct, syntactically valid, idiomatic code.",
    },
}

_FALLBACK = {"score": None, "explanation": "Judge unavailable", "rubric_breakdown": {}}


class ClaudeJudge:
    def __init__(self, api_key=None, model="claude-haiku-4-5-20251001"):
        self.model = model
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=key)
            except ImportError:
                self._client = None
        else:
            self._client = None

    def _build_system_prompt(self, category, rubric_hints=""):
        rubric = RUBRICS.get(category, RUBRICS["Factual"])
        criteria_lines = "\n".join(
            f"  - {name} (weight {weight:.0%})"
            for name, weight in rubric["criteria"].items()
        )
        criteria_keys = list(rubric["criteria"].keys())
        example_breakdown = {k: 0.85 for k in criteria_keys}

        hint_section = f"\nAdditional context: {rubric_hints}" if rubric_hints else ""

        return (
            f"You are an expert evaluator for {category} tasks. "
            f"{rubric['description']}{hint_section}\n\n"
            f"Score the response on these criteria:\n{criteria_lines}\n\n"
            "Return ONLY a JSON object — no markdown, no explanation outside JSON — with this exact schema:\n"
            '{"score": <float 0-1>, "explanation": "<1-2 sentence summary>", '
            f'"rubric_breakdown": {json.dumps(example_breakdown)}' + "}\n\n"
            "Compute score as the weighted average of rubric_breakdown values using the weights above."
        )

    def _call_api(self, system_prompt, user_message, max_tokens=512):
        if self._client is None:
            return None

        def attempt(extra_instruction=""):
            msg = user_message + extra_instruction
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": msg}],
            )
            raw = response.content[0].text.strip()
            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)

        try:
            return attempt()
        except (json.JSONDecodeError, Exception):
            try:
                return attempt("\n\nIMPORTANT: Respond with valid JSON only, no other text.")
            except Exception:
                return None

    def _evaluate(self, category, prompt, response, reference_answer="", rubric_hints=""):
        if self._client is None:
            return dict(_FALLBACK)

        rubric = RUBRICS.get(category, RUBRICS["Factual"])
        ref_section = f"\nReference answer: {reference_answer}" if reference_answer else ""
        user_message = (
            f"Prompt: {prompt}{ref_section}\n\nResponse to evaluate:\n{response}"
        )

        system_prompt = self._build_system_prompt(category, rubric_hints)
        result = self._call_api(system_prompt, user_message)

        if result is None:
            return dict(_FALLBACK)

        # Recompute weighted score from breakdown to ensure consistency
        breakdown = result.get("rubric_breakdown", {})
        weights = rubric["criteria"]
        if breakdown and weights:
            computed = sum(breakdown.get(k, 0) * w for k, w in weights.items())
            result["score"] = round(computed, 4)

        return {
            "score": result.get("score"),
            "explanation": result.get("explanation", ""),
            "rubric_breakdown": result.get("rubric_breakdown", {}),
        }

    def evaluate_factual(self, prompt, response, reference_answer="", rubric_hints=""):
        return self._evaluate("Factual", prompt, response, reference_answer, rubric_hints)

    def evaluate_reasoning(self, prompt, response, reference_answer="", rubric_hints=""):
        return self._evaluate("Reasoning", prompt, response, reference_answer, rubric_hints)

    def evaluate_creative(self, prompt, response, reference_answer="", rubric_hints=""):
        return self._evaluate("Creative", prompt, response, reference_answer, rubric_hints)

    def evaluate_coding(self, prompt, response, reference_answer="", rubric_hints=""):
        return self._evaluate("Coding", prompt, response, reference_answer, rubric_hints)

    def evaluate_response(self, category, prompt, response, reference_answer="", rubric_hints=""):
        """Dispatch to the appropriate category evaluator."""
        dispatch = {
            "Factual": self.evaluate_factual,
            "Reasoning": self.evaluate_reasoning,
            "Creative": self.evaluate_creative,
            "Coding": self.evaluate_coding,
        }
        fn = dispatch.get(category, self._evaluate)
        if fn is self._evaluate:
            return self._evaluate(category, prompt, response, reference_answer, rubric_hints)
        return fn(prompt, response, reference_answer, rubric_hints)

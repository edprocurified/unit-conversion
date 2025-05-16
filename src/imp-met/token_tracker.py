import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.usage import Usage

# Pricing per 1M tokens (in USD)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    # Default fallback pricing
    "default": {"input": 1.0, "output": 2.0},
}


class TokenTracker:
    """Tracks token usage and costs across OpenAI API calls."""

    def __init__(self) -> None:
        """Initialize a new token tracker."""
        # usage buckets: dynamic phases plus a 'total'
        self.usage: Dict[str, Dict[str, float]] = {
            "total": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "calls": 0}
        }
        self.detailed_logs: List[Dict[str, Any]] = []

    def track_usage(
        self,
        phase: str,
        usage: Usage,
        model_name: str,
        call_description: Optional[str] = None,
    ) -> None:
        """Now accepts a Usage dataclass and model_name."""
        # ensure phase bucket
        if phase not in self.usage:
            self.usage[phase] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
                "calls": 0,
            }

        inp = usage.input_tokens
        out = usage.output_tokens

        # pick pricing
        pricing = MODEL_PRICING.get(model_name)
        if pricing is None:
            prefix = max(
                (p for p in MODEL_PRICING if model_name.startswith(p)),
                default="default",
            )
            pricing = MODEL_PRICING[prefix]

        cost = (inp / 1_000_000) * pricing["input"] + (out / 1_000_000) * pricing[
            "output"
        ]

        # update buckets
        bucket = self.usage[phase]
        bucket["input_tokens"] += inp
        bucket["output_tokens"] += out
        bucket["cost"] += cost
        bucket["calls"] += 1

        total = self.usage["total"]
        total["input_tokens"] += inp
        total["output_tokens"] += out
        total["cost"] += cost
        total["calls"] += 1

        # log detail
        self.detailed_logs.append(
            {
                "phase": phase,
                "model": model_name,
                "description": call_description,
                "input_tokens": inp,
                "output_tokens": out,
                "cost": cost,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_summary(self) -> str:
        """
        Return a human-readable summary of token usage and costs.
        """
        lines: List[str] = ["\n===== TOKEN USAGE SUMMARY ====="]
        for phase, data in self.usage.items():
            lines.append(f"\n📊 {phase.upper()}: ")
            lines.append(f"   Calls:           {data['calls']}")
            lines.append(f"   Input Tokens:    {data['input_tokens']:,}")
            lines.append(f"   Output Tokens:   {data['output_tokens']:,}")
            lines.append(f"   Estimated Cost:  ${data['cost']:.4f}")
        return "\n".join(lines)

    def save_to_file(self, file_path: str) -> None:
        """
        Save the token usage data to a JSON file.
        """
        output = {
            "summary": self.usage,
            "detailed_logs": self.detailed_logs,
            "generated_at": datetime.now().isoformat(),
        }
        with open(file_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Token usage data saved to {file_path}")


# global instance
token_tracker = TokenTracker()

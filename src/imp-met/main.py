import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from agents import Agent, Runner, function_tool, trace
from agents.model_settings import ModelSettings
from agents.result import RunResult
from agents.usage import Usage
from dotenv import load_dotenv
from pydantic import BaseModel
from token_tracker import token_tracker  # To track usage
from tool import evaluate as raw_evaluate

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY must be set"


class LineItemInput(BaseModel):
    id: str
    description: str


class ConvertedDescription(BaseModel):
    id: str
    old_description: str
    new_description: str
    reasoning: str


class BatchResults(BaseModel):
    line_items: List[ConvertedDescription]


@function_tool
def evaluate(expr: str) -> float:
    return raw_evaluate(expr=expr)


agent = Agent(
    name="Imperial to Metric Converter",
    instructions="""
    You are an engineering wizard.

    You are responsible for converting line item descriptions from imperial to metric.

    Only convert the imperial units (inch, ft, etc.). Already existing metric units should not be converted nor changed. Keep them as is.
    Use the evaluation tool to compute the conversion using string mathematical expressions, only when you're making a conversion.
    Each input has an ID — carry it over into the output.
    For the reasoning, please mention the conversion factor used so it's clear.
    Convert inches to millimeters and feet to meters.
    """,
    tools=[evaluate],
    model="gpt-4.1-mini",
    output_type=BatchResults,
    model_settings=ModelSettings(include_usage=True),
)


def load_sample_input() -> List[Dict[str, Any]]:
    """Load sample input from JSON file in the same directory."""
    current_dir = Path(__file__).parent
    json_path = current_dir / "sample_input.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Sample input file not found at {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def batch_items(data: List[Dict[str, Any]], size: int) -> List[List[LineItemInput]]:
    items = [
        LineItemInput(
            id=str(item["quote_id"]),  # Using quote_id as the identifier
            description=item["Description"],
        )
        for item in data
    ]
    return [items[i : i + size] for i in range(0, len(items), size)]


async def run_batch(batch: List[LineItemInput]) -> Dict[str, ConvertedDescription]:
    payload_dict = {item.id: {"description": item.description} for item in batch}
    input_str = json.dumps(payload_dict, ensure_ascii=False)

    with trace("Imperial Converter"):
        result: RunResult = await Runner.run(agent, input_str)

    total_usage = Usage()
    for resp in result.raw_responses:
        total_usage.add(resp.usage)

    model_name: str = str(agent.model)
    token_tracker.track_usage(
        phase="batch",
        usage=total_usage,
        model_name=model_name,
        call_description=f"Batch IDs: {','.join(i.id for i in batch)}",
    )

    return {item.id: item for item in result.final_output.line_items}


async def main() -> None:
    BATCH_SIZE = 10
    sample_input: List[Dict[str, Any]] = load_sample_input()
    batches = batch_items(sample_input, BATCH_SIZE)

    all_results: List[Dict[str, ConvertedDescription]] = await asyncio.gather(
        *(run_batch(batch) for batch in batches)
    )

    merged_result: Dict[str, ConvertedDescription] = {
        k: v for batch in all_results for k, v in batch.items()
    }

    final_output: Dict[str, Dict[str, str]] = {
        id_: {
            "old_description": conv.old_description,
            "new_description": conv.new_description,
            "reasoning": conv.reasoning,
        }
        for id_, conv in merged_result.items()
    }

    # Save the complete output to a JSON file
    output_file = Path(__file__).parent / "conversion_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    # Save usage statistics
    usage_file = Path(__file__).parent / "usage_stats.json"
    usage_stats = {
        "summary": token_tracker.get_summary(),
        "detailed_logs": token_tracker.detailed_logs,
    }
    with open(usage_file, "w", encoding="utf-8") as f:
        json.dump(usage_stats, f, indent=2)

    # Still print to console for immediate feedback
    print(json.dumps(final_output, indent=2, ensure_ascii=False))
    print(token_tracker.get_summary())
    print("\n===== DETAILED PER-BATCH USAGE =====")
    for log in token_tracker.detailed_logs:
        desc = log["description"]
        inp = log["input_tokens"]
        out = log["output_tokens"]
        cost = log["cost"]
        print(f"{desc}: {inp} in / {out} out → ${cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())

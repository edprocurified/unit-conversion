import asyncio
import json
import os
from typing import Dict, List

from agents import Agent, Runner, function_tool, trace
from agents.model_settings import ModelSettings
from agents.result import RunResult
from agents.usage import Usage
from dotenv import load_dotenv
from pydantic import BaseModel

from token_tracker import token_tracker
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
    Only convert the imperial units. Already existing metric units should not be converted nor changed. Keep them as is.
    Use the evaluation tool to compute the conversion using string mathematical expressions.
    Each input has an ID — carry it over into the output.
    For the reasoning, please mention the conversion factor used so it's clear.
    """,
    tools=[evaluate],
    model="gpt-4.1-mini",
    output_type=BatchResults,
    model_settings=ModelSettings(include_usage=True),
)

sample_input: Dict[str, str] = {
    "Q0001": '3/4"X6 MTR UPVC PR PIPE CLS E - EFFAST',
    "Q0002": '1"X6 MTR UPVC PR PIPE CLS E - EFFAST',
    "Q0003": '1 1/4"X6 MTR UPVC PR PIPE CLS E - EFFAST',
    "Q0004": '1 1/2"X6 MTR UPVC PR PIPE CLS E - EFFAST',
    "Q0005": '2"X6 MTR UPVC PR PIPE CLS E EFFAST',
    "Q0006": '3"X6 MTR UPVC PR PIPE CLS E EFFAST',
    "Q0007": '4"X6 MTR UPVC PR PIPE CLS E EFFAST',
    "Q0008": '6"X6 MTR UPVC PR PIPE CLS E EFFAST',
    "Q0009": '8"X6 MTR UPVC PR PIPE CLS E EFFAST',
    "Q0010": 'PVC REDUCING BUSH 3/4"x1/2"',
    "Q0011": 'PVC REDUCING BUSH 1"X3/4"',
    "Q0012": "PV REDUCING BUSH 30mmx25mm",
}


def batch_items(data: Dict[str, str], size: int) -> List[List[LineItemInput]]:
    items = [LineItemInput(id=k, description=v) for k, v in data.items()]
    return [items[i : i + size] for i in range(0, len(items), size)]


async def run_batch(batch: List[LineItemInput]) -> Dict[str, ConvertedDescription]:
    payload_dict = {item.id: {"description": item.description} for item in batch}
    input_str = json.dumps(payload_dict)

    with trace("Imperial Converter"):
        result: RunResult = await Runner.run(agent, input_str)

    total_usage = Usage()
    for resp in result.raw_responses:
        total_usage.add(resp.usage)

    model_name: str = str(agent.model)
    print(model_name)
    token_tracker.track_usage(
        phase="batch",
        usage=total_usage,
        model_name=model_name,
        call_description=f"Batch IDs: {','.join(i.id for i in batch)}",
    )

    return {item.id: item for item in result.final_output.line_items}


async def main() -> None:
    BATCH_SIZE = 10
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

    print(json.dumps(final_output, indent=2))

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

import asyncio
import json
import os
from typing import Dict, List

from agents import Agent, Runner, function_tool, trace
from dotenv import load_dotenv
from pydantic import BaseModel

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
    Each input has an ID—carry it over into the output.
    """,
    tools=[evaluate],
    model="gpt-4.1-mini",
    output_type=BatchResults,
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
}


def batch_items(data: Dict[str, str], size: int) -> List[List[LineItemInput]]:
    items = [LineItemInput(id=k, description=v) for k, v in data.items()]
    return [items[i : i + size] for i in range(0, len(items), size)]


async def run_batch(batch: List[LineItemInput]) -> Dict[str, ConvertedDescription]:
    payload_dict = {item.id: {"description": item.description} for item in batch}
    input = json.dumps(payload_dict)
    with trace("Imperial Converter"):
        result = await Runner.run(agent, input)
        print(result)
        return {item.id: item for item in result.final_output.line_items}


async def main() -> None:
    BATCH_SIZE = 10
    batches = batch_items(sample_input, BATCH_SIZE)
    print(batches)

    all_results: List[Dict[str, ConvertedDescription]] = await asyncio.gather(
        *(run_batch(batch) for batch in batches)
    )

    merged_result: Dict[str, ConvertedDescription] = {
        k: v for batch in all_results for k, v in batch.items()
    }

    for key, converted in merged_result.items():
        print(f"{key}: {converted.new_description} ({converted.reasoning})")

    final_output: Dict[str, Dict[str, str]] = {
        id_: {
            "old_description": conv.old_description,
            "new_description": conv.new_description,
            "reasoning": conv.reasoning,
        }
        for id_, conv in merged_result.items()
    }

    print(json.dumps(final_output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

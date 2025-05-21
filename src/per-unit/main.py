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
    quantity: int | float
    uom: str = "Each"
    unit_rate: int | float


class ConvertedLineItem(BaseModel):
    id: str
    old_description: str
    new_description: str
    old_quantity: int | float
    new_quantity: int | float
    old_uom: str
    new_uom: str
    old_unit_rate: int | float
    new_unit_rate: int | float
    reasoning: str


class BatchResults(BaseModel):
    line_items: List[ConvertedLineItem]


@function_tool
def evaluate(expr: str) -> float:
    return raw_evaluate(expr=expr)


agent = Agent(
    name="Per-Unit Converter",
    instructions="""
    You are an engineering wizard.
    The user is requesting to convert line items to a per unit version. This will effectively impact:
    1. The item's description.
    2. The item's quantity.
    3. The item's unit price/rate.
    4. The item's unit of measure.

    For calculations, use the evaluation tool to compute a conversion/operation using string mathematical expressions.
    Each input has an ID - carry it over into the output.
    For the reasoning, please mention the operations performed and why.
    This per-unit conversion is typically generally done for lengths (like pipes) when an item is reported for example being 5 meters.
    If it is already per unit, do not do anything. Report the item as is.

    Example input:
        {{
        "quote_rows": [
            {{
                "quote_row_uuid": "Q0001",
                "description": "1\" x 5 MTR UPVC Pipe",
                "quantity": 1,
                "uom": "Ea",
                "unit_rate": 60
            }},
            ...
        ]
        }}

    Example Output:
        {{
        "quote_rows": [
            {{
                "quote_row_uuid": "Q0001",
                "description": "1\" x 1 MTR UPVC Pipe", # from 5 MTR -> 1 MTR ( = PER UNIT) -> Factor = 5
                "quantity": 5, # Multiply by factor = 1*5=5
                "uom": "M", # From each to meter
                "unit_rate": 12 # Divide by factor = 60/5 = 12
            }},
            ...
        ]
        }}

    This is how it is done:
    1. Update the description accordingly. You will get a factor. In our case, 5.
    2. Multiply quantity by the factor.
    3. Update the unit of measure to reflect the unit.
    4. Update the unit prace by dividing by the factor.
    5. Verification: OLD QUANTITY X OLD UNIT RATE = NEW QUANTITY X NEW UNIT RATE.
    6. Make the reasoning human friendly to be displayed on the frontend. Example: Converted from 5 meter per pipe to 1 meter per pipe. Quantity multiplied by 5 (1 x 5 = 5). Unit rate divided by 5 (60 / 5 = 12).
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
            id=str(item["quote_id"]),
            description=str(item["Description"]),
            quantity=float(item["Qty"]),
            unit_rate=float(item["Rate"]),
            # Using default "Each" for uom since it's not in the input
        )
        for item in data
    ]
    return [items[i : i + size] for i in range(0, len(items), size)]


async def run_batch(batch: List[LineItemInput]) -> Dict[str, ConvertedLineItem]:
    payload_dict = {
        item.id: {
            "description": item.description,
            "quantity": item.quantity,
            "uom": item.uom,
            "unit_rate": item.unit_rate,
        }
        for item in batch
    }
    input_str = json.dumps(payload_dict)

    with trace("Per-Unit Converter"):
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
    sample_input = load_sample_input()
    batches = batch_items(sample_input, BATCH_SIZE)

    all_results: List[Dict[str, ConvertedLineItem]] = await asyncio.gather(
        *(run_batch(batch) for batch in batches)
    )

    merged_result: Dict[str, ConvertedLineItem] = {
        k: v for batch in all_results for k, v in batch.items()
    }

    final_output: Dict[str, Dict[str, str | int | float]] = {
        id_: {
            "old_description": conv.old_description,
            "new_description": conv.new_description,
            "old_quantity": conv.old_quantity,
            "new_quantity": conv.new_quantity,
            "old_uom": conv.old_uom,
            "new_uom": conv.new_uom,
            "old_unit_rate": conv.old_unit_rate,
            "new_unit_rate": conv.new_unit_rate,
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

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
from token_tracker import token_tracker  # To track usage
from tool import evaluate as raw_evaluate

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY must be set"


class LineItemInput(BaseModel):
    id: str
    description: str
    quantity: int | float
    uom: str
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
    If it is already per unit, do not do anything. Report the item as is.

    Example input:
        {{
        "quote_rows": [
            {{
                "quote_row_uuid": "Q0001",
                "description": "1\" x 6 MTR UPVC Pipe",
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
                "description": "1\" x 1 MTR UPVC Pipe", # from 6 MTR -> 1 MTR ( = PER UNIT) -> Factor = 6
                "quantity": 6, # Multiply by factor = 1*6=6
                "uom": "M", # From each to meter
                "unit_rate": 60 # Divide by factor = 60/6 = 10
            }},
            ...
        ]
        }}

    This is how it is done:
    1. Update the description accordingly. You will get a factor. In our case, 6
    2. Multiply quantity by the factor.
    3. Update the unit of measure to reflect the unit.
    4. Update the unit prace by dividing by the factor.
    5. Verification: OLD QUANTITY X OLD UNIT RATE = NEW QUANTITY X NEW UNIT RATE.
    6. Make the reasoning human friendly to be displayed on the frontend. Example: Converted from 6 meter per pipe to 1 meter per pipe. Quantity multiplied by 6 (1 x 6 = 6). Unit rate divided by 6 (60 / 6 = 10).
    """,
    tools=[evaluate],
    model="gpt-4.1-mini",
    output_type=BatchResults,
    model_settings=ModelSettings(include_usage=True),
)

# More structured sample input with details for per-unit conversion
sample_input: Dict[str, Dict[str, str | int | float]] = {
    "Q0001": {
        "description": '3/4"X6 MTR UPVC PR PIPE CLS E - EFFAST',
        "quantity": 1,
        "uom": "Ea",
        "unit_rate": 11,
    },
    "Q0002": {
        "description": '1"X6 MTR UPVC PR PIPE CLS E - EFFAST',
        "quantity": 138,
        "uom": "Ea",
        "unit_rate": 15,
    },
    "Q0003": {
        "description": '1 1/4"X6 MTR UPVC PR PIPE CLS E - EFFAST',
        "quantity": 22,
        "uom": "Ea",
        "unit_rate": 25,
    },
    "Q0004": {
        "description": '1 1/2"X6 MTR UPVC PR PIPE CLS E - EFFAST',
        "quantity": 2,
        "uom": "Ea",
        "unit_rate": 31.5,
    },
    "Q0005": {
        "description": '2"X6 MTR UPVC PR PIPE CLS E EFFAST',
        "quantity": 65,
        "uom": "Ea",
        "unit_rate": 50,
    },
    "Q0006": {
        "description": '3"X6 MTR UPVC PR PIPE CLS E EFFAST',
        "quantity": 270,
        "uom": "Ea",
        "unit_rate": 95,
    },
    "Q0007": {
        "description": '4"X6 MTR UPVC PR PIPE CLS E EFFAST',
        "quantity": 225,
        "uom": "Ea",
        "unit_rate": 161.40,
    },
    "Q0008": {
        "description": '6"X6 MTR UPVC PR PIPE CLS E EFFAST',
        "quantity": 21,
        "uom": "Ea",
        "unit_rate": 349.10,
    },
    "Q0009": {
        "description": '8"X6 MTR UPVC PR PIPE CLS E EFFAST',
        "quantity": 31,
        "uom": "Ea",
        "unit_rate": 585.66,
    },
    "Q0010": {
        "description": 'PVC REDUCING BUSH 3/4"x1/2"',
        "quantity": 1,
        "uom": "Ea",
        "unit_rate": 1.63,
    },
    "Q0011": {
        "description": 'PVC REDUCING BUSH 1"X3/4"',
        "quantity": 50,
        "uom": "Ea",
        "unit_rate": 3.06,
    },
    "Q0012": {
        "description": "PV REDUCING BUSH 30mmx25mm",
        "quantity": 1,
        "uom": "Ea",
        "unit_rate": 25,
    },
}


def batch_items(
    data: Dict[str, Dict[str, str | int | float]], size: int
) -> List[List[LineItemInput]]:
    # Create LineItemInput objects with the structured input data
    items = [
        LineItemInput(
            id=k,
            description=str(v["description"]),
            quantity=float(v["quantity"]),
            uom=str(v["uom"]),
            unit_rate=float(v["unit_rate"]),
        )
        for k, v in data.items()
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

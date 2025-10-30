"""
agentic_controller.py

This is just a  reference implementation of a minimal agentic controller loop with planning.

What is in this file:
- Tool catalog with strict JSON Schemas (prevents tool/field hallucination)
- Argument validation + one-shot LLM repair when validation fails
- Budgets: max steps, tokens, and cost with simple accounting
- Loop detection to stop repeated ineffective actions
- Rolling history summarization to keep context small
- Planner that chooses the next action (tool or 'answer')
- Executor stub that simulates two tools (replace with your backends)
- Final synthesis step that composes the final answer

Notes on LLM Provider:
- This version uses OpenAI's SDK for planning, repair, summarization, and synthesis.
- To swap providers, replace client calls in the following functions:
  - repair_args_with_llm()
  - update_summary()
  - plan_next_action()
  - synthesize_answer()

Run:
  python agentic_controller.py
"""

# Imports & Setup ------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
from collections import deque
from httpx import request
from jsonschema import Draft202012Validator
from dotenv import load_dotenv
from openai import OpenAI, embeddings
import hashlib
import json
import os
import time
import random
import argparse
import requests
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tool Catalog ---------------------------------------------------------------------------------------------------------
# the tool catalog provides precise JSON Schemas for arguments. This helps to prevent the model from inventing fields or
# tools, and helps validation & auto-repair.

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    # Tool 1: a weather tool
    # STUDENT_COMPLETE --> You need to replace this with the correct one for the real weather tool call
    "weather.get_current": {
        "type": "object",
        "description": "Get the current weather",
        "properties": {
            "city": {"type": "string", "minLength": 1},
            "units": {
                "type": "string",
                "enum": ["metric", "imperial"],
                "default": "metric",
            },
        },
        "required": ["city"],
        "additionalProperties": False,
    },
    # Tool 2: knowledge-base search tool
    "kb.search": {
        "type": "object",
        "description": "search a knowledge base for information",
        "properties": {
            "query": {"type": "string", "minLength": 2},
            "k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    # STUDENT_COMPLETE --> You need to add a new tool schema for your custom tool
    # If LSU sports game tool doesn't work this is the first attempt at custom tool
    # Its a zip code locater, simple integration due to not having massive parsing requirements
    # also non api key endpoint. Will only be used if LSU game is not implemented by deadline
    "location.get_zip_info": {
        "type": "object",
        "description": "Get location information (city, state) for a US zip code.",
        "properties": {
            "zip_code": {
                "type": "string",
                "description": "The 5-digit US zip code, e.g., '70803'.",
                "pattern": "^\\d{5}$",
            }
        },
        "required": ["zip_code"],
        "additionalProperties": False,
    },
}

# Optional hints: rough latency/cost so planner can reason about budgets. I recommend replacing the default values
# with estimates that are accurate based on measurements.
TOOL_HINTS: Dict[str, Dict[str, Any]] = {
    "weather.get_current": {"avg_ms": 400, "avg_tokens": 50},
    "kb.search": {"avg_ms": 120, "avg_tokens": 30},
    "location.get_zip_info": {
        "avg_ms": 300,
        "avg_tokens": 40,
    },  # Added for custom Zip code tooling
}


# Controller State -----------------------------------------------------------------------------------------------------
@dataclass
class StepRecord:
    """Telemetry for each executed step (action)."""

    action: str  # tool name or 'answer'
    args: Dict[str, Any]  # arguments supplied
    ok: bool  # success flag
    latency_ms: int  # latency in milliseconds
    info: Dict[str, Any] = field(default_factory=dict)  # normalized payload


@dataclass
class ControllerState:
    """Mutable task state carried through the controller loop."""

    goal: str  # user task/goal
    history_summary: str = ""  # compact running summary (LLM-generated)
    tool_trace: List[StepRecord] = field(default_factory=list)
    tokens_used: int = 0  # simple token accounting
    cost_cents: float = 0.0  # simple cost accounting
    steps_taken: int = 0  # how many actions executed
    last_observation: str = ""  # short feedback string from last step
    done: bool = False  # termination flag


# Budgets & Accounting -------------------------------------------------------------------------------------------------
# Hard ceilings to avoid runaway cost
MAX_STEPS = 8
MAX_TOKENS = 20_000
MAX_COST_CENTS = 75.0


def within_budget(s: ControllerState) -> bool:
    """
    Check hard ceilings for steps, tokens, and cost.

    :param s: instance of ControllerState
    :return: True if still within budget, false if over-budget
    """
    return (
        s.steps_taken < MAX_STEPS
        and s.tokens_used < MAX_TOKENS
        and s.cost_cents < MAX_COST_CENTS
    )


def record_usage(s: ControllerState, usage) -> None:
    """
    Update token/cost counters using the response.usage object if available.
    This is a simplified accounting model for demonstration purposes.

    :param s: instance of ControllerState object
    :param usage: a response.usage object from OpenAI model response
    :return: None
    """
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    total = pt + ct
    s.tokens_used += total
    # gpt-5-mini is $0.25/million token
    s.cost_cents += total * 0.25 / 1e4


# Loop Detection -------------------------------------------------------------------------------------------------------
# Detect repeated (action, args) to avoid "stuck" ReAct oscillations.
LAST_ACTIONS = deque(maxlen=3)


def fingerprint_action(action: str, args: Dict[str, Any]) -> str:
    """
    Hash the tool call pair (action,args) to compare recent moves.

    :param action: the action the model selected
    :param args: the arguments the model selected for the action
    :return: A sha256 hash
    """
    blob = json.dumps({"a": action, "x": args}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def looks_stuck(action: str, args: Dict[str, Any]) -> bool:
    """
    Return True if the last N actions are identical (loop).

    :param action: the action the model selected
    :param args: the arguments the model selected for the action
    :return: True if this action is the same as the N actions, False otherwise
    """
    fp = fingerprint_action(action, args)
    LAST_ACTIONS.append(fp)
    return len(LAST_ACTIONS) == LAST_ACTIONS.maxlen and len(set(LAST_ACTIONS)) == 1


# Arg Validation & Repair ----------------------------------------------------------------------------------------------


def validate_args(tool_name: str, args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate args against the JSON Schema for the given tool. Return (ok, error_message). Error is concise for LLM
    repair prompt.

    :param tool_name: the name of the tool that the model selected
    :param args: the arguments the model selected for that tool
    :return: (True, None) if validates, (False, error message) if not
    """
    schema = TOOL_SCHEMAS[tool_name]
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(args), key=lambda e: e.path)
    if errors:
        e = errors[0]
        path = "/".join([str(p) for p in e.path]) or "(root)"
        return False, f"Invalid arguments at {path}: {e.message}"
    return True, None


def repair_args_with_llm(
    tool_name: str, bad_args: Dict[str, Any], error_msg: str
) -> Dict[str, Any]:
    """
    Ask the LLM to fix only the invalid parts to satisfy the JSON Schema.
    We enforce JSON-only output and re-validate after repair.

    :param tool_name: name of the selected tool
    :param bad_args: dictionary of bad arguments provided by the model
    :param error_msg: the error message provided by the validator
    :return: corrected (hopefully) arguments
    """
    schema = TOOL_SCHEMAS[tool_name]
    dev = (
        "You fix JSON arguments to match a JSON Schema. "
        "Return VALID JSON only—no prose, no code fences, no comments."
    )
    user = json.dumps(
        {
            "tool_name": tool_name,
            "schema": schema,
            "invalid_args": bad_args,
            "validator_error": error_msg,
        }
    )
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        response_format={"type": "json_object"},  # force JSON
        messages=[
            {"role": "developer", "content": dev},
            {"role": "user", "content": user},
        ],
    )
    return json.loads(resp.choices[0].message.content)


# History Summarization ------------------------------------------------------------------------------------------------


def update_summary(state: ControllerState, new_evidence: str) -> None:
    """
    Compress the prior summary + new evidence into a short rolling memory.
    Keeps context small but preserves key facts and decisions.

    :param state: instance of ControllerState
    :param new_evidence: this would be the response from the tool call
    :return:
    """
    sys = (
        "Compress facts and decisions into ≤120 tokens. Keep IDs and key numbers. Do not include anything that is "
        "unnecessary, only things that are strictly useful for the goal."
    )
    user = json.dumps(
        {"prior_summary": state.history_summary, "new_evidence": new_evidence}
    )
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "developer", "content": sys},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content.strip()
    state.history_summary = content
    if hasattr(resp, "usage"):
        record_usage(state, resp.usage)


# Planner --------------------------------------------------------------------------------------------------------------


def plan_next_action(state: ControllerState) -> Tuple[str, Dict[str, Any], str]:
    """
    Ask the LLM to pick ONE next action:
      - a known tool from TOOL_SCHEMAS with arguments, OR
      - the literal string 'answer' when it can synthesize the final answer.

    :param state: instance of ControllerState
    :return: (action, args, rationale)
    """
    # Pass the schema to the model. We also pass tool latency/token count for budget control and example to help the
    # model choose.
    tool_specs = []
    for name, schema in TOOL_SCHEMAS.items():
        spec = {
            "name": name,
            "schema": schema,  # including the full JSON Schema
            "budget_hint": {
                "avg_ms": TOOL_HINTS[name]["avg_ms"],
                "avg_tokens": TOOL_HINTS[name]["avg_tokens"],
            },
            # (Optional Few-shot prompting approach) keep examples tiny and schema-compliant
            # STUDENT_COMPLETE --> You may need to change this to be in line with your custom weather tool implementation
            "examples": {
                "weather.get_current": [
                    {"city": "Paris", "units": "metric"},
                    {"city": "New York"},  # units defaults via schema
                ],
                "kb.search": [{"query": "VPN policy for contractors", "k": 3}],
            }.get(name, []),
        }
        tool_specs.append(spec)

    # updating prompt due to model continually using gneeral knowledge instead of accessing tools
    dev = (
        "You are a planner. Your goal is to choose the next action. \n"
        "You MUST NOT answer from your own internal knowledge. You MUST use a tool to gather information first.\n"
        "1. Choose a tool from the `tool_catalog` if you need more information to answer the goal.\n"
        "2. Choose 'answer' ONLY IF the `history_summary` already contains all the information needed to answer the goal.\n"
        "3. If the goal is a question about general knowledge, you MUST use `kb.search` first.\n"
        "Do not call actions towards information already contained in the history summary provided below.\n"
        "When using a tool, produce arguments that VALIDATE against its JSON Schema.\n"
        "Allowed output format (JSON only):\n"
        '{"action":"<tool_name|answer>","args":{...}, "rationale":"<brief reason>"}'
    )

    user = json.dumps(
        {
            "goal": state.goal,
            "budget": {
                "steps_remaining": MAX_STEPS - state.steps_taken,
                "tokens_remaining": MAX_TOKENS - state.tokens_used,
                "cost_cents_remaining": round(MAX_COST_CENTS - state.cost_cents, 2),
            },
            "history_summary": state.history_summary,
            "tool_catalog": tool_specs,
            "last_observation": state.last_observation,
        }
    )

    resp = client.chat.completions.create(
        model="gpt-5-mini",
        response_format={
            "type": "json_object"
        },  # we could use strict mode if we wanted to
        messages=[
            {"role": "developer", "content": dev},
            {"role": "user", "content": user},
        ],
    )
    obj = json.loads(resp.choices[0].message.content)
    if hasattr(resp, "usage"):
        record_usage(state, resp.usage)
    return obj["action"], obj.get("args", {}), obj.get("rationale", "")


# Executor -------------------------------------------------------------------------------------------------------------


def execute_action(
    action: str, args: Dict[str, Any]
) -> Tuple[bool, str, Dict[str, Any], int]:
    """
    Execute the selected action with validation, repair, retries, and error handling.

    Replace the stubbed tool bodies with your real backends (APIs, DBs, etc.). Right now, they are just dummie entries.

    :param action: tool selected by the model
    :param args: arguments selected by the model
    :return: (ok, observation_text, normalized_payload, latency_ms)
    """

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    t0 = time.time()

    # 'answer' is a "virtual tool" signaling that we should synthesize the final answer.
    if action == "answer":
        obs = "Ready to synthesize final answer from working memory and evidence."
        return True, obs, {}, int((time.time() - t0) * 1000)

    # Guard: only call known tools from the catalog.
    if action not in TOOL_SCHEMAS:
        return False, f"Unknown tool: {action}", {}, int((time.time() - t0) * 1000)

    # 1) Validate arguments against schema.
    ok, msg = validate_args(action, args)
    if not ok:
        # 2) One-shot repair via LLM; re-validate.
        fixed = repair_args_with_llm(action, args, msg)
        ok2, msg2 = validate_args(action, fixed)
        if not ok2:
            return (
                False,
                f"Arg repair failed: {msg2}",
                {},
                int((time.time() - t0) * 1000),
            )
        args = fixed

    # 3) Execute the tool with basic retry on transient failures (e.g., timeouts).
    try:
        if (
            action == "weather.get_current"
        ):  # STUDENT_COMPLETE --> make this an actual weather API call
            try:
                city = args["city"]

                state = args.get("state")
                units = args.get("units", "metric")

                # using ua for this
                headers = {"User-Agent": "CSC4700-HW5 mgarc84@lsu.edu"}
                geo_url = "https://nominatim.openstreetmap.org/search"

                geo_params = {
                    "city": city,
                    "state": state,
                    "country": "USA",  # Restrict to USA for NWS
                    "format": "json",
                    "limit": 1,
                }
                geo_resp = requests.get(geo_url, params=geo_params, headers=headers)
                geo_resp.raise_for_status()  # raieses error if requests fails

                if not geo_resp.json():
                    return (
                        False,
                        f"Could not find coordinates for city: {city}",
                        {},
                        int((time.time() - t0) * 1000),
                    )

                geo_data = geo_resp.json()[0]
                latitude = geo_data["lat"]
                longitude = geo_data["lon"]

                # Get NWS grid from URL of lat and long

                points_url = f"https://api.weather.gov/points/{latitude},{longitude}"
                points_response = requests.get(points_url, headers=headers)
                points_response.raise_for_status()

                forecast_hourly_url = points_response.json()["properties"][
                    "forecastHourly"
                ]

                # Get the hourly forecast from the endpoint

                forecast_resp = requests.get(forecast_hourly_url, headers=headers)
                forecast_resp.raise_for_status()

                # current weather will be the first period as its the latest
                current_weather = forecast_resp.json()["properties"]["periods"][0]

                # Here we start processing the results to make cohesive info blocks
                temp_fahrenheit = current_weather["temperature"]
                temp_units = current_weather["temperatureUnit"]
                conditions = current_weather["shortForecast"]

                # Adding unit conversions
                # Didn't realize that NWS obviously wouldn't let me use worldwide cities
                # so this is redundant but I have it

                final_temp = temp_fahrenheit
                final_units_char = "F"

                if units == "metric":
                    if temp_units == "F":
                        final_temp = round((temp_fahrenheit - 32) * 5 / 9, 1)
                        final_units_char = "C"

                elif units == "imperial":
                    if temp_units == "C":
                        final_temp = round((temp_fahrenheit * 9 / 5) + 32, 1)
                        final_units_char = "F"

                payload = {
                    "city": city,
                    "units": units,
                    "temp": final_temp,
                    "conditions": conditions,
                    "temp_original": f"{temp_fahrenheit}°{temp_units}",
                }
                # Actual obs sent after final build of the payload
                obs = f"Current weather in {city}: {final_temp}°{final_units_char} ({conditions})"

                return True, obs, payload, int((time.time() - t0) * 1000)

            except requests.exceptions.RequestException as e:
                # Handle any API or network errors
                return False, f"API error: {e}", {}, int((time.time() - t0) * 1000)
            except (KeyError, IndexError) as e:
                # Handle unexpected JSON structure
                return (
                    False,
                    f"Error parsing API resp: {e}",
                    {},
                    int((time.time() - t0) * 1000),
                )

        elif (
            action == "kb.search"
        ):  # STUDENT_COMPLETE --> make this a vector search over a Chroma database
            try:
                # points chromadb to kb folder made from HW4
                client = chromadb.PersistentClient(path="./kb")

                # Must use same embedding fucntion as HW4 that created the original kb embeddings.
                # failure to do this prevents vector search from working
                openai_embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    model_name="text-embedding-3-small", api_key=openai_api_key
                )

                collection_name = "dataset_contexts"  # Matches name of the HW4 name

                collection = client.get_or_create_collection(
                    name=collection_name, embedding_function=openai_embedding_function
                )  # type ignore

                # gets arguments from the planner
                query = args["query"]
                k = args.get("k", 5)  # Default to 5 results

                # Query the collection
                results = collection.query(query_texts=[query], n_results=k)

                # Formats results for the observation
                snippets = []
                if results and results.get("documents") and results["documents"][0]:
                    snippets = results["documents"][0]

                if not snippets:
                    obs = "No relevant documents found for query."
                    return True, obs, {"results": []}, int((time.time() - t0) * 1000)

                obs = f"Retrieved {len(snippets)} snippets related to '{query}'"

                # return the list of document strings in the payload
                payload = {"results": snippets}

                return True, obs, payload, int((time.time() - t0) * 1000)

            except Exception as e:
                # Handle any ChromaDB / query errors
                return False, f"ChromaDB error: {e}", {}, int((time.time() - t0) * 1000)

        # This is my custom tool call to find the location of a zip code
        # There aren't many free api's with no tool access and although I am trying to complete
        # an LSU game lookup that would be able to return the scores of previous LSU games against certain
        # opponents however they are very dense to parse. You will onyl see this if I failed before deadline

        elif action == "location.get_zip_info":
            try:
                zip_code = args["zip_code"]
                api_url = f"https://api.zippopotam.us/us/{zip_code}"  # Creates api endpoint to hit

                # Zippo doesn't use UA but its good practice from what I learned
                headers = {"User-Agent": "CSC4700-HW5 (mgarc84@lsu.edu)"}

                resp = requests.get(api_url, headers=headers)
                resp.raise_for_status()  # Raise error for 4xx error codes

                data = resp.json()

                # Extract the first place from the places list
                place_name = data["places"][0]["place name"]
                state = data["places"][0]["state"]

                obs = f"Zip code {zip_code} is for {place_name}, {state}."
                payload = {
                    "zip_code": data["post code"],
                    "country": data["country"],
                    "city": place_name,
                    "state": state,
                    "longitude": data["places"][0]["longitude"],
                    "latitude": data["places"][0]["latitude"],
                }
                return True, obs, payload, int((time.time() - t0) * 1000)

            except requests.exceptions.RequestException as e:
                # This catches 404s if the zip code is not found
                return (
                    False,
                    f"API error: Zip code not found or invalid.",
                    {},
                    int((time.time() - t0) * 1000),
                )
            except (KeyError, IndexError) as e:
                # This catches errors if the JSON response is expected
                return (
                    False,
                    f"Error parsing API response: {e}",
                    {},
                    int((time.time() - t0) * 1000),
                )
        else:
            # no executor wired for this tool, had 1 or two times this errored out
            return (
                False,
                f"No executor bound for tool: {action}",
                {},
                int((time.time() - t0) * 1000),
            )

    except Exception as e:
        # Non-transient or unexpected error
        return (
            False,
            f"Tool error: {type(e).__name__}: {e}",
            {},
            int((time.time() - t0) * 1000),
        )


# Final Synthesis ------------------------------------------------------------------------------------------------------
def synthesize_answer(state: ControllerState) -> str:
    """
    Compose the final answer using the compact working summary accumulated in state.history_summary. The full raw trace
    can be logged elsewhere.

    :param state: instance of ControllerState
    :return: model's response
    """
    sys = (
        "Your goal is to produce a final answer to a goal (likely a question) using only evidence provided in the "
        "working summary."
    )
    user = (
        f"Goal: {state.goal}\n\n"
        f"Working summary:\n{state.history_summary}\n\n"
        f"Produce the final answer in ≤ 200 tokens."
    )
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "developer", "content": sys},
            {"role": "user", "content": user},
        ],
    )
    if hasattr(resp, "usage"):
        record_usage(state, resp.usage)
    return resp.choices[0].message.content.strip()


# Controller Loop ------------------------------------------------------------------------------------------------------


def run_agent(goal: str) -> str:
    """
    Main controller loop:
      while budgets remain and not done:
        1) Build context (we keep a rolling summary in state)
        2) Plan next action (tool or 'answer')
        3) Loop detection guard
        4) Execute with validation/repair/retry
        5) Update summary and telemetry
        6) If 'answer', synthesize final output and stop

    :param goal:
    :return:
    """
    state = ControllerState(goal=goal)

    while within_budget(state) and not state.done:
        # Ask the planner to choose the next action
        action, args, rationale = plan_next_action(state)
        print(
            f"Action selected: {action}\n\targuments: {args}\n\trationale: {rationale}"
        )

        # Prevent infinite ReAct loops by hashing last few actions
        if looks_stuck(action, args):
            print("\tdetected being stuck in loop...")
            state.last_observation = (
                "Loop detected: revise plan with a different next action."
            )
            # Do not increment steps or execute; let planner try again
            continue

        # Execute the chosen action (or 'answer' pseudo-tool)
        ok, obs, payload, ms = execute_action(action, args)
        print(f"\t\ttool payload: {payload}")

        # Record step telemetry
        state.steps_taken += 1
        state.tool_trace.append(
            StepRecord(action=action, args=args, ok=ok, latency_ms=ms, info=payload)
        )

        # Provide short observation back to planner for next turn
        state.last_observation = obs

        # Summarize new evidence into compact working memory
        update_summary(state, f"{action}({args}) -> {obs}")

        # If planner signaled 'answer', produce final answer and exit
        if action == "answer" and ok:
            final = synthesize_answer(state)
            state.done = True
            print("hello")
            return final

        # If a tool failed, we do not crash; the planner sees the observation
        # and can pivot on the next iteration. The loop will also stop on budgets.
    print(within_budget(state), state.done)
    # If we exit naturally, budgets are exhausted or we never reached 'answer'
    return "Stopped: budget exhausted or no progress."


# Demo -----------------------------------------------------------------------------------------------------------------
#

if __name__ == "__main__":
    # Argparse
    parser = argparse.ArgumentParser(description="Agentic Controller ...")
    parser.add_argument("query", type=str, help="User Query/Prompt/Question ...")
    args = parser.parse_args()

    # The 'goal' is the query from the command line
    goal = args.query

    if not goal:
        print('Usage: python agentic_controller.py "<your query here>"')
        exit(1)

    print(f'\n--- Running Agent for Goal: "{goal}" ---\n')
    answer = run_agent(goal)
    print("\n--- Final Answer ---\n")
    print(answer)

    # You could also print telemetry for inspection:
    # - steps taken, tokens used, cost, brief trace, etc.

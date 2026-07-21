import logging
import time
from dotenv import load_dotenv
from pprint import pprint

from atomic_agentic.agents import ReActKAgent
from atomic_agentic.llm import OpenAIEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)

llm_engine = OpenAIEngine(model="gpt-4o-mini")


def testDelayPrint() -> None:
    print("Called Print, now waiting 10 seconds...")
    time.sleep(10)


async_tester = ReActKAgent(
    name="Async_Delay_Tester",
    namespace="examples",
    description="Tests whether steps_per_round throttles achievable concurrency.",
    llm_engine=llm_engine,
    context_enabled=False,
    steps_per_round=5,
    tool_calls_limit=6,
)

# Register the callable (capture the fully-qualified tool id)
async_tester.register(
    testDelayPrint,
    name="DelayPrint",
    description="delay for 10 seconds.",
)


def run_once(*, sequentially: bool, steps_per_round: int) -> float:
    async_tester.steps_per_round = steps_per_round
    prompt = (
        f"Call 'DelayPrint' EXACTLY FIVE TIMES, but call them "
        f"{'SEQUENTIALLY' if sequentially else 'CONCURRENTLY'}. Then return None.\n"
    )
    start = time.time()
    result = async_tester.invoke({"prompt": prompt})
    elapsed = time.time() - start
    async_tester.clear_memory()
    pprint(result)
    return elapsed


if __name__ == "__main__":
    choice = input("Run steps sequentially? (y/n): ").strip().lower()
    sequentially = choice == "y"

    if sequentially:
        # steps_per_round=5 gives the model room to batch all five calls at
        # once; strict ordering is no longer structurally guaranteed (no
        # "await"), so this now tests whether the model, given only "call
        # them SEQUENTIALLY" plus the CONCURRENCY rule, correctly emits five
        # separate single-element subplans across five rounds instead.
        elapsed = run_once(sequentially=True, steps_per_round=5)
        print(f"\nSequential time taken: {elapsed:.2f} seconds")
    else:
        # Same concurrent task, run twice under two different round budgets.
        # steps_per_round=5 lets all 5 independent calls land in one round
        # (true concurrency, matching PlanAct's one-shot behavior).
        # steps_per_round=2 forces the same 5 independent calls across ~3
        # rounds, throttling wall time even though nothing is sequential.
        elapsed_k5 = run_once(sequentially=False, steps_per_round=5)
        print(f"\nConcurrent time taken (steps_per_round=5): {elapsed_k5:.2f} seconds")

        elapsed_k2 = run_once(sequentially=False, steps_per_round=3)
        print(f"Concurrent time taken (steps_per_round=3): {elapsed_k2:.2f} seconds")

        print(
            "\nSame independent task, same executor — the only difference is "
            "the per-round generation cap throttling achievable concurrency."
        )

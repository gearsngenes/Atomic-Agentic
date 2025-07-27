import sys
from pathlib import Path
# Set root to repo root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import PrePostAgent, ChainSequenceAgent
from modules.LLMEngines import *

# define a global llm engine to give to each of our agents
llm_engine = OpenAIEngine(model = "gpt-4o-mini")

# Prompts for each agent
CPP_2_ENG = """
You are a C++ code explainer. Given C++ code, respond ONLY with a clear,
step-by-step English description of its core functionality. No extra 
commentary.
"""
ENG_2_PY = """
You are a Python code adapter. Given an English description of a C++ 
program, write equivalent Python code that performs the same core 
functionality. Respond ONLY with valid Python code, no extra text or tags.
"""

# Preprocessor to print cpp to english output
def print_and_pass(val):
    print(f"\n=============\nTransformed Code:\n{val}\n================\n")
    return val

# helper to execute the function
def exec_python(code: str) -> str:
    try:
        result = exec(code, globals())
        # If 'output' is set in globals, return it
        return result if result else f"{code} returns 'None'"
    except Exception as e:
        return f"Erroneous Code:\n{code}\n\nError: {e}"

# Define and build our C++ to English translator
cpp_2_eng = PrePostAgent(
    name        = "C++_to_English",
    description = "Translates C++ code to an English representation",
    llm_engine  = llm_engine,
    role_prompt = CPP_2_ENG)
cpp_2_eng.add_poststep(print_and_pass)

# Define and build our English to Python executor
eng_2_py = PrePostAgent(
    name        = "English_to_Python",
    description = "Translates English code outlines/descriptions into python code and executes the generated code",
    llm_engine  = llm_engine,
    role_prompt = ENG_2_PY)
eng_2_py.add_poststep(print_and_pass)
eng_2_py.add_poststep(exec_python)


# Build our ChainSequence C++ to Python translator
cpp_2_py = ChainSequenceAgent("C++_to_Python-Translator")
cpp_2_py.add(cpp_2_eng)
cpp_2_py.add(eng_2_py)

if __name__ == "__main__":
    cpp_code = '''
#include <iostream>
using namespace std;

// Function to reverse an array
void reverseArray(int arr[], int n) {
    for (int i = 0; i < n / 2; ++i) {
        int temp = arr[i];
        arr[i] = arr[n - i - 1];
        arr[n - i - 1] = temp;
    }
}

// Recursive function to calculate Fibonacci
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Function to print an array
void printArray(int arr[], int n) {
    for (int i = 0; i < n; ++i) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    cout << "Original array: ";
    printArray(arr, 5);

    reverseArray(arr, 5);
    cout << "Reversed array: ";
    printArray(arr, 5);

    int n = 6;
    cout << "Fibonacci of " << n << " is " << fibonacci(n) << endl;
    return 0;
}
'''
    print("C++ code:\n", cpp_code)
    cpp_2_py.invoke(cpp_code)
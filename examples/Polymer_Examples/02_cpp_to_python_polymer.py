import sys
from pathlib import Path
# Set root to repo root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent, PolymerAgent
from modules.LLMNuclei import *

# define a global nucleus to give to each of our agents
nucleus = OpenAINucleus(model = "gpt-4o-mini")

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
    print(f"\n\nTransformed Code:\n{val}\n\n")
    return val

# helper to execute the function
def exec_python(code: str) -> str:
    try:
        result = exec(code, globals())
        # If 'output' is set in globals, return it
        return result if result else f"{code} returns 'None'"
    except Exception as e:
        return f"Erroneous Code:\n{code}\n\nError: {e}"

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

    # Create our polymer agents
    chain_cpp2eng = PolymerAgent(
        seed = Agent(name= "C++_to_English",
                nucleus= nucleus,
                role_prompt = CPP_2_ENG)
    )
    # register a method to print the english tranformed code
    chain_cpp2eng.register_tool(print_and_pass)
    
    # Register a python execution tool
    chain_eng2py = PolymerAgent(
        seed = Agent(name= "English_to_Python",
                    nucleus= nucleus,
                    role_prompt= ENG_2_PY)
    )
    # Register a method to print the python transformed code
    chain_eng2py.register_tool(print_and_pass)
    # Register a method to run the python code
    chain_eng2py.register_tool(exec_python)
    
    # Link the c++ to english translator to the english to python translator
    chain_cpp2eng.talks_to(chain_eng2py)
    
    # Invoke the chain method to see the full output
    chain_cpp2eng.invoke(cpp_code)
    
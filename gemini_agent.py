import os
from google import genai
from google.genai import types

def run_agent():
    client = genai.Client()

    print("🤖 Gemini Coding Agent is ready!")
    user_prompt = input("What would you like the agent to do? ")

    target_files = [
        "src/terrain_mapper/planner.py",
        "src/terrain_mapper/web_sim.py",
        "src/terrain_mapper/simulation.py"
    ]

    code_context = {}
    for path in target_files:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                code_context[path] = f.read()

    system_instruction = (
        "You are an expert software engineering coding agent working on a Python terrain mapping project. "
        "Analyze the user's request and the provided file contents. "
        "Provide the complete, updated code changes clearly so they can be written back to the project."
    )

    user_content = f"Task: {user_prompt}\n\nHere are the current project files:\n"
    for path, content in code_context.items():
        user_content += f"\n=== FILE: {path} ===\n{content}\n"

    print("⏳ Sending request to Gemini...")
    response = client.models.generate_content(
       model='gemini-3.1-pro-preview',
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2
        )
    )

    print("\n=== GEMINI AGENT RESPONSE ===")
    print(response.text)

if __name__ == "__main__":
    run_agent()
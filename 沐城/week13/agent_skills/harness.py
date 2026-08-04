#!/usr/bin/env python3
import os
from pathlib import Path
from dotenv import load_dotenv
from agent import Harness

def main():
    load_dotenv()
    
    # 初始化 Harness
    skills_dir = Path("./skills")
    harness = Harness(
        skills_dir=skills_dir,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        model=os.getenv("MODEL_ID", "gpt-4o")
    )
    
    print("🤖 Progressive Loading Harness")
    print(f"📁 Skills directory: {skills_dir.absolute()}")
    print(f"📚 Available skills: {harness.registry.list_skill_names()}")
    print("\n" + "="*50)
    print("Type 'quit' to exit, 'reset' to clear session\n")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
            if user_input.lower() == 'reset':
                harness.reset()
                print("🔄 Session reset.")
                continue
            if not user_input:
                continue
            
            print("\nAgent: ", end="", flush=True)
            response = harness.chat(user_input)
            print(response)
            
            # 显示当前加载的技能
            if harness.state['loaded_skills']:
                print(f"\n📦 Loaded: {list(harness.state['loaded_skills'].keys())}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
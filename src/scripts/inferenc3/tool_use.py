#!/usr/bin/env python3
"""
Minimal direct llm.generate() tool calling — WITH robust nested-JSON parser
"""

import json, uuid, sys, gc, atexit
from vllm import LLM, SamplingParams

# ─────────────────────────────────────────────────────────────
# 🎛️ Global LLM + cleanup
# ─────────────────────────────────────────────────────────────
llm = None

def init_llm():
    global llm
    if llm is None:
        llm = LLM(
            model="Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ",
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            enforce_eager=True,
            disable_log_stats=True,
        )
    return llm

def shutdown_llm():
    global llm
    if llm is not None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except: pass
        del llm
        llm = None
        gc.collect()

atexit.register(shutdown_llm)

# ─────────────────────────────────────────────────────────────
# 🛠️ Tools
# ─────────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get weather of a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }
]

# ─────────────────────────────────────────────────────────────
# 🔍 ROBUST parser: brace-counting for nested JSON + CoT handling
# ─────────────────────────────────────────────────────────────
def parse_tool_call(text: str, available_tools: list) -> dict | None:
    """Extract tool call JSON even with CoT preamble and nested objects"""
    tool_names = [t["name"] for t in available_tools]
    
    # Strategy 1: Try parsing entire text as JSON (if model outputs pure JSON)
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and obj.get("name") in tool_names and isinstance(obj.get("arguments"), dict):
            return {"name": obj["name"], "arguments": obj["arguments"], "id": f"call_{uuid.uuid4().hex[:8]}"}
    except: pass
    
    # Strategy 2: Brace-counting to extract balanced JSON from end (handles CoT + nested objects)
    # Search backwards for '{' and extract balanced JSON object
    for start in range(len(text) - 1, -1, -1):
        if text[start] == '{':
            count, end = 1, start + 1
            while end < len(text) and count > 0:
                if text[end] == '{':
                    count += 1
                elif text[end] == '}':
                    count -= 1
                end += 1
            if count == 0:  # Found balanced JSON object
                try:
                    obj = json.loads(text[start:end])
                    if isinstance(obj, dict) and obj.get("name") in tool_names and isinstance(obj.get("arguments"), dict):
                        return {"name": obj["name"], "arguments": obj["arguments"], "id": f"call_{uuid.uuid4().hex[:8]}"}
                except (json.JSONDecodeError, TypeError):
                    continue
    return None

# ─────────────────────────────────────────────────────────────
# 📝 Prompt builder
# ─────────────────────────────────────────────────────────────
def build_prompt(messages: list[dict], tools: list | None = None) -> str:
    parts = ["<|begin_of_sentence|>"]
    if tools:
        schema = "\n".join([f"• {t['name']}: {t['description']}" for t in tools])
        parts.append(f"<|System|>Available tools:\n{schema}\n\nTo use a tool, reply with ONLY JSON:\n{{\"name\": \"tool_name\", \"arguments\": {{...}}}}\n<|end_of_system|>")
    for msg in messages:
        role = "User" if msg["role"] in ("user", "tool") else "Assistant"
        content = msg.get("content", "")
        if msg["role"] == "tool":
            content = f"[Tool Result: {content}]"
        if content.strip():
            parts.append(f"<|{role}|>{content}")
    parts.append("<|Assistant|>")
    prompt = " ".join(parts)
    # Sanitize duplicate tags
    prompt = prompt.replace("<|Assistant|><|Assistant|>", "<|Assistant|>")
    prompt = prompt.replace("<|User|><|User|>", "<|User|>")
    return prompt

# ─────────────────────────────────────────────────────────────
# 🔄 Tool executor (mock — replace with real API)
# ─────────────────────────────────────────────────────────────
def execute_tool(name: str, args: dict) -> str:
    if name == "get_weather":
        location = args.get("location", "unknown")
        return f"24°C, Sunny in {location}"
    return "Unknown tool"

# ─────────────────────────────────────────────────────────────
# 💬 Chat with proper tool execution flow
# ─────────────────────────────────────────────────────────────
def chat_with_tools(user_input: str, tools: list | None = None, max_turns: int = 5) -> str:
    messages = [{"role": "user", "content": user_input}]
    llm = init_llm()
    
    for turn in range(max_turns):
        prompt = build_prompt(messages, tools)
        
        output = llm.generate(
            [prompt],
            SamplingParams(temperature=0.1, top_p=0.95, max_tokens=512, stop=["<|User|>", "<|end_of_system|>"])
        )[0].outputs[0].text.strip()
        
        print(f"\n[Turn {turn+1}] Raw model output:\n{output[:300]}{'...' if len(output)>300 else ''}\n", file=sys.stderr)
        
        # Parse tool call (robust version)
        tool_call = None
        if tools:
            tool_call = parse_tool_call(output, tools)
            print(f"[Turn {turn+1}] Parsed tool_call: {tool_call}\n", file=sys.stderr)
        
        if tool_call:
            print(f"🔧 Executing: {tool_call['name']}({tool_call['arguments']})", file=sys.stderr)
            tool_result = execute_tool(tool_call["name"], tool_call["arguments"])
            print(f"📦 Result: {tool_result}", file=sys.stderr)
            
            # Add to history and continue for final answer
            messages.append({"role": "assistant", "content": f"[Called {tool_call['name']}]"})
            messages.append({"role": "tool", "content": tool_result, "tool_call_id": tool_call["id"]})
            continue
        else:
            print(f"✅ No tool call → returning final answer", file=sys.stderr)
            return output
    
    return "⚠️ Max turns reached"

# ─────────────────────────────────────────────────────────────
# 🧪 Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        print("="*70)
        print("TEST 1: Weather query (tool call → execute → final answer)")
        print("="*70)
        result = chat_with_tools("How's the weather in Hangzhou, Zhejiang?", tools=TOOLS)
        print(f"\n🤖 FINAL ANSWER:\n{result}\n")
        
        print("="*70)
        print("TEST 2: Greeting (no tool)")
        print("="*70)
        result = chat_with_tools("Hello!", tools=TOOLS)
        print(f"\n🤖 FINAL ANSWER:\n{result}")
        
    finally:
        shutdown_llm()
        print("\n🔌 Clean shutdown")
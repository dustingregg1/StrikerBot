#!/usr/bin/env python3
"""Multi-AI Review System for StrikerBot - 6 AI reviewers in parallel."""

import os
import asyncio
from pathlib import Path

REVIEW_FILES = ['STRATEGY_REVIEW_REQUEST.md', 'MVP_REVIEW_REQUEST.md']

REVIEW_PROMPT = """Review this StrikerBot momentum/breakout trading bot code. Be HARSH and DATA-DRIVEN.

StrikerBot is a momentum/breakout crypto bot targeting 35-45% win rate with 3:1 reward/risk.
It uses EMA crossovers, Hurst Exponent regime detection, ATR-based stops, and Half-Kelly sizing.

Your focus: {focus}

Grade each component A-F. Identify fatal flaws. Suggest alternatives. Verdict: SHIP/FIX/KILL.

Documents:
{documents}
"""

FOCUS = {
    'Claude': 'Risk management edge cases, stop loss logic, position sizing safety',
    'ChatGPT': 'Strategy feasibility, EMA/ATR implementation correctness',
    'Gemini': 'Hurst Exponent math, Kelly criterion calculations, statistical rigor',
    'Mistral': 'Code quality, performance bottlenecks, async patterns',
    'DeepSeek': 'Alternative momentum strategies, contrarian analysis of approach',
    'Perplexity': 'Recent market research, historical backtest comparisons'
}


async def review_claude(docs: str) -> dict:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        msg = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=8000,
            messages=[{"role": "user", "content": REVIEW_PROMPT.format(
                focus=FOCUS['Claude'], documents=docs
            )}]
        )
        return {'ai': 'Claude', 'review': msg.content[0].text, 'ok': True}
    except Exception as e:
        return {'ai': 'Claude', 'review': str(e), 'ok': False}


async def review_chatgpt(docs: str) -> dict:
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        resp = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": REVIEW_PROMPT.format(
                focus=FOCUS['ChatGPT'], documents=docs
            )}],
            max_tokens=4000
        )
        return {'ai': 'ChatGPT', 'review': resp.choices[0].message.content, 'ok': True}
    except Exception as e:
        return {'ai': 'ChatGPT', 'review': str(e), 'ok': False}


async def review_gemini(docs: str) -> dict:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        model = genai.GenerativeModel('gemini-1.5-pro')
        resp = model.generate_content(REVIEW_PROMPT.format(
            focus=FOCUS['Gemini'], documents=docs
        ))
        return {'ai': 'Gemini', 'review': resp.text, 'ok': True}
    except Exception as e:
        return {'ai': 'Gemini', 'review': str(e), 'ok': False}


async def review_mistral(docs: str) -> dict:
    try:
        from mistralai.client import MistralClient
        client = MistralClient(api_key=os.environ['MISTRAL_API_KEY'])
        resp = client.chat(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": REVIEW_PROMPT.format(
                focus=FOCUS['Mistral'], documents=docs
            )}]
        )
        return {'ai': 'Mistral', 'review': resp.choices[0].message.content, 'ok': True}
    except Exception as e:
        return {'ai': 'Mistral', 'review': str(e), 'ok': False}


async def review_deepseek(docs: str) -> dict:
    try:
        import requests
        resp = requests.post(
            'https://api.deepseek.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {os.environ["DEEPSEEK_API_KEY"]}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'deepseek-chat',
                'messages': [{'role': 'user', 'content': REVIEW_PROMPT.format(
                    focus=FOCUS['DeepSeek'], documents=docs
                )}],
                'max_tokens': 4000
            }
        )
        return {'ai': 'DeepSeek', 'review': resp.json()['choices'][0]['message']['content'], 'ok': True}
    except Exception as e:
        return {'ai': 'DeepSeek', 'review': str(e), 'ok': False}


async def review_perplexity(docs: str) -> dict:
    try:
        import requests
        resp = requests.post(
            'https://api.perplexity.ai/chat/completions',
            headers={
                'Authorization': f'Bearer {os.environ["PERPLEXITY_API_KEY"]}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'llama-3.1-sonar-large-128k-online',
                'messages': [{'role': 'user', 'content': REVIEW_PROMPT.format(
                    focus=FOCUS['Perplexity'], documents=docs
                )}]
            }
        )
        return {'ai': 'Perplexity', 'review': resp.json()['choices'][0]['message']['content'], 'ok': True}
    except Exception as e:
        return {'ai': 'Perplexity', 'review': str(e), 'ok': False}


async def main() -> None:
    docs = "\n\n".join(
        [f"## {f}\n{Path(f).read_text()}" for f in REVIEW_FILES if Path(f).exists()]
    )
    if not docs.strip():
        # If no review request files, review the main source
        src_files = list(Path('.').glob('**/*.py'))
        docs = "\n\n".join(
            [f"## {f}\n{f.read_text()}" for f in src_files[:10]]
        )

    print("Launching 6 AI reviews in parallel...")

    reviews = await asyncio.gather(
        review_claude(docs),
        review_chatgpt(docs),
        review_gemini(docs),
        review_mistral(docs),
        review_deepseek(docs),
        review_perplexity(docs)
    )

    success = sum(1 for r in reviews if r['ok'])
    print(f"{success}/6 reviews completed")

    output = ["# Multi-AI Strategy Review Results\n\n"]
    output.append(f"**Successful Reviews:** {success}/6\n\n---\n\n")

    for r in reviews:
        output.append(f"## {r['ai']} Review\n\n")
        if r['ok']:
            output.append(f"{r['review']}\n\n")
        else:
            output.append(f"**Error:** {r['review']}\n\n")
        output.append("---\n\n")

    Path('AI_REVIEW_RESULTS.md').write_text("".join(output))
    print("Results written to AI_REVIEW_RESULTS.md")


if __name__ == '__main__':
    asyncio.run(main())

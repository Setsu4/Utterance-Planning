import openai
import re
import difflib
from dotenv import load_dotenv
import os
import sys
import select

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORGANIZATION")
)

# ユーザからの入力を受け取る
user_input = input("ニュース記事を入力してください：")

# プロンプト
system_prompt = """
あなたはユーザーにニュース記事の要約を生成した発話計画を使用して説明するシステムです。以下のフォーマットに従って発話計画を生成してください。
# 条件
- 生成する発話計画はすべて話し言葉であること。
- システムの発話3~5個を生成してください。
- 各システムの発話に対し、ユーザーからの質問とその回答3~10個を生成してください。
- 発話計画内で生成する質問は必ず該当するシステムの発話から生成すること。
- ニュース記事に載っていない情報は発話計画に含めないこと。
# 発話計画のフォーマット（質問数,発話は可変）
システム(発話1): システムの最初の発話
質問1. 質問内容1 回答: 回答内容1
質問2. 質問内容2 回答: 回答内容2
...
質問n. 質問内容n 回答: 回答内容n
システム(発話2): システムの二つ目の発話
質問1. 質問内容1 回答: 回答内容1
...
質問m. 質問内容m 回答: 回答内容m
システム(発話3): システムの三つ目の発話
質問1. 質問内容1 回答: 回答内容1
...
質問x. 質問内容x 回答: 回答内容x
# 発話計画のフォーマット例（※質問数は可変です）
システム(発話1):大谷翔平選手がロサンゼルス・ドジャースと10年7億ドルの契約を結んだって。
質問1. 契約期間は何年？ 回答: 10年だよ。
質問2. 契約金額はいくら？ 回答: 約7億ドルって言われてるよ。
質問3. どこの球団と契約したの？ 回答: ロサンゼルス・ドジャースだよ。
システム(発話2): その契約の中には後払い分もあるんだって。
質問1. 後払いってどういうこと？ 回答: 今もらわず、将来支払われるお金のことだよ。
質問2. なんで後払いにしたの？ 回答: 節税のためらしいよ。
質問3.契約って何？ 回答: ドジャースと10年7億ドルの契約だよ
"""

# 発話計画の生成
res = client.chat.completions.create(
    model="gpt-4.1-nano-2025-04-14",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
)

# 発話計画の解析
plan = res.choices[0].message.content
print("\n生成された発話計画:")
print(plan)

# 正規表現で発話計画を解析
system_pattern = re.compile(r'システム\(発話\d+\):(.*?)\n')
qa_pattern = re.compile(r'質問\d+.(.*?)\s+回答:(.*?)\n')
system_responses = system_pattern.findall(plan)
qa_pairs = qa_pattern.findall(plan)
raw_blocks = re.split(r'システム\(発話\d+\):', plan)[1:]

# 発話ごとのQAを格納するリスト
dialogue_plan = []

#ユーザ入力のタイムアウト処理
def input_with_timeout(prompt, timeout=20):
    print(prompt, end='', flush=True)
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        return sys.stdin.readline().rstrip('\n')
    else:
        return None

# 発話ブロックごとに処理
for block in raw_blocks:
    lines = block.strip().split("\n")
    if not lines:
        continue
    system_response = lines[0].strip()
    qa_pairs = []
    for line in lines[1:]:
        match = re.match(r'質問\d+\.(.*?)回答:(.*)', line)
        if match:
            question, answer = match.groups()
            qa_pairs.append({
                "question": question.strip(),
                "answer": answer.strip()
            })
    dialogue_plan.append({
        "system_response": system_response,
        "qa_pairs": qa_pairs
    })

print("\n--- dialogue_plan ---")
for i, d in enumerate(dialogue_plan):
    print(f"dialogue_plan[{i}]: {d}")

# 類似度計算
def calculate_similarity(user_question, plan_question):
    return difflib.SequenceMatcher(None, user_question, plan_question).ratio()

# マッチング質問を検索
def find_matching_question(user_question, qa_pairs):
    best_match = None
    best_similarity = 0
    for qa_pair in qa_pairs:
        similarity = calculate_similarity(user_question, qa_pair["question"])
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = qa_pair
    if best_match is not None:
        print(f"ユーザ質問: '{user_question}' | 最も類似度が高い発話計画質問: '{best_match['question']}' | 類似度: {best_similarity:.2f}")
    return best_match, best_similarity

# 相槌判定
def is_acknowledgement(user_responding):
    response = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=[
            {"role": "system", "content": "あなたは会話の相槌を判定する役割です。"},
            {"role": "user", "content": f"以下のテキストが会話の相槌かどうかを判定してください。\n\nユーザの入力: {user_responding}\n\nこの入力は会話の相槌ですか？（はい/いいえ）"}
        ],
        max_tokens=5,
        temperature=0
    )
    return "はい" in response.choices[0].message.content.strip()

# 対話関数
def user_interaction(dialogue_plan):
    for dialogue in dialogue_plan:
        print(f"\nシステム: {dialogue['system_response']}")
        while True:
            user_input = input_with_timeout("ユーザ（20秒以内に入力してください）: ", timeout=20)
            if user_input is None:
                print("（無反応のため次の発話に進みます）")
                break
            if is_acknowledgement(user_input):
                break
            matched_question, best_similarity = find_matching_question(user_input, dialogue["qa_pairs"])
            if best_similarity > 0.6:
                print(f"システム: {matched_question['answer']}")
            else:
                fallback_response = client.chat.completions.create(
                    model="gpt-4.1-nano-2025-04-14",
                    messages=[
                        {
                            "role": "system",
                            "content": "あなたはユーザーのニュースに関する質問に対し、正確で簡潔な回答を生成するシステムです。"
                        },
                        {
                            "role": "user",
                            "content": f"ニュース内容: {dialogue['system_response']}\n質問: {user_input}\n\nこの質問に対する正確で簡潔な回答を日本語で1文以内で答えてください。"
                        }
                    ],
                    max_tokens=100,
                    temperature=0.5,
                )
                print("システム:", fallback_response.choices[0].message.content.strip())

# 実行
user_interaction(dialogue_plan)
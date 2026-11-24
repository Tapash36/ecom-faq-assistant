import json
import os
import re
from collections import Counter
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent / ".env")
api_key = os.getenv("GEMINI_API_KEY")
SUPPORT_EMAIL = "teamX@gmail.com"
FAST_MODELS = ("gemini-3.5-flash-lite", "gemini-3.5-flash")

STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
    "how", "man", "new", "now", "old", "see", "two", "way", "who", "boy",
    "did", "its", "let", "put", "say", "she", "too", "use", "what", "when",
    "where", "which", "why", "with", "your", "have", "from", "this", "that",
    "will", "would", "could", "should", "about", "after", "before", "does",
    "into", "more", "other", "some", "such", "than", "then", "them", "these",
    "they", "very", "want", "need", "please", "help", "store", "item",
    "items", "product", "products", "order", "orders",
}


def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text, remove_stopwords=False):
    tokens = [token for token in normalize_text(text).split() if len(token) > 2]
    if remove_stopwords:
        return [token for token in tokens if token not in STOPWORDS]
    return tokens


@st.cache_resource
def get_client(key):
    return genai.Client(api_key=key)


@st.cache_data
def load_faq_knowledge():
    file_path = Path(__file__).resolve().parent / "data" / "train_expanded.json"
    if not file_path.exists():
        return []
    content = file_path.read_text(encoding="utf-8").strip()

    if not content:
        return []

    if content.startswith("["):
        data = json.loads(content)
    else:
        data = [json.loads(line) for line in content.splitlines() if line.strip()]

    cleaned = []
    seen = set()
    for item in data:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        answer = answer.replace("[email address]", SUPPORT_EMAIL)
        if question and answer:
            norm_q = normalize_text(question)
            if norm_q not in seen:
                seen.add(norm_q)
                q_tokens = Counter(tokenize(question, remove_stopwords=False))
                a_tokens = Counter(tokenize(answer, remove_stopwords=False))
                cleaned.append({
                    "question": question,
                    "answer": answer,
                    "norm_q": norm_q,
                    "norm_a": normalize_text(answer),
                    "q_tokens": q_tokens,
                    "a_tokens": a_tokens,
                })

    return cleaned


def retrieve_relevant_faqs(user_query, faq_entries, top_n=3):
    if not user_query or not faq_entries:
        return []

    norm_query = normalize_text(user_query)
    meaningful_tokens = set(tokenize(user_query, remove_stopwords=True))
    all_tokens = set(tokenize(user_query, remove_stopwords=False))
    query_tokens = meaningful_tokens or all_tokens

    if not query_tokens:
        return []

    ranked = []
    for item in faq_entries:
        norm_q = item["norm_q"]
        norm_a = item["norm_a"]
        q_tokens = item["q_tokens"]
        a_tokens = item["a_tokens"]

        score = 0
        matched_meaningful = 0
        for token in query_tokens:
            in_q = q_tokens.get(token, 0)
            in_a = a_tokens.get(token, 0)
            if in_q or in_a:
                if token in meaningful_tokens:
                    matched_meaningful += 1
                    score += in_q * 4 + in_a * 2
                else:
                    score += in_q * 1 + in_a * 1

        if norm_query and norm_query in norm_q:
            score += 12
        elif norm_query and norm_query in norm_a:
            score += 5

        if meaningful_tokens and matched_meaningful == 0 and norm_query not in norm_q:
            continue

        if score >= 3:
            ranked.append((score, {"question": item["question"], "answer": item["answer"]}))

    ranked.sort(key=lambda entry: entry[0], reverse=True)
    return [item for _, item in ranked[:top_n]]


def build_fast_prompt(relevant_faqs, history, current_query):
    if relevant_faqs:
        faq_block = "\n".join(
            f"- Q: {faq['question']}\n  A: {faq['answer']}"
            for faq in relevant_faqs
        )
    else:
        faq_block = "No matching store FAQ entry."

    recent_turns = history[-3:-1] if len(history) > 1 else []
    history_lines = [
        f"{'Customer' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent_turns
    ]
    history_block = f"\nRecent context:\n" + "\n".join(history_lines) if history_lines else ""

    system_instruction = (
        "You are TeamX (TaSaHiYo) AI Support Assistant. Give a fast, smooth, warm, and concise answer (2-4 sentences).\n"
        f"- If Store FAQs below answer the question, use those facts naturally.\n"
        f"- If there is no matching FAQ, give a helpful Gemini-generated answer and smoothly tell the user to contact **{SUPPORT_EMAIL}** for store-specific confirmation.\n"
        f"- For any extra, follow-up, or account-specific questions, always remind them to contact **{SUPPORT_EMAIL}**."
    )

    user_contents = (
        f"Store FAQs:\n{faq_block}{history_block}\n\n"
        f"Customer question: {current_query}"
    )
    return system_instruction, user_contents


def stream_gemini_response(client, system_instruction, user_contents):
    last_error = None
    for model_name in FAST_MODELS:
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=300,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            )
            return client.models.generate_content_stream(
                model=model_name,
                contents=user_contents,
                config=config,
            )
        except Exception as err:
            last_error = err
            try:
                # Fallback if thinking_level="minimal" is rejected on a specific model
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=300,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                )
                return client.models.generate_content_stream(
                    model=model_name,
                    contents=user_contents,
                    config=config,
                )
            except Exception as err2:
                last_error = err2
                continue
    raise last_error


def generate_fallback_reply(user_query, relevant_faqs):
    if relevant_faqs:
        best = relevant_faqs[0]["answer"]
        return (
            f"{best}\n\n"
            f"If you have any extra questions or need further help with your order, "
            f"please feel free to contact us at **{SUPPORT_EMAIL}**!"
        )
    return (
        "Thank you for reaching out! While I don't have a specific policy article for that exact question right now, "
        f"our support team would be happy to help you directly. Please contact us at **{SUPPORT_EMAIL}** for any extra or other questions!"
    )


st.set_page_config(
    page_title="TeamX Support Assistant",
    page_icon="🛍️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div.stButton > button {
        border-radius: 20px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        font-size: 0.88rem;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        border-color: #ff4b4b;
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("🛍️ TeamX Support")
    st.markdown(
        "Welcome to **TeamX (TaSaHiYo)** customer care! Ask anything about orders, shipping, returns, payments, or general inquiries."
    )
    if api_key:
        st.success("⚡ Gemini AI Connected (Fast Mode)", icon="✅")
    else:
        st.warning("⚠️ Gemini API Key Missing", icon="⚠️")

    st.divider()
    st.markdown(
        f"📧 **Need Human Support?**\n\nFor extra or account-specific questions, email us at:\n[**{SUPPORT_EMAIL}**](mailto:{SUPPORT_EMAIL})"
    )
    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 👋 Welcome to **TeamX Support**. I can help you with orders, shipping, returns, payments, "
                    f"and store policies—or answer any other questions you have. If you ever need direct assistance, you can also reach us at **{SUPPORT_EMAIL}**. How can I help you today?"
                ),
            }
        ]
        st.session_state.response_cache = {}
        st.rerun()

st.title("🛍️ TeamX (TaSaHiYo): AI Support Assistant")
st.caption(
    f"Powered by Gemini AI • Fast, smooth answers • Extra questions? Contact {SUPPORT_EMAIL}"
)

faq_entries = load_faq_knowledge()

quick_questions = [
    "📦 Track my order",
    "🔄 How do I return an item?",
    "💳 What payment methods do you accept?",
    "🚚 How long does shipping take?",
    "❌ Can I cancel my order?",
    "✉️ How can I contact customer support?",
]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 Welcome to **TeamX Support**. I can help you with orders, shipping, returns, payments, "
                f"and store policies—or answer any other questions you have. If you ever need direct assistance, you can also reach us at **{SUPPORT_EMAIL}**. How can I help you today?"
            ),
        }
    ]

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}

st.markdown("**✨ Quick topics:**")
cols = st.columns(3)
for idx, label in enumerate(quick_questions):
    clean_question = re.sub(r"^[^\w]+", "", label).strip()
    with cols[idx % 3]:
        if st.button(label, key=f"quick_{idx}", use_container_width=True):
            st.session_state.pending_query = clean_question

st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("faqs"):
            with st.expander("📚 Related Store FAQ References"):
                for idx, faq in enumerate(msg["faqs"], start=1):
                    st.markdown(f"**{idx}. {faq.get('question', '')}**")
                    st.caption(faq.get("answer", ""))

chat_input_query = st.chat_input("Ask a question about orders, policies, or anything else...")
user_query = st.session_state.pending_query or chat_input_query
if st.session_state.pending_query:
    st.session_state.pending_query = None

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    relevant_faqs = retrieve_relevant_faqs(user_query, faq_entries)
    norm_key = normalize_text(user_query)

    if not api_key:
        bot_reply = generate_fallback_reply(user_query, relevant_faqs)
        with st.chat_message("assistant"):
            st.warning("GEMINI_API_KEY is missing from `.env`. Showing fallback response:")
            st.markdown(bot_reply)
    elif norm_key in st.session_state.response_cache:
        bot_reply = st.session_state.response_cache[norm_key]
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
            if relevant_faqs:
                with st.expander("📚 Related Store FAQ References"):
                    for idx, faq in enumerate(relevant_faqs, start=1):
                        st.markdown(f"**{idx}. {faq.get('question', '')}**")
                        st.caption(faq.get("answer", ""))
    else:
        system_instruction, user_contents = build_fast_prompt(
            relevant_faqs, st.session_state.messages, user_query
        )

        try:
            client = get_client(api_key)
            answer_parts = []
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("⚡ *Generating answer...*")
                response_stream = stream_gemini_response(
                    client, system_instruction, user_contents
                )
                for chunk in response_stream:
                    chunk_text = getattr(chunk, "text", None)
                    if chunk_text:
                        answer_parts.append(chunk_text)
                        message_placeholder.markdown("".join(answer_parts) + " ▌")

                bot_reply = "".join(answer_parts).strip()
                if not bot_reply:
                    bot_reply = generate_fallback_reply(user_query, relevant_faqs)

                if SUPPORT_EMAIL.lower() not in bot_reply.lower():
                    bot_reply += (
                        f"\n\n💡 *For any extra questions, feel free to contact us at **{SUPPORT_EMAIL}**.*"
                    )

                message_placeholder.markdown(bot_reply)
                st.session_state.response_cache[norm_key] = bot_reply

                if relevant_faqs:
                    with st.expander("📚 Related Store FAQ References"):
                        for idx, faq in enumerate(relevant_faqs, start=1):
                            st.markdown(f"**{idx}. {faq.get('question', '')}**")
                            st.caption(faq.get("answer", ""))

        except Exception as error:
            bot_reply = generate_fallback_reply(user_query, relevant_faqs)
            with st.chat_message("assistant"):
                st.markdown(bot_reply)
                st.caption(f"*(Note: Live AI response unavailable right now: {error})*")

    st.session_state.messages.append(
        {"role": "assistant", "content": bot_reply, "faqs": relevant_faqs}
    )

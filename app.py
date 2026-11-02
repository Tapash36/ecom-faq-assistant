import json
import os
import re
from collections import Counter
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
SUPPORT_EMAIL = "support@store.com"


def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    return [token for token in normalize_text(text).split() if len(token) > 2]


@st.cache_resource
def get_client():
    return genai.Client(api_key=api_key)


@st.cache_data
def load_faq_knowledge():
    file_path = Path(__file__).resolve().parent / "data" / "train_expanded.json"
    content = file_path.read_text(encoding="utf-8").strip()

    if not content:
        return []

    if content.startswith("["):
        data = json.loads(content)
    else:
        data = [json.loads(line) for line in content.splitlines() if line.strip()]

    cleaned = []
    for item in data:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            cleaned.append({"question": question, "answer": answer})

    return cleaned


def retrieve_relevant_faqs(user_query, faq_entries, top_n=4):
    if not user_query or not faq_entries:
        return []

    query_tokens = set(tokenize(user_query))
    if not query_tokens:
        return []

    ranked = []
    for item in faq_entries:
        searchable_text = f"{item.get('question', '')} {item.get('answer', '')}"
        searchable_tokens = tokenize(searchable_text)
        token_counts = Counter(searchable_tokens)

        score = 0
        for token in query_tokens:
            score += token_counts.get(token, 0)

        if normalize_text(user_query) in normalize_text(item.get("question", "")):
            score += 8
        if normalize_text(user_query) in normalize_text(item.get("answer", "")):
            score += 3

        if score > 0:
            ranked.append((score, item))

    ranked.sort(key=lambda entry: entry[0], reverse=True)
    return [item for _, item in ranked[:top_n]]


def get_greeting_response(user_query):
    greetings = {
        "hello",
        "hello there",
        "hi",
        "hi there",
        "hey",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening",
    }
    if normalize_text(user_query) in greetings:
        return "Hello! I can help with orders, shipping, returns, payments, and store policies."
    return None


def find_direct_faq(user_query, faq_entries):
    query = normalize_text(user_query)
    query_words = set(query.split())
    if not query_words:
        return None

    for faq in faq_entries:
        question = normalize_text(faq.get("question", ""))
        if query == question or (
            len(query_words) >= 3 and query_words.issubset(set(question.split()))
        ):
            return faq
    return None


st.set_page_config(page_title="SmartDesk FAQ Assistant", page_icon="🛍️")
st.title("🛍️ TeamX(TaSaHiYo): Support Assistant")
st.caption("AI-powered customer FAQ assistant grounded on store policies and support articles.")

faq_entries = load_faq_knowledge()

quick_questions = [
    "Track my order",
    "How do I return an item?",
    "What payment methods do you accept?",
    "How long does shipping take?",
    "Can I cancel my order?",
]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I can help with orders, shipping, returns, payments, and store policies."}
    ]

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

st.write("Try a quick question:")
cols = st.columns(3)
for idx, question in enumerate(quick_questions):
    with cols[idx % 3]:
        if st.button(question, key=f"quick_{idx}", use_container_width=True):
            st.session_state.pending_query = question

chat_input_query = st.chat_input("Ask a question about store policies...")
user_query = st.session_state.pending_query or chat_input_query
if st.session_state.pending_query:
    st.session_state.pending_query = None

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    greeting_reply = get_greeting_response(user_query)
    direct_faq = find_direct_faq(user_query, faq_entries)
    relevant_faqs = [direct_faq] if direct_faq else retrieve_relevant_faqs(user_query, faq_entries)

    if greeting_reply:
        bot_reply = greeting_reply
        with st.chat_message("assistant"):
            st.write(bot_reply)
    elif direct_faq:
        bot_reply = direct_faq["answer"]
        with st.chat_message("assistant"):
            st.write(bot_reply)
    elif not api_key:
        bot_reply = "I can’t answer that right now because GEMINI_API_KEY is missing. Check your .env file."
        with st.chat_message("assistant"):
            st.error(bot_reply)
    else:
        faq_context = "\n\n".join(
            f"FAQ {idx + 1}: Q: {faq.get('question', '')}\nA: {faq.get('answer', '')}"
            for idx, faq in enumerate(relevant_faqs)
        ) or "No directly relevant FAQ entries were found for this question."

        system_prompt = (
            "You are an official customer support assistant for an online store. "
            "Answer the customer question using only the relevant FAQ entries below. "
            "If the customer asks something outside the FAQ, politely explain that the answer is not available "
            "in the knowledge base and recommend contacting human support at "
            f"{SUPPORT_EMAIL}. Do not invent facts, policies, discounts, or shipping promises.\n\n"
            f"RELEVANT FAQ ENTRIES:\n{faq_context}"
        )
        full_prompt = f"{system_prompt}\n\nCustomer Question: {user_query}\nSupport Answer:"

        try:
            client = get_client()
            response_stream = client.models.generate_content_stream(
                model="gemini-3.8-flash",
                contents=full_prompt,
            )

            answer_parts = []
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                for chunk in response_stream:
                    if getattr(chunk, "text", None):
                        answer_parts.append(chunk.text)
                        message_placeholder.write("".join(answer_parts))

                bot_reply = "".join(answer_parts).strip()
                if not bot_reply:
                    bot_reply = "I couldn’t generate an answer right now. Please try again."
                    st.error(bot_reply)
        except APIError as error:
            bot_reply = "I couldn’t get an answer just now. Please try again."
            with st.chat_message("assistant"):
                st.error(f"Gemini request failed: {error}")
                st.write(bot_reply)

    if relevant_faqs:
        with st.expander("Relevant FAQ references"):
            for idx, faq in enumerate(relevant_faqs, start=1):
                st.markdown(f"**{idx}.** {faq.get('question', '')}")
                st.write(faq.get('answer', ''))

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
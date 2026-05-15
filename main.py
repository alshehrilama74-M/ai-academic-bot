# -*- coding: utf-8 -*-
"""
AI Academic Advisor Bot  —  improved
=====================================
Requirements:
    pip install python-telegram-bot google-generativeai

Setup:
    1. Get a Telegram bot token from @BotFather
    2. Get a Gemini API key from https://aistudio.google.com/app/apikey
    3. Replace TELEGRAM_TOKEN and GEMINI_API_KEY below
    4. Run in Colab or terminal
"""

import logging
import os

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)
import google.generativeai as genai

# ─────────────────────────────────────────────
#  CONFIGURATION  –  Replace with your own keys
# ─────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "models/gemini-2.5-flash"

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  GEMINI CLIENT
#  max_output_tokens raised to 3000 to prevent cut-off responses
# ─────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL,
    generation_config=genai.types.GenerationConfig(
        max_output_tokens=2000,   # was 1000 — increased to avoid truncation
        temperature=0.7,
    ),
)

# ─────────────────────────────────────────────
#  CONVERSATION STATES
# ─────────────────────────────────────────────
# Onboarding states
CHOOSING_LANGUAGE = 0
CHOOSING_MAJOR    = 1
TYPING_MAJOR      = 2   # NEW: user types their major when they pick "Other"

# Feature flow states — each feature button leads to AWAITING_TOPIC
AWAITING_TOPIC = 3

# ─────────────────────────────────────────────
#  SUPPORTED MAJORS
# ─────────────────────────────────────────────
MAJORS = [
    "Computer Science",
    "Medicine",
    "Media",
    "Business",
    "Sports",
    "Other",
]

MAJOR_KEYBOARD = ReplyKeyboardMarkup(
    [MAJORS[i : i + 2] for i in range(0, len(MAJORS), 2)],
    one_time_keyboard=True,
    resize_keyboard=True,
)

# Language selection keyboard
LANGUAGE_KEYBOARD = ReplyKeyboardMarkup(
    [["English", "العربية"]],
    one_time_keyboard=True,
    resize_keyboard=True,
)

# Main feature menu — shown after onboarding is complete
MAIN_MENU_KEYBOARD = ReplyKeyboardMarkup(
    [
        ["Explain", "Study Plan"],
        ["Summary", "Quiz"],
        ["Resources", "Help"],
    ],
    resize_keyboard=True,
    one_time_keyboard=False,
)

# Maps the button label the user taps to an internal action key
FEATURE_LABEL_TO_ACTION = {
    "Explain":    "explain",
    "Study Plan": "plan",
    "Summary":    "summary",
    "Quiz":       "quiz",
    "Resources":  "resources",
}

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def get_lang(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Return 'ar' or 'en' based on the user's saved language choice."""
    return context.user_data.get("lang", "en")


def get_major(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.user_data.get("major", "General Studies")


def tr(context: ContextTypes.DEFAULT_TYPE, en: str, ar: str) -> str:
    """Return the correct string for the user's chosen language."""
    return ar if get_lang(context) == "ar" else en


def ask_gemini(system_prompt: str, user_message: str) -> str:
    """
    Send a combined system + user prompt to Gemini and return the reply.
    Uses the original google.generativeai GenerativeModel.
    """
    try:
        full_prompt = f"{system_prompt}\n\n{user_message}"
        response = gemini_model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        logger.error("Gemini error: %s", e)
        return (
            "Sorry, I could not reach the AI service right now. "
            "Please try again in a moment.\n\n"
            "عذرًا، لم أتمكن من الاتصال بالخدمة. يرجى المحاولة مرة أخرى."
        )


def build_system_prompt(context: ContextTypes.DEFAULT_TYPE, task: str) -> str:
    """
    Build the Gemini system prompt.
    Injects the student's major and enforces the chosen language.
    Keeps a clean, professional style with minimal emojis.
    """
    major = get_major(context)
    lang  = get_lang(context)

    if lang == "ar":
        lang_rule = "يجب أن تكون جميع ردودك باللغة العربية فقط. لا تستخدم الإنجليزية."
    else:
        lang_rule = "You must respond entirely in English. Do not use Arabic."

    return (
        f"You are an expert AI Academic Advisor specialising in {major} "
        f"at the university level.\n"
        f"{lang_rule}\n\n"
        f"Task: {task}\n\n"
        "Style rules:\n"
        "- Use clear section headings in bold.\n"
        "- Use numbered lists for steps and bullet points for items.\n"
        "- Keep paragraphs short and easy to read.\n"
        "- Use a professional, academic tone.\n"
        "- Use minimal emojis — at most one per section heading.\n"
        "- Always write a COMPLETE answer. Never stop mid-sentence or mid-section.\n"
        f"- Tailor every example and explanation to a {major} university student.\n"
    )


async def send_reply(update: Update, text: str) -> None:
    """
    Send a reply with the main menu keyboard attached.
    Automatically splits messages that exceed Telegram's 4096-char limit.
    """
    if len(text) <= 4096:
        await update.message.reply_text(
            text,
            reply_markup=MAIN_MENU_KEYBOARD,
        )
    else:
        chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)]
        for idx, chunk in enumerate(chunks):
            await update.message.reply_text(
                chunk,
                reply_markup=MAIN_MENU_KEYBOARD if idx == len(chunks) - 1 else None,
            )


# ─────────────────────────────────────────────
#  TASK PROMPTS
#  Detailed instructions for each feature so Gemini
#  produces complete, well-structured responses.
# ─────────────────────────────────────────────

def get_task_prompt(action: str, topic: str) -> str:
    prompts = {
        "explain": (
            f"Explain the topic '{topic}' clearly and completely.\n"
            "Structure your answer with these sections:\n"
            "1. Definition — what it is in 2-3 sentences.\n"
            "2. Core Concepts — the 4-6 most important ideas, each briefly explained.\n"
            "3. Real-World Example — a concrete example relevant to the student's field.\n"
            "4. Common Misconceptions — 2-3 things students often get wrong.\n"
            "5. Key Takeaways — 3 concise bullet points to remember.\n"
            "Do not skip any section."
        ),
        "plan": (
            f"Create a comprehensive study plan for the topic '{topic}'.\n"
            "Structure your answer with these sections:\n"
            "1. Overview — why this topic matters (2-3 sentences).\n"
            "2. Recommended Study Order — subtopics listed in the order they should be studied.\n"
            "3. Weekly Schedule — a realistic 2-4 week plan with daily tasks and time estimates.\n"
            "4. Study Tips — 5 specific, actionable tips for this topic.\n"
            "5. Milestones — 3-4 checkpoints to verify understanding.\n"
            "6. Recommended Resources — 2 books, 2 online courses or websites, 1 YouTube channel.\n"
            "Be detailed and practical. Do not skip any section."
        ),
        "summary": (
            f"Write a thorough, well-structured summary of '{topic}'.\n"
            "Structure your answer with these sections:\n"
            "1. Overview — what this topic is about (3-4 sentences).\n"
            "2. Background — historical context or how this topic developed.\n"
            "3. Key Concepts — the 5-8 most important ideas, each explained in 2-3 sentences.\n"
            "4. Important Terms — a glossary of 5-7 essential terms with short definitions.\n"
            "5. Connections — how this topic relates to other topics in the field.\n"
            "6. Conclusion — a 2-3 sentence wrap-up of the main ideas.\n"
            "Be complete. Do not cut off any section."
        ),
        "quiz": (
            f"Generate a complete practice quiz on '{topic}'.\n"
            "Include exactly:\n"
            "- 3 Multiple-choice questions (4 options each; mark the correct answer).\n"
            "- 2 True/False questions (state the answer and explain why in 2-3 sentences).\n"
            "- 2 Short-answer questions (provide a model answer of 2-4 sentences each).\n"
            "- 1 Applied scenario question relevant to the student's major (provide a model answer).\n"
            "After each question, provide the answer and a brief explanation."
        ),
        "resources": (
            f"Recommend the best learning resources for '{topic}'.\n"
            "Organise as:\n"
            "1. Books — 3 titles with author and a one-sentence description.\n"
            "2. Online Courses / Websites — 3 platforms or courses (include URL if well-known).\n"
            "3. YouTube Channels or Videos — 2-3 specific recommendations.\n"
            "4. Apps or Tools — 2 practical tools that support studying this topic.\n"
            "5. Study Tip — one piece of advice on how to use these resources effectively.\n"
            "Be specific. Avoid vague or generic recommendations."
        ),
    }
    return prompts.get(action, f"Help the student understand: {topic}")


async def run_feature(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    action: str,
    topic: str,
) -> None:
    """Core dispatcher: build the prompt, call Gemini, send the result."""
    lang = get_lang(context)

    # Thinking indicator (language-aware)
    thinking_text = {
        "explain":   ("Generating explanation...",   "جاري الشرح..."),
        "plan":      ("Building your study plan...", "جاري إنشاء خطة الدراسة..."),
        "summary":   ("Summarising...",              "جاري التلخيص..."),
        "quiz":      ("Creating quiz questions...",  "جاري إعداد الأسئلة..."),
        "resources": ("Finding resources...",        "جاري البحث عن المصادر..."),
    }
    en_t, ar_t = thinking_text.get(action, ("Working...", "جاري العمل..."))
    indicator = await update.message.reply_text(ar_t if lang == "ar" else en_t)

    # Response headers
    headers = {
        "explain":   (f"Explanation: {topic}",   f"شرح: {topic}"),
        "plan":      (f"Study Plan: {topic}",    f"خطة دراسة: {topic}"),
        "summary":   (f"Summary: {topic}",       f"ملخص: {topic}"),
        "quiz":      (f"Quiz: {topic}",          f"اختبار: {topic}"),
        "resources": (f"Resources: {topic}",     f"مصادر: {topic}"),
    }
    en_h, ar_h = headers.get(action, (topic, topic))
    header = ar_h if lang == "ar" else en_h

    task   = get_task_prompt(action, topic)
    system = build_system_prompt(context, task)
    reply  = ask_gemini(system, topic)

    await indicator.delete()
    await send_reply(update, f"*{header}*\n\n{reply}")


# ═════════════════════════════════════════════════════════
#  ONBOARDING CONVERSATION
#  /start → language choice → major choice (→ type major if "Other")
# ═════════════════════════════════════════════════════════

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Entry point: clear previous session and ask for language."""
    context.user_data.clear()
    name = update.effective_user.first_name or "Student"
    await update.message.reply_text(
        f"Welcome, {name}!\n\n"
        "Please choose your preferred language:\n"
        "اختر لغتك المفضلة:",
        reply_markup=LANGUAGE_KEYBOARD,
    )
    return CHOOSING_LANGUAGE


async def receive_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Save the chosen language and ask for the student's major."""
    text = update.message.text.strip()
    if text == "العربية":
        context.user_data["lang"] = "ar"
        prompt = "تم اختيار اللغة العربية.\n\nالآن اختر تخصصك الجامعي:"
    else:
        context.user_data["lang"] = "en"
        prompt = "English selected.\n\nNow please choose your major:"

    await update.message.reply_text(prompt, reply_markup=MAJOR_KEYBOARD)
    return CHOOSING_MAJOR


async def receive_major(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Save the chosen major.
    If the user picked 'Other', ask them to type their actual major.
    """
    text = update.message.text.strip()

    if text == "Other":
        ask = tr(context, "Please type your major.", "اكتب تخصصك من فضلك.")
        await update.message.reply_text(ask, reply_markup=ReplyKeyboardRemove())
        return TYPING_MAJOR

    context.user_data["major"] = text
    return await _finish_onboarding(update, context)


async def receive_typed_major(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Save the custom major the user typed and finish onboarding."""
    context.user_data["major"] = update.message.text.strip().title()
    return await _finish_onboarding(update, context)


async def _finish_onboarding(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Confirm saved settings and show the main menu."""
    major = get_major(context)
    msg = tr(
        context,
        f"All set.\n\nMajor: {major}\n\nUse the menu below to get started.",
        f"تم الحفظ.\n\nالتخصص: {major}\n\nاضغط على أي زر للبدء.",
    )
    await update.message.reply_text(msg, reply_markup=MAIN_MENU_KEYBOARD)
    return ConversationHandler.END


# ═════════════════════════════════════════════════════════
#  FEATURE CONVERSATION
#  Feature button → bot asks for topic → user types topic → response
# ═════════════════════════════════════════════════════════

async def feature_button_pressed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Triggered when the user taps a feature button (Explain, Study Plan, etc.).
    Saves the action and asks for the topic.
    """
    action = FEATURE_LABEL_TO_ACTION.get(update.message.text.strip())
    if not action:
        return ConversationHandler.END

    context.user_data["pending_action"] = action

    topic_prompts = {
        "explain":   ("What topic would you like explained?",
                      "ما الموضوع الذي تريد شرحه؟"),
        "plan":      ("What topic should the study plan cover?",
                      "ما الموضوع الذي تريد خطة دراسة له؟"),
        "summary":   ("What topic or lesson should I summarise?",
                      "ما الدرس أو الموضوع الذي تريد تلخيصه؟"),
        "quiz":      ("What topic should the quiz be on?",
                      "ما الموضوع الذي تريد أسئلة تدريبية عنه؟"),
        "resources": ("What topic should I find resources for?",
                      "ما الموضوع الذي تريد مصادر تعليمية عنه؟"),
    }
    en_p, ar_p = topic_prompts[action]
    lang = get_lang(context)
    await update.message.reply_text(
        ar_p if lang == "ar" else en_p,
        reply_markup=ReplyKeyboardRemove(),
    )
    return AWAITING_TOPIC


async def receive_topic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receive the topic and run the selected feature."""
    topic  = update.message.text.strip()
    action = context.user_data.pop("pending_action", None)

    if not action or not topic:
        await update.message.reply_text(
            tr(context, "Something went wrong. Please try again.", "حدث خطأ. حاول مرة أخرى."),
            reply_markup=MAIN_MENU_KEYBOARD,
        )
        return ConversationHandler.END

    await run_feature(update, context, action, topic)
    return ConversationHandler.END


# ═════════════════════════════════════════════════════════
#  /help COMMAND
# ═════════════════════════════════════════════════════════

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show a clean help message in the user's chosen language."""
    major = get_major(context)
    lang  = get_lang(context)

    if lang == "ar":
        text = (
            f"*المساعدة — المستشار الأكاديمي*\n\n"
            f"التخصص المحفوظ: {major}\n\n"
            "الأزرار المتاحة:\n"
            "Explain — اشرح موضوعاً بوضوح\n"
            "Study Plan — خطة دراسية مفصّلة\n"
            "Summary — ملخص درس أو موضوع\n"
            "Quiz — أسئلة تدريبية مع إجابات\n"
            "Resources — كتب ومواقع وقنوات تعليمية\n\n"
            "لتغيير التخصص أو اللغة أرسل /start"
        )
    else:
        text = (
            f"*AI Academic Advisor — Help*\n\n"
            f"Saved major: {major}\n\n"
            "Available actions:\n"
            "Explain — clear explanation of any topic\n"
            "Study Plan — detailed, personalised study plan\n"
            "Summary — structured lesson or topic summary\n"
            "Quiz — practice questions with model answers\n"
            "Resources — books, sites, and videos\n\n"
            "To change your major or language: send /start"
        )
    await update.message.reply_text(
        text, reply_markup=MAIN_MENU_KEYBOARD
    )


# ═════════════════════════════════════════════════════════
#  FREE-TEXT FALLBACK
#  Handles direct questions and the "Help" button
# ═════════════════════════════════════════════════════════

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle any plain-text message that falls outside a conversation.
    Treats it as a direct academic question answered by Gemini.
    """
    text = update.message.text.strip()

    if text == "Help":
        await help_command(update, context)
        return

    task = (
        "Answer the student's academic question clearly and completely. "
        "If the question is not academic, politely let the student know "
        "and suggest they use the menu buttons."
    )
    system = build_system_prompt(context, task)
    thinking = await update.message.reply_text(
        tr(context, "Thinking...", "جاري التفكير...")
    )
    reply = ask_gemini(system, text)
    await thinking.delete()
    await send_reply(update, reply)


# ─────────────────────────────────────────────
#  ERROR HANDLER
# ─────────────────────────────────────────────

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors and notify the user."""
    logger.error("Exception while handling an update:", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text(
            "An unexpected error occurred. Please try again.\n"
            "حدث خطأ غير متوقع. يرجى المحاولة مرة أخرى.",
            reply_markup=MAIN_MENU_KEYBOARD,
        )


# ═════════════════════════════════════════════════════════
#  MAIN — wire everything together and run
# ═════════════════════════════════════════════════════════

def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # ── 1. Onboarding conversation ──────────────────────────────────────────
    # /start → language → major (→ type major if "Other")
    onboarding_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            CHOOSING_LANGUAGE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_language)
            ],
            CHOOSING_MAJOR: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_major)
            ],
            TYPING_MAJOR: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_typed_major)
            ],
        },
        fallbacks=[CommandHandler("start", start)],
    )

    # ── 2. Feature conversation ─────────────────────────────────────────────
    # Feature button tap → bot asks for topic → user types topic → response
    feature_handler = ConversationHandler(
        entry_points=[
            MessageHandler(
                filters.TEXT & filters.Regex(
                    r"^(Explain|Study Plan|Summary|Quiz|Resources)$"
                ),
                feature_button_pressed,
            )
        ],
        states={
            AWAITING_TOPIC: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_topic)
            ],
        },
        fallbacks=[CommandHandler("start", start)],
    )

    # Register in order: onboarding first, features second, catch-all last
    app.add_handler(onboarding_handler)
    app.add_handler(feature_handler)
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_error_handler(error_handler)

    logger.info("AI Academic Advisor Bot is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

"""Gradio chatbot UI for Zoo Audio AI.

Wraps prediction, feedback collection, stats, fine-tuning, and model version
management into a conversational interface.
"""

import io
import random
import shutil
import uuid
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr

from zoo_audio.feedback import (
    FeedbackEntry,
    FeedbackStore,
    fine_tune,
    get_model_versions,
)
from zoo_audio.predict import predict
from zoo_audio.train import load_config


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """Per-session state for the Gradio chatbot."""

    chat_state: str = "IDLE"
    active_checkpoint: str = ""
    last_prediction: dict | None = None
    feedback_store: FeedbackStore | None = None
    config: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.config:
            self.config = load_config()
        if self.feedback_store is None:
            fb_dir = self.config.get("feedback", {}).get(
                "feedback_dir", "data/feedback"
            )
            self.feedback_store = FeedbackStore(fb_dir)
        if not self.active_checkpoint:
            cp_dir = self.config.get("data", {}).get("checkpoint_dir", "checkpoints")
            self.active_checkpoint = str(Path(cp_dir) / "best_model.pth")


# ---------------------------------------------------------------------------
# Audio persistence
# ---------------------------------------------------------------------------

def _persist_audio(audio_path: str, feedback_dir: str = "data/feedback") -> str:
    """Copy uploaded audio to a permanent location and return the new path."""
    src = Path(audio_path)
    dest_dir = Path(feedback_dir) / "audio"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4()}{src.suffix}"
    shutil.copy2(str(src), str(dest))
    return str(dest.resolve())


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_predictions(result: dict) -> str:
    """Format prediction results as a markdown table with bars."""
    preds = result["predictions"]
    lines = [
        "| Rank | Animal | Confidence | |",
        "|------|--------|------------|---|",
    ]
    for i, p in enumerate(preds, 1):
        bar = "\u2588" * int(p["probability"] * 20)
        lines.append(
            f"| {i} | **{p['class']}** | {p['probability']:.1%} | {bar} |"
        )
    lines.append(
        f"\n*Model: {result['model_version']} \u2014 "
        f"Checkpoint: `{Path(result['checkpoint_path']).name}`*"
    )
    return "\n".join(lines)


def _format_versions(versions: list[dict]) -> str:
    """Format model versions as a markdown table."""
    if not versions:
        return "No model checkpoints found."
    lines = [
        "| Version | Classes | Accuracy | Feedback | Parent | File |",
        "|---------|---------|----------|----------|--------|------|",
    ]
    for v in versions:
        acc = (
            f"{v['val_accuracy']:.4f}"
            if v["val_accuracy"] is not None
            else "N/A"
        )
        parent = v["parent"] or "\u2014"
        fname = Path(v["file"]).name
        lines.append(
            f"| {v['version']} | {v['num_classes']} | {acc} | "
            f"{v['feedback_count']} | {parent} | `{fname}` |"
        )
    return "\n".join(lines)


def _format_stats(entries: list[FeedbackEntry]) -> str:
    """Format feedback statistics as markdown."""
    if not entries:
        return "No feedback entries recorded yet."

    total = len(entries)
    correct = sum(1 for e in entries if e.is_correct)
    wrong = total - correct

    lines = [
        "**Feedback Summary**",
        f"- Total entries: **{total}**",
        f"- Confirmed correct: **{correct}**",
        f"- Corrections: **{wrong}**",
    ]

    corrections = [e for e in entries if not e.is_correct]
    if corrections:
        lines.append("\n**Corrections by true label:**")
        counts = Counter(e.true_label for e in corrections)
        for label, count in counts.most_common():
            lines.append(f"- {label}: {count}")

    predicted_labels = set(e.predicted_label for e in entries)
    true_labels = set(e.true_label for e in entries)
    new_cats = true_labels - predicted_labels
    if new_cats:
        lines.append(
            f"\n**New categories in feedback:** {', '.join(sorted(new_cats))}"
        )

    return "\n".join(lines)


def _format_history(entries: list[FeedbackEntry], limit: int = 20) -> str:
    """Format last N feedback entries as a markdown table."""
    if not entries:
        return "No feedback entries recorded yet."

    recent = entries[-limit:]
    lines = [
        f"*Showing last {len(recent)} of {len(entries)} entries*\n",
        "| ID | Timestamp | Predicted | Conf | True | Correct |",
        "|-----|-----------|-----------|------|------|---------|",
    ]
    for e in recent:
        short_id = e.id[:8]
        ts = e.timestamp[:19]
        correct_str = "yes" if e.is_correct else "no"
        lines.append(
            f"| {short_id} | {ts} | {e.predicted_label} | "
            f"{e.predicted_confidence:.1%} | {e.true_label} | {correct_str} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_help() -> str:
    return (
        "**Available Commands**\n\n"
        "| Command | Description |\n"
        "|---------|-------------|\n"
        "| `/help` | Show this help message |\n"
        "| `/stats` | Feedback summary statistics |\n"
        "| `/history` | Last 20 feedback entries |\n"
        "| `/versions` | List model checkpoints |\n"
        "| `/switch <version>` | Switch active model (e.g. `/switch v2`) |\n"
        "| `/finetune` | Fine-tune model with collected feedback |\n"
        "| `/export` | Export feedback as CSV |\n"
        "| `/play <animal>` | Play a random sample (e.g. `/play dog`) |\n"
        "\n**Audio upload:** Use the audio widget to upload or record a sound.\n"
        "After prediction, type **yes** to confirm, a **label** to correct, "
        "or **skip** to discard."
    )


def _cmd_stats(state: SessionState) -> str:
    entries = state.feedback_store.load_all()
    return _format_stats(entries)


def _cmd_history(state: SessionState) -> str:
    entries = state.feedback_store.load_all()
    return _format_history(entries)


def _cmd_versions(state: SessionState) -> str:
    cp_dir = state.config.get("data", {}).get("checkpoint_dir", "checkpoints")
    versions = get_model_versions(cp_dir)
    table = _format_versions(versions)
    table += f"\n\n*Active checkpoint:* `{Path(state.active_checkpoint).name}`"
    return table


def _cmd_switch(state: SessionState, version: str) -> str:
    cp_dir = state.config.get("data", {}).get("checkpoint_dir", "checkpoints")
    versions = get_model_versions(cp_dir)

    target = None
    for v in versions:
        if v["version"] == version:
            target = v
            break

    if target is None:
        available = ", ".join(v["version"] for v in versions)
        return f"Version `{version}` not found. Available: {available or 'none'}"

    state.active_checkpoint = target["file"]
    return f"Switched to **{version}** (`{Path(target['file']).name}`)"


def _cmd_finetune(state: SessionState) -> str:
    fb_config = state.config.get("feedback", {})
    epochs = fb_config.get("fine_tune_epochs", 10)
    lr = fb_config.get("fine_tune_lr", 0.0001)
    freeze_blocks = fb_config.get("freeze_conv_blocks", 2)
    min_feedback = fb_config.get("min_feedback", 5)

    entries = state.feedback_store.load_all()
    if len(entries) < min_feedback:
        return (
            f"Not enough feedback to fine-tune. "
            f"Have {len(entries)}, need at least {min_feedback}."
        )

    if not Path(state.active_checkpoint).exists():
        return f"Checkpoint not found: `{state.active_checkpoint}`"

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            new_path = fine_tune(
                config=state.config,
                feedback_entries=entries,
                base_checkpoint=state.active_checkpoint,
                epochs=epochs,
                lr=lr,
                freeze_conv_blocks=freeze_blocks,
            )
        log = buf.getvalue()
        state.active_checkpoint = new_path
        return (
            f"Fine-tuning complete! New checkpoint: "
            f"`{Path(new_path).name}`\n\n"
            f"<details><summary>Training log</summary>\n\n"
            f"```\n{log}\n```\n</details>"
        )
    except Exception as e:
        log = buf.getvalue()
        return (
            f"Fine-tuning failed: {e}\n\n"
            f"<details><summary>Partial log</summary>\n\n"
            f"```\n{log}\n```\n</details>"
        )


def _cmd_export(state: SessionState) -> str:
    export_dir = Path(
        state.config.get("feedback", {}).get("feedback_dir", "data/feedback")
    )
    csv_path = export_dir / "feedback_export.csv"

    entries = state.feedback_store.load_all()
    if not entries:
        return "No feedback entries to export."

    buf = io.StringIO()
    with redirect_stdout(buf):
        state.feedback_store.export_csv(str(csv_path))
    return f"Exported {len(entries)} entries to `{csv_path}`"


def _cmd_play(state: SessionState, animal: str) -> tuple[str, str | None]:
    """Pick a random sample from data/organized/<animal>/ and return (message, audio_path)."""
    data_dir = Path(state.config.get("data", {}).get("data_dir", "data/organized"))
    animal_dir = data_dir / animal.lower()

    if not animal_dir.is_dir():
        available = sorted(
            d.name for d in data_dir.iterdir() if d.is_dir()
        ) if data_dir.is_dir() else []
        return (
            f"No samples found for **{animal}**. "
            f"Available: {', '.join(available) or 'none'}",
            None,
        )

    audio_files = list(animal_dir.glob("*.wav")) + list(animal_dir.glob("*.mp3"))
    if not audio_files:
        return f"No audio files found in `{animal_dir}`.", None

    chosen = random.choice(audio_files)
    return (
        f"Playing a random **{animal}** sound (`{chosen.name}`)",
        str(chosen.resolve()),
    )


# ---------------------------------------------------------------------------
# Core dispatch
# ---------------------------------------------------------------------------

def _handle_audio_upload(audio_path: str, state: SessionState) -> str:
    """Run prediction on uploaded audio."""
    feedback_dir = state.config.get("feedback", {}).get(
        "feedback_dir", "data/feedback"
    )
    persistent_path = _persist_audio(audio_path, feedback_dir)

    buf = io.StringIO()
    with redirect_stdout(buf):
        result = predict(
            audio_path=persistent_path,
            checkpoint_path=state.active_checkpoint,
        )

    if result is None:
        state.chat_state = "IDLE"
        return "Prediction failed. Make sure a trained model checkpoint exists."

    result["audio_path"] = persistent_path
    state.last_prediction = result
    state.chat_state = "AWAITING_FEEDBACK"

    table = _format_predictions(result)
    top = result["predictions"][0]
    return (
        f"{table}\n\n"
        f"Top prediction: **{top['class']}** ({top['probability']:.1%})\n\n"
        f"Is this correct? Type **yes** to confirm, a **label** to correct, "
        f"or **skip** to discard."
    )


def _handle_feedback_response(user_text: str, state: SessionState) -> str:
    """Process feedback response when in AWAITING_FEEDBACK state."""
    result = state.last_prediction
    top_pred = result["predictions"][0]
    predicted_label = top_pred["class"]
    predicted_conf = top_pred["probability"]
    text = user_text.strip().lower()

    if text == "skip":
        state.chat_state = "IDLE"
        state.last_prediction = None
        return "Feedback skipped."

    if text == "yes":
        is_correct = True
        true_label = predicted_label
    else:
        true_label = user_text.strip()
        is_correct = true_label.lower() == predicted_label.lower()

    entry = FeedbackEntry(
        audio_path=result["audio_path"],
        predicted_label=predicted_label,
        predicted_confidence=predicted_conf,
        true_label=true_label,
        is_correct=is_correct,
        model_version=result["model_version"],
        checkpoint_path=result["checkpoint_path"],
    )

    state.feedback_store.add(entry)
    state.chat_state = "IDLE"
    state.last_prediction = None

    if is_correct:
        return f"Feedback saved: confirmed **{predicted_label}**"
    return f"Feedback saved: corrected to **{true_label}**"


def _handle_command(user_text: str, state: SessionState) -> tuple[str, str | None]:
    """Dispatch slash commands. Returns (reply_text, playback_audio_path | None)."""
    parts = user_text.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/play":
        if not arg:
            return "Usage: `/play <animal>` (e.g. `/play dog`)", None
        return _cmd_play(state, arg)

    handlers = {
        "/help": lambda: _cmd_help(),
        "/stats": lambda: _cmd_stats(state),
        "/history": lambda: _cmd_history(state),
        "/versions": lambda: _cmd_versions(state),
        "/switch": lambda: (
            _cmd_switch(state, arg) if arg
            else "Usage: `/switch <version>` (e.g. `/switch v2`)"
        ),
        "/finetune": lambda: _cmd_finetune(state),
        "/export": lambda: _cmd_export(state),
    }

    handler = handlers.get(cmd)
    if handler:
        return handler(), None
    return f"Unknown command: `{cmd}`. Type `/help` for available commands.", None


def handle_message(
    user_text: str,
    audio_file,
    chat_history: list,
    state: SessionState,
):
    """Core message dispatcher for the chatbot.

    Returns (chat_history, state, status_text, text_clear, audio_input_clear, playback_audio).
    """
    user_text = user_text or ""
    bot_reply = ""
    playback_audio = None

    # Audio upload takes priority
    if audio_file is not None:
        user_msg = "[Audio uploaded]"
        chat_history = chat_history + [{"role": "user", "content": user_msg}]
        bot_reply = _handle_audio_upload(audio_file, state)
    elif state.chat_state == "AWAITING_FEEDBACK" and user_text.strip():
        chat_history = chat_history + [{"role": "user", "content": user_text}]
        bot_reply = _handle_feedback_response(user_text, state)
    elif user_text.strip().startswith("/"):
        chat_history = chat_history + [{"role": "user", "content": user_text}]
        bot_reply, playback_audio = _handle_command(user_text, state)
    elif user_text.strip():
        chat_history = chat_history + [{"role": "user", "content": user_text}]
        bot_reply = (
            "Upload an audio file to classify, or type a command.\n"
            "Type `/help` to see available commands."
        )
    else:
        return chat_history, state, _status_text(state), "", None, None

    chat_history = chat_history + [{"role": "assistant", "content": bot_reply}]
    return chat_history, state, _status_text(state), "", None, playback_audio


def _status_text(state: SessionState) -> str:
    """Generate status bar HTML badges for the header."""
    cp_name = Path(state.active_checkpoint).name
    count = len(state.feedback_store.load_all())
    state_cls = (
        "state-pill-idle" if state.chat_state == "IDLE" else "state-pill-awaiting"
    )
    return (
        f'<div class="status-row">'
        f'<span class="status-pill model-pill">{cp_name}</span>'
        f'<span class="status-pill {state_cls}">{state.chat_state}</span>'
        f'<span class="status-pill feedback-pill">{count} feedback</span>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

_GOOGLE_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800'
    '&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">'
)

_CUSTOM_CSS = """
/* ── Global · Dark Wildlife ───────────────────────────────────── */
:root {
    --bg: #0d0f1a;
    --surface: #161829;
    --surface-alt: #1e2038;
    --surface-high: #272a45;
    --border: rgba(255,255,255,0.06);
    --border-light: rgba(255,255,255,0.03);
    --primary: #4ade80;
    --primary-hover: #22c55e;
    --primary-glow: rgba(74,222,128,0.15);
    --primary-dim: rgba(74,222,128,0.08);
    --accent-gold: #fbbf24;
    --accent-gold-dim: rgba(251,191,36,0.12);
    --accent-purple: #a78bfa;
    --accent-purple-dim: rgba(167,139,250,0.12);
    --text: #e8e8ed;
    --text-secondary: #9399b2;
    --text-muted: #6b7085;
    --error: #f87171;
    --radius: 24px;
    --radius-lg: 28px;
    --radius-pill: 9999px;
    --font-headline: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --shadow-glow: 0 0 20px var(--primary-glow);
    --glass-bg: rgba(22,24,41,0.82);
    --glass-blur: blur(20px);
}

/* ── Base reset ───────────────────────────────────────────────── */
body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
}
.gradio-container {
    max-width: 860px !important;
    margin: 0 auto !important;
    font-family: var(--font) !important;
    padding: 0 16px !important;
}
footer { display: none !important; }

/* ── Glass Header ─────────────────────────────────────────────── */
#app-header {
    position: sticky;
    top: 0;
    z-index: 50;
    background: var(--glass-bg) !important;
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    border-bottom: 1px solid var(--border);
    padding: 14px 20px !important;
    margin: 0 -16px 0 !important;
    border-radius: 0 !important;
}
.header-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    max-width: 860px;
    margin: 0 auto;
}
.header-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.header-icon {
    width: 36px;
    height: 36px;
    border-radius: 12px;
    background: var(--primary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    color: #0d0f1a;
    font-weight: 800;
    font-family: var(--font-headline);
    box-shadow: var(--shadow-glow);
}
.header-title {
    font-family: var(--font-headline);
    font-weight: 800;
    font-size: 18px;
    letter-spacing: -0.4px;
    color: var(--text);
}
.header-badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
#header-block, #header-subtitle {
    display: none !important;
}

/* ── Status Badges (in header) ────────────────────────────────── */
.status-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: var(--radius-pill);
    border: 1px solid var(--border);
    background: var(--surface-alt);
    color: var(--text-secondary);
    letter-spacing: 0.02em;
    font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
}
.model-pill {
    background: var(--accent-purple-dim);
    color: var(--accent-purple);
    border-color: rgba(167,139,250,0.2);
}
.feedback-pill {
    background: var(--accent-gold-dim);
    color: var(--accent-gold);
    border-color: rgba(251,191,36,0.2);
}
.state-pill-idle {
    background: var(--primary-dim);
    color: var(--primary);
    border-color: rgba(74,222,128,0.2);
}
.state-pill-awaiting {
    background: var(--accent-gold-dim);
    color: var(--accent-gold);
    border-color: rgba(251,191,36,0.2);
}

/* ── Welcome Card ─────────────────────────────────────────────── */
#welcome-card {
    padding: 0 !important;
    margin-bottom: 8px !important;
}
.welcome-wrap {
    background: linear-gradient(135deg, var(--surface-alt) 0%, #1a1640 50%, #1e1245 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 48px 40px 40px;
    position: relative;
    overflow: hidden;
}
.welcome-wave {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 60px;
    opacity: 0.06;
    background: repeating-linear-gradient(
        90deg,
        transparent,
        transparent 8px,
        var(--primary) 8px,
        var(--primary) 10px
    );
    mask-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 800 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 30 Q100 0 200 30 T400 30 T600 30 T800 30 V60 H0Z' fill='white'/%3E%3C/svg%3E");
    -webkit-mask-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 800 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 30 Q100 0 200 30 T400 30 T600 30 T800 30 V60 H0Z' fill='white'/%3E%3C/svg%3E");
}
.welcome-title {
    font-family: var(--font-headline);
    font-weight: 800;
    font-size: 28px;
    letter-spacing: -0.5px;
    color: var(--text);
    margin: 0 0 8px;
}
.welcome-subtitle {
    font-size: 15px;
    color: var(--text-muted);
    margin: 0 0 28px;
    line-height: 1.5;
}
.welcome-actions {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.welcome-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 18px;
    border-radius: var(--radius-pill);
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    color: var(--text-secondary);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    text-decoration: none;
}
.welcome-pill:hover {
    background: var(--primary-dim);
    border-color: rgba(74,222,128,0.25);
    color: var(--primary);
}
.welcome-pill .wp-icon {
    font-size: 15px;
    opacity: 0.7;
}

/* ── ZOO watermark ────────────────────────────────────────────── */
.zoo-watermark {
    position: fixed;
    bottom: -8%;
    right: -3%;
    font-family: var(--font-headline);
    font-weight: 800;
    font-size: 22rem;
    line-height: 1;
    color: var(--text);
    opacity: 0.018;
    pointer-events: none;
    user-select: none;
    z-index: 0;
    letter-spacing: -0.04em;
}

/* ── Chatbot ──────────────────────────────────────────────────── */
#chatbot {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
    box-shadow: none !important;
}
#chatbot .wrapper {
    background: var(--surface) !important;
}
#chatbot .message-row {
    padding: 16px 20px !important;
    max-width: 100% !important;
}
#chatbot .bot, #chatbot .user {
    border: none !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
}

/* User bubbles: green, right-aligned */
#chatbot .user {
    background: var(--primary) !important;
    color: #0d0f1a !important;
    border-radius: 24px 24px 6px 24px !important;
    box-shadow: 0 4px 16px var(--primary-glow) !important;
    font-weight: 500 !important;
}

/* Bot bubbles: dark surface */
#chatbot .bot {
    background: var(--surface-alt) !important;
    color: var(--text) !important;
    border-radius: 24px 24px 24px 6px !important;
}

/* Tables inside chat */
#chatbot table {
    font-size: 13px !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    width: 100%;
}
#chatbot th {
    background: var(--surface-high) !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    padding: 8px 12px !important;
    border-bottom: 1px solid var(--border) !important;
    text-align: left !important;
}
#chatbot td {
    padding: 6px 12px !important;
    border-bottom: 1px solid var(--border-light) !important;
    color: var(--text) !important;
    background: var(--surface) !important;
}
#chatbot tr:last-child td {
    border-bottom: none !important;
}
#chatbot code {
    background: var(--surface-high) !important;
    padding: 2px 6px !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    color: var(--accent-purple) !important;
}
#chatbot strong {
    color: var(--primary) !important;
}

/* ── Input area (floating pill) ───────────────────────────────── */
#input-container {
    background: var(--surface-alt) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-pill) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
    padding: 6px 6px 6px 16px !important;
    margin-top: 16px !important;
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
}
#input-container .gr-group,
#input-container > div {
    gap: 8px !important;
}

/* Text input */
#msg-input textarea {
    border: none !important;
    background: transparent !important;
    font-size: 14px !important;
    padding: 10px 4px !important;
    box-shadow: none !important;
    resize: none !important;
    font-family: var(--font) !important;
    color: var(--text) !important;
    min-height: 24px !important;
}
#msg-input textarea:focus {
    outline: none !important;
    box-shadow: none !important;
    border: none !important;
}
#msg-input textarea::placeholder {
    color: var(--text-muted) !important;
}
#msg-input {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

/* Send button (green circle) */
#send-btn {
    min-width: 44px !important;
    height: 44px !important;
    width: 44px !important;
    border-radius: 50% !important;
    background: var(--primary) !important;
    color: #0d0f1a !important;
    border: none !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
    flex-shrink: 0 !important;
    box-shadow: var(--shadow-glow) !important;
}
#send-btn:hover {
    background: var(--primary-hover) !important;
    transform: scale(1.05);
}

/* ── Audio widgets ────────────────────────────────────────────── */
#audio-upload {
    margin-top: 10px;
}
#audio-upload .wrap {
    border: 1.5px dashed rgba(255,255,255,0.1) !important;
    border-radius: 20px !important;
    background: var(--surface) !important;
    transition: border-color 0.2s ease;
}
#audio-upload .wrap:hover {
    border-color: var(--primary) !important;
}
#audio-upload button {
    color: var(--text-secondary) !important;
}
#audio-upload .text-lg {
    color: var(--text-muted) !important;
}

#playback-output {
    margin-top: 8px;
}
#playback-output .wrap {
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    background: var(--surface) !important;
}

/* ── Audio player dark overrides ──────────────────────────────── */
audio {
    filter: invert(1) hue-rotate(180deg);
    border-radius: 12px;
}

/* ── Labels ───────────────────────────────────────────────────── */
.gr-input-label, label, .label-wrap {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    font-family: var(--font) !important;
}

/* ── Blocks / panels ──────────────────────────────────────────── */
.gr-panel, .gr-box, .gr-form, .gr-block {
    background: transparent !important;
    border: none !important;
}
.contain {
    background: transparent !important;
}

/* ── Scrollbar ────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.14); }

/* ── Dark overrides for Gradio internals ──────────────────────── */
.gr-button {
    color: var(--text) !important;
}
.gr-input, .gr-text-input, textarea, input[type="text"] {
    color: var(--text) !important;
    background: var(--surface-alt) !important;
}
.label-wrap > span {
    color: var(--text-muted) !important;
}
.icon-wrap svg, .upload-icon svg {
    color: var(--text-muted) !important;
}
"""


# ---------------------------------------------------------------------------
# Gradio app builder
# ---------------------------------------------------------------------------

_WELCOME_HTML = """
<div class="welcome-wrap">
    <div class="welcome-wave"></div>
    <div class="welcome-title">Welcome to Zoo Audio AI</div>
    <div class="welcome-subtitle">
        Upload or record animal sounds to classify them instantly.
        Train and fine-tune your model with feedback.
    </div>
    <div class="welcome-actions">
        <span class="welcome-pill" onclick="
            document.querySelector('#msg-input textarea').value='/help';
            document.querySelector('#msg-input textarea').dispatchEvent(new Event('input',{bubbles:true}));
            document.querySelector('#send-btn').click();
        "><span class="wp-icon">?</span> Commands</span>
        <span class="welcome-pill" onclick="
            document.querySelector('#msg-input textarea').value='/stats';
            document.querySelector('#msg-input textarea').dispatchEvent(new Event('input',{bubbles:true}));
            document.querySelector('#send-btn').click();
        "><span class="wp-icon">#</span> Feedback Stats</span>
        <span class="welcome-pill" onclick="
            document.querySelector('#msg-input textarea').value='/versions';
            document.querySelector('#msg-input textarea').dispatchEvent(new Event('input',{bubbles:true}));
            document.querySelector('#send-btn').click();
        "><span class="wp-icon">&darr;</span> Model Versions</span>
    </div>
</div>
"""

_HEADER_HTML = """
<div class="header-inner">
    <div class="header-brand">
        <div class="header-icon">Z</div>
        <span class="header-title">Zoo Audio AI</span>
    </div>
    <div class="header-badges" id="header-badges-slot"></div>
</div>
"""

_WATERMARK_HTML = '<div class="zoo-watermark">ZOO</div>'


def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""
    with gr.Blocks(title="Zoo Audio AI") as app:
        gr.HTML(f"{_GOOGLE_FONTS}<style>{_CUSTOM_CSS}</style>{_WATERMARK_HTML}")
        state = gr.State(SessionState())

        # ── Glass header ──
        gr.HTML(_HEADER_HTML, elem_id="app-header")

        # ── Status badges (below header, flows into header visually) ──
        status = gr.Markdown("", elem_id="status-bar")

        # ── Welcome card ──
        gr.HTML(_WELCOME_HTML, elem_id="welcome-card")

        # ── Chat area ──
        chatbot = gr.Chatbot(
            label="Chat",
            height=480,
            elem_id="chatbot",
            show_label=False,
        )

        # ── Audio upload ──
        audio_input = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Upload or record audio",
            elem_id="audio-upload",
        )

        # ── Input bar (floating pill) ──
        with gr.Group(elem_id="input-container"):
            with gr.Row():
                text_input = gr.Textbox(
                    placeholder="Message Zoo Audio AI...",
                    show_label=False,
                    scale=6,
                    container=False,
                    elem_id="msg-input",
                    lines=1,
                    max_lines=4,
                )
                send_btn = gr.Button(
                    "\u2191",
                    variant="primary",
                    scale=0,
                    elem_id="send-btn",
                    min_width=44,
                )

        # ── Playback output ──
        playback_output = gr.Audio(
            label="Playback",
            type="filepath",
            autoplay=True,
            elem_id="playback-output",
        )

        # ── Event wiring ──
        def on_submit(user_text, audio_file, chat_history, sess_state):
            history, new_state, status_md, txt_clear, audio_clear, playback = (
                handle_message(user_text, audio_file, chat_history, sess_state)
            )
            return history, new_state, status_md, txt_clear, audio_clear, playback

        submit_inputs = [text_input, audio_input, chatbot, state]
        submit_outputs = [
            chatbot, state, status, text_input, audio_input, playback_output,
        ]

        send_btn.click(
            fn=on_submit,
            inputs=submit_inputs,
            outputs=submit_outputs,
        )
        text_input.submit(
            fn=on_submit,
            inputs=submit_inputs,
            outputs=submit_outputs,
        )

        app.load(
            fn=lambda s: _status_text(s),
            inputs=[state],
            outputs=[status],
        )

    return app


if __name__ == "__main__":
    build_app().launch()

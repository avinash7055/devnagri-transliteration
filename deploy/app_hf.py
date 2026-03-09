"""
Gradio-based interactive demo for multilingual transliteration.
Uses CTranslate2 optimized model for fast inference.

Deployed on Hugging Face Spaces.
"""

import os
import sys
from pathlib import Path

import gradio as gr

try:
    import ctranslate2
except ImportError:
    print("ERROR: ctranslate2 not installed. Run: pip install ctranslate2")
    sys.exit(1)


# Language configuration
LANGUAGES = {
    "Hindi (हिन्दी)": "hin",
    "Bengali (বাংলা)": "ben",
    "Tamil (தமிழ்)": "tam",
}

# Model directory — relative to script location for HF Spaces
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ct2_model")


class TransliterationModel:
    """Wrapper for CTranslate2 transliteration model."""

    def __init__(self, model_dir: str):
        """Load the CTranslate2 model."""
        self.model_dir = model_dir
        print(f"Loading model from: {model_dir}")
        self.translator = ctranslate2.Translator(model_dir, device="cpu")
        print("Model loaded successfully!")

    def transliterate(self, text: str, lang_code: str, beam_size: int = 5) -> str:
        """Transliterate English/romanized text to target script.

        Args:
            text: Input text (English/romanized)
            lang_code: Target language code (hin, ben, tam)
            beam_size: Beam size for decoding

        Returns:
            Transliterated text in target script
        """
        if not text.strip():
            return ""

        # Process word by word
        words = text.strip().split()
        transliterated_words = []

        for word in words:
            # Tokenize into characters with language prefix
            tokens = [f"<{lang_code}>"] + list(word.lower())

            # Run translation
            results = self.translator.translate_batch(
                [tokens],
                beam_size=beam_size,
                max_decoding_length=150,
            )

            # Extract output and join characters
            output_tokens = results[0].hypotheses[0]
            transliterated_word = "".join(output_tokens)
            transliterated_words.append(transliterated_word)

        return " ".join(transliterated_words)


# Load model at startup
print(f"Looking for model at: {MODEL_DIR}")
if not Path(MODEL_DIR).exists():
    print(f"ERROR: Model directory not found: {MODEL_DIR}")
    print("Make sure ct2_model/ directory exists with model.bin, config.json, and shared_vocabulary.json")
    sys.exit(1)

model = TransliterationModel(MODEL_DIR)


def transliterate_all(text):
    """Transliterate input to all three languages at once."""
    if not text or not text.strip():
        return "", "", ""

    hindi_out = model.transliterate(text, "hin")
    bengali_out = model.transliterate(text, "ben")
    tamil_out = model.transliterate(text, "tam")

    return hindi_out, bengali_out, tamil_out


def transliterate_selected(text, selected_langs):
    """Handle transliteration for selected languages."""
    hindi_out = ""
    bengali_out = ""
    tamil_out = ""

    if not text or not text.strip():
        return hindi_out, bengali_out, tamil_out

    for lang_name in selected_langs:
        lang_code = LANGUAGES[lang_name]
        result = model.transliterate(text, lang_code)

        if lang_code == "hin":
            hindi_out = result
        elif lang_code == "ben":
            bengali_out = result
        elif lang_code == "tam":
            tamil_out = result

    return hindi_out, bengali_out, tamil_out


def clear_all():
    return "", [], "", "", ""


# Custom CSS for premium look
css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
.main-header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}
.subtitle {
    text-align: center;
    color: #888;
    font-size: 1.1rem;
    margin-top: 0 !important;
    margin-bottom: 1.5rem !important;
}
.output-box textarea {
    font-size: 1.4rem !important;
    line-height: 2 !important;
    letter-spacing: 0.02em;
}
.model-info {
    text-align: center;
    padding: 1rem;
    border-radius: 12px;
    font-size: 0.85rem;
    color: #aaa;
    margin-top: 1rem;
}
.stats-row {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin: 1rem 0;
}
.stat-badge {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    text-align: center;
    color: #e2e8f0;
}
"""

# Build the Gradio interface
with gr.Blocks(
    title="Devnagri — Multilingual Transliteration",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
    ),
    css=css,
) as demo:
    gr.HTML("""
        <div style="text-align: center; padding: 1.5rem 0 0.5rem 0;">
            <h1 style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 2.8rem;
                font-weight: 800;
                margin: 0;
                letter-spacing: -0.02em;
            ">🔤 Devnagri</h1>
            <p style="color: #94a3b8; font-size: 1.15rem; margin-top: 0.3rem;">
                English → Hindi • Bengali • Tamil Transliteration
            </p>
            <div style="display: flex; justify-content: center; gap: 1.5rem; margin-top: 0.8rem; flex-wrap: wrap;">
                <span style="
                    background: linear-gradient(135deg, #1e293b, #334155);
                    border: 1px solid #475569;
                    border-radius: 20px;
                    padding: 0.4rem 1rem;
                    font-size: 0.85rem;
                    color: #cbd5e1;
                ">⚡ CTranslate2 Optimized</span>
                <span style="
                    background: linear-gradient(135deg, #1e293b, #334155);
                    border: 1px solid #475569;
                    border-radius: 20px;
                    padding: 0.4rem 1rem;
                    font-size: 0.85rem;
                    color: #cbd5e1;
                ">🧠 7.5M Param Transformer</span>
                <span style="
                    background: linear-gradient(135deg, #1e293b, #334155);
                    border: 1px solid #475569;
                    border-radius: 20px;
                    padding: 0.4rem 1rem;
                    font-size: 0.85rem;
                    color: #cbd5e1;
                ">📦 12 MB Model</span>
            </div>
        </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="✏️ Enter English / Romanized Text",
                placeholder="Type a word or phrase… e.g. namaste, kolkata, vanakkam",
                lines=3,
                max_lines=5,
            )

            language_select = gr.CheckboxGroup(
                choices=list(LANGUAGES.keys()),
                value=list(LANGUAGES.keys()),
                label="🌐 Target Languages",
            )

            with gr.Row():
                submit_btn = gr.Button(
                    "🔄 Transliterate",
                    variant="primary",
                    scale=2,
                )
                clear_btn = gr.Button("🗑️ Clear", scale=1)

        with gr.Column(scale=1):
            output_hindi = gr.Textbox(
                label="🇮🇳 Hindi (हिन्दी)",
                interactive=False,
                lines=2,
                elem_classes=["output-box"],
            )
            output_bengali = gr.Textbox(
                label="🇧🇩 Bengali (বাংলা)",
                interactive=False,
                lines=2,
                elem_classes=["output-box"],
            )
            output_tamil = gr.Textbox(
                label="🇱🇰 Tamil (தமிழ்)",
                interactive=False,
                lines=2,
                elem_classes=["output-box"],
            )

    # Examples section
    gr.Examples(
        examples=[
            ["namaste"],
            ["bharat"],
            ["diwali ki shubhkamnayein"],
            ["cricket world cup"],
            ["mumbai delhi kolkata chennai"],
            ["vanakkam"],
            ["dhanyabad"],
        ],
        inputs=input_text,
        label="📝 Try These Examples",
    )

    # Footer
    gr.HTML("""
        <div style="text-align: center; margin-top: 1.5rem; padding: 1rem; border-top: 1px solid #334155;">
            <p style="color: #64748b; font-size: 0.9rem; margin: 0.3rem 0;">
                Character-level Transformer trained on
                <a href="https://huggingface.co/datasets/ai4bharat/Aksharantar" target="_blank" style="color: #818cf8;">Aksharantar</a>
                dataset (300K pairs) · Optimized with
                <a href="https://github.com/OpenNMT/CTranslate2" target="_blank" style="color: #818cf8;">CTranslate2</a>
                int8 quantization
            </p>
            <p style="color: #475569; font-size: 0.8rem; margin: 0.3rem 0;">
                Built with OpenNMT-py · CTranslate2 · Gradio | 
                <a href="https://github.com/avinash7055/devnagri-transliteration" target="_blank" style="color: #818cf8;">GitHub</a>
            </p>
        </div>
    """)

    # Event handlers
    submit_btn.click(
        fn=transliterate_selected,
        inputs=[input_text, language_select],
        outputs=[output_hindi, output_bengali, output_tamil],
    )

    input_text.submit(
        fn=transliterate_selected,
        inputs=[input_text, language_select],
        outputs=[output_hindi, output_bengali, output_tamil],
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[input_text, language_select, output_hindi, output_bengali, output_tamil],
    )

# Launch for HF Spaces
demo.launch()

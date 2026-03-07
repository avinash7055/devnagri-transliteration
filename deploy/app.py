"""
Gradio-based interactive demo for multilingual transliteration.
Uses CTranslate2 optimized model for fast inference.

Usage:
    python deploy/app.py
    python deploy/app.py --model_dir models/ct2_model --port 7860
"""

import os
import sys
import argparse
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


def create_demo(model: TransliterationModel) -> gr.Blocks:
    """Create the Gradio demo interface."""

    css = """
    .main-title {
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .output-box {
        font-size: 1.3rem !important;
        line-height: 2 !important;
    }
    .lang-label {
        font-weight: bold;
        color: #4a9eff;
    }
    """

    with gr.Blocks(
        title="Devnagri - Multilingual Transliteration",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
            neutral_hue="slate",
        ),
        css=css,
    ) as demo:
        gr.Markdown(
            """
            # 🔤 Devnagri — Multilingual Transliteration
            ### Convert English/Romanized text to Hindi, Bengali & Tamil scripts
            """,
            elem_classes=["main-title"],
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="✏️ Enter English/Romanized Text",
                    placeholder="e.g., namaste, bharat, tamil",
                    lines=3,
                    max_lines=5,
                )

                language_select = gr.CheckboxGroup(
                    choices=list(LANGUAGES.keys()),
                    value=list(LANGUAGES.keys()),
                    label="🌐 Target Languages",
                )

                with gr.Row():
                    submit_btn = gr.Button("🔄 Transliterate", variant="primary", scale=2)
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

        # Examples
        gr.Examples(
            examples=[
                ["namaste"],
                ["bharat"],
                ["diwali ki shubhkamnayein"],
                ["cricket world cup"],
                ["mumbai delhi kolkata chennai"],
            ],
            inputs=input_text,
            label="📝 Try These Examples",
        )

        gr.Markdown(
            """
            ---
            **How it works:** This model uses a character-level Transformer trained on the
            [Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar) dataset,
            optimized with CTranslate2 for fast inference.

            Built with ❤️ using OpenNMT-py, CTranslate2, and Gradio.
            """,
        )

        def transliterate(text, selected_langs):
            """Handle transliteration for selected languages."""
            hindi_out = ""
            bengali_out = ""
            tamil_out = ""

            if not text.strip():
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

        submit_btn.click(
            fn=transliterate,
            inputs=[input_text, language_select],
            outputs=[output_hindi, output_bengali, output_tamil],
        )

        input_text.submit(
            fn=transliterate,
            inputs=[input_text, language_select],
            outputs=[output_hindi, output_bengali, output_tamil],
        )

        clear_btn.click(
            fn=clear_all,
            outputs=[input_text, language_select, output_hindi, output_bengali, output_tamil],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Transliteration Demo")
    parser.add_argument("--model_dir", type=str, default="models/ct2_model",
                        help="Path to CTranslate2 model directory")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to run the demo on")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    args = parser.parse_args()

    # Verify model exists
    if not Path(args.model_dir).exists():
        print(f"ERROR: Model directory not found: {args.model_dir}")
        print("Run 'python optimize/convert_ct2.py' first to create the CTranslate2 model.")
        sys.exit(1)

    # Load model
    model = TransliterationModel(args.model_dir)

    # Create and launch demo
    demo = create_demo(model)
    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0",
    )


if __name__ == "__main__":
    main()

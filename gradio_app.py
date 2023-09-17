import gradio as gr

from tts import models, tts_voices, tts


initial_md = """
# RVC text-to-speech webui

This is a text-to-speech webui of RVC models.

Input text ➡[(edge-tts)](https://github.com/rany2/edge-tts)➡ Speech mp3 file ➡[(RVC)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)➡ Final output
"""

app = gr.Blocks()
with app:
    gr.Markdown(initial_md)
    with gr.Row():
        with gr.Column():
            model_name = gr.Dropdown(label="Model", choices=models, value=models[0])
            f0_key_up = gr.Number(
                label="Transpose (the best value depends on the models and speakers)",
                value=0,
            )
        with gr.Column():
            f0_method = gr.Radio(
                label="Pitch extraction method (pm: very fast, low quality, rmvpe: a little slow, high quality)",
                choices=["pm", "rmvpe"],  # harvest and crepe is too slow
                value="rmvpe",
                interactive=True,
            )
            index_rate = gr.Slider(
                minimum=0,
                maximum=1,
                label="Index rate",
                value=1,
                interactive=True,
            )
            protect0 = gr.Slider(
                minimum=0,
                maximum=0.5,
                label="Protect",
                value=0.33,
                step=0.01,
                interactive=True,
            )
    with gr.Row():
        with gr.Column():
            tts_voice = gr.Dropdown(
                label="Edge-tts speaker (format: language-Country-Name-Gender)",
                choices=tts_voices,
                allow_custom_value=False,
                value="ja-JP-NanamiNeural-Female",
            )
            speed = gr.Slider(
                minimum=-100,
                maximum=100,
                label="Speech speed (%)",
                value=0,
                step=10,
                interactive=True,
            )
            tts_text = gr.Textbox(label="Input Text", value="これは日本語テキストから音声への変換デモです。")
        with gr.Column():
            but0 = gr.Button("Convert", variant="primary")
            info_text = gr.Textbox(label="Output info")
        with gr.Column():
            edge_tts_output = gr.Audio(label="Edge Voice", type="filepath")
            tts_output = gr.Audio(label="Result")
        but0.click(
            tts,
            [
                model_name,
                speed,
                tts_text,
                tts_voice,
                f0_key_up,
                f0_method,
                index_rate,
                protect0,
            ],
            [info_text, edge_tts_output, tts_output],
        )
    with gr.Row():
        examples = gr.Examples(
            examples_per_page=100,
            examples=[
                ["これは日本語テキストから音声への変換デモです。", "ja-JP-NanamiNeural-Female"],
                [
                    "This is an English text to speech conversation demo.",
                    "en-US-AriaNeural-Female",
                ],
                ["这是一个中文文本到语音的转换演示。", "zh-CN-XiaoxiaoNeural-Female"],
                ["한국어 텍스트에서 음성으로 변환하는 데모입니다.", "ko-KR-SunHiNeural-Female"],
                [
                    "Il s'agit d'une démo de conversion du texte français à la parole.",
                    "fr-FR-DeniseNeural-Female",
                ],
                [
                    "Dies ist eine Demo zur Umwandlung von Deutsch in Sprache.",
                    "de-DE-AmalaNeural-Female",
                ],
                [
                    "Tämä on suomenkielinen tekstistä puheeksi -esittely.",
                    "fi-FI-NooraNeural-Female",
                ],
                [
                    "Это демонстрационный пример преобразования русского текста в речь.",
                    "ru-RU-SvetlanaNeural-Female",
                ],
                [
                    "Αυτή είναι μια επίδειξη μετατροπής ελληνικού κειμένου σε ομιλία.",
                    "el-GR-AthinaNeural-Female",
                ],
                [
                    "Esta es una demostración de conversión de texto a voz en español.",
                    "es-ES-ElviraNeural-Female",
                ],
                [
                    "Questa è una dimostrazione di sintesi vocale in italiano.",
                    "it-IT-ElsaNeural-Female",
                ],
                [
                    "Esta é uma demonstração de conversão de texto em fala em português.",
                    "pt-PT-RaquelNeural-Female",
                ],
                [
                    "Це демонстрація тексту до мовлення українською мовою.",
                    "uk-UA-PolinaNeural-Female",
                ],
                [
                    "هذا عرض توضيحي عربي لتحويل النص إلى كلام.",
                    "ar-EG-SalmaNeural-Female",
                ],
                [
                    "இது தமிழ் உரையிலிருந்து பேச்சு மாற்ற டெமோ.",
                    "ta-IN-PallaviNeural-Female",
                ],
            ],
            inputs=[tts_text, tts_voice],
        )


app.launch(inbrowser=True)

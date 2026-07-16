import gradio as gr


def load_readme() -> str:
    with open("README.md", "r", encoding="utf-8") as file:
        return file.read()


demo = gr.Markdown(load_readme())


if __name__ == "__main__":
    demo.launch()


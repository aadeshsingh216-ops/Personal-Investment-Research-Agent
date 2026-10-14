"""Step 9: Gradio deployment with disclaimer."""
import gradio as gr
from agent import ask

DISCLAIMER = ("⚠️ **Disclaimer:** This tool is for educational research only. "
              "It is **not personalized financial advice**. Verify all figures before investing.")


def respond(message, history):
    msgs = [{"role": h["role"], "content": h["content"]} for h in history] + \
           [{"role": "user", "content": message}]
    return ask(msgs)


with gr.Blocks(title="Investment Research Agent") as demo:
    gr.Markdown("# 📈 Personal Investment Research Agent")
    gr.Markdown(DISCLAIMER)
    gr.ChatInterface(
        respond,
        type="messages",
        examples=[
            "What was the company's revenue in the latest year in the annual report?",
            "What is the current price of RELIANCE.NS?",
            "Compare the reported net profit in the report with today's share price of RELIANCE.NS",
            "Any recent news on TCS.NS?",
        ],
    )
    gr.Markdown(DISCLAIMER)

if __name__ == "__main__":
    demo.launch()

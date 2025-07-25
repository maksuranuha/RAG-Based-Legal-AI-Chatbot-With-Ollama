from main import run_rag_pipeline
import gradio as gr

def handle_question(uploaded_file, user_question):
    if not user_question.strip():
        return "Hey! Ask me something about Bangladesh cyber law!"
    
    try:
        answer = run_rag_pipeline(user_question, uploaded_file)
        return answer
    except Exception as e:
        return f"Oops! Something went wrong: {str(e)}"

with gr.Blocks(title="Legal AI Helper") as app:
    gr.Markdown("Ask questions about cyber law and more!")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                label="Upload PDF", 
                file_types=[".pdf"]
            )
            question_input = gr.Textbox(
                label="Your Question", 
                lines=4,
                placeholder="What do you want to know cyber law?"
            )
            submit_btn = gr.Button("Ask", variant="primary")
    
    with gr.Row():
        output = gr.Textbox(
            label="Legal Assistant Response", 
            lines=12, 
            interactive=False
        )
    
    submit_btn.click(
        fn=handle_question,
        inputs=[file_input, question_input], 
        outputs=output
    )

if __name__ == "__main__":
    app.launch(share=False)
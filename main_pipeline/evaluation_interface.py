import gradio as gr
import json
import random

# Load the dataset
with open('evaluation_dataset.json') as f:
    data = json.load(f)

# Store evaluation state
question_status = {q["id"]: [False] * 5 for q in data["questions"]}  # False for unanswered levels

# Function to check progress
def update_progress(selected_answers):
    progress = []
    for question_id, levels in question_status.items():
        completed_levels = sum(levels)
        color = "green" if completed_levels == 5 else "red"
        progress.append(f"Question {question_id}: {completed_levels}/5 levels evaluated")
    return "\n".join(progress)

# Function to evaluate responses
def evaluate_responses(*selected_answers):
    results = {}
    index = 0
    for question in data['questions']:
        question_id = question['id']
        results[question_id] = []
        for level in range(5):
            selected = selected_answers[index]
            results[question_id].append({"level": level, "selected": selected})
            question_status[question_id][level] = selected is not None
            index += 1
    # Save results to a JSON file
    with open("evaluation_results.json", "w") as outfile:
        json.dump(results, outfile, indent=4)

    progress_summary = update_progress(selected_answers)
    return "Preferences submitted successfully. Results saved to 'evaluation_results.json'.", progress_summary

# Create Gradio interface
def create_interface():
    questions = data['questions']

    # Randomize initial position of human and AI answers
    human_on_left = random.choice([True, False])

    inputs = []  # List to collect Radio components
    with gr.Blocks() as interface:
        gr.Markdown("## Evaluation Interface: Human vs AI Responses")
        progress_summary = gr.Textbox(value=update_progress([]), label="Progress Summary", interactive=False)

        for question_index, question in enumerate(questions):
            question_id = question['id']
            question_text = question['question']
            ground_truth = question['ground_truth']

            # Determine the positions for this question
            if (question_index % 2 == 0 and human_on_left) or (question_index % 2 != 0 and not human_on_left):
                left_label, right_label = "Human", "AI"
            else:
                left_label, right_label = "AI", "Human"

            with gr.Accordion(f"Question {question_id}: {question_text}", open=False):
                gr.Markdown(f"**Ground Truth:** {ground_truth}")

                for level, (human_response, ai_response) in enumerate(
                        [(q['human'], q['ai']) for q in question['answers'].values()]):
                    # Assign responses based on the determined positions
                    if left_label == "Human":
                        left_response, right_response = human_response, ai_response
                    else:
                        left_response, right_response = ai_response, human_response

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(f"**Response 1:** {left_response}")
                        with gr.Column():
                            gr.Markdown(f"**Response 2:** {right_response}")
                    radio = gr.Radio(
                        choices=["Response 1", "Response 2", "Both are good", "Both are bad"],
                        label=f"Level {level}: Which response is better?",
                        interactive=True
                    )
                    inputs.append(radio)

        submit_button = gr.Button("Submit Preferences")
        output_text = gr.Textbox(label="Evaluation Results", interactive=False)

        # Function to update progress summary
        def on_selection_change(*selections):
            progress_updates = update_progress(selections)
            return progress_updates

        # Update progress dynamically when answers are selected
        for radio in inputs:
            radio.change(on_selection_change, inputs=inputs, outputs=progress_summary)

        # Final submission button
        submit_button.click(evaluate_responses, inputs=inputs, outputs=[output_text, progress_summary])

    return interface

# Launch the interface
if __name__ == "__main__":
    create_interface().launch()

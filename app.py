"""Gradio interface for the bank subscription model from ML_Project.ipynb."""

from pathlib import Path

import gradio as gr
import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
TRAIN_PATH = ROOT / "Dataset" / "train_split.csv"

model = joblib.load(MODEL_PATH)
training_data = pd.read_csv(TRAIN_PATH)
feature_names = list(model.feature_names_in_)
feature_data = training_data[feature_names]

# Keep UI field types and category choices aligned with the training data.
numeric_features = [name for name in feature_names if pd.api.types.is_numeric_dtype(feature_data[name])]
categorical_choices = {
    name: sorted(feature_data[name].dropna().astype(str).unique().tolist())
    for name in feature_names
    if name not in numeric_features
}


def predict(*values):
    row = pd.DataFrame([dict(zip(feature_names, values))], columns=feature_names)
    predicted = int(model.predict(row)[0])
    label = "SUBSCRIBE (yes)" if predicted == 1 else "NOT SUBSCRIBE (no)"

    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)[0]
        classes = list(model.classes_)
        if 1 in classes:
            probability = float(probabilities[classes.index(1)])
    return label, probability


with gr.Blocks(title="Bank Subscription Predictor") as demo:
    gr.Markdown(
        "# Bank Subscription Predictor\n"
        "Enter a customer's pre-call details. `duration` is omitted, matching the notebook's "
        "pre-call prediction setup."
    )
    with gr.Row():
        with gr.Column():
            components = []
            for name in feature_names:
                if name in numeric_features:
                    component = gr.Number(
                        label=name,
                        value=float(feature_data[name].median()),
                        precision=0,
                    )
                else:
                    choices = categorical_choices[name]
                    component = gr.Dropdown(
                        choices=choices,
                        value=choices[0] if choices else None,
                        label=name,
                    )
                components.append(component)
            predict_button = gr.Button("Predict", variant="primary")
        with gr.Column():
            prediction_output = gr.Textbox(label="Prediction", interactive=False)
            probability_output = gr.Number(label="Estimated probability of subscription", interactive=False)

    outputs = [prediction_output, probability_output]
    predict_button.click(predict, inputs=components, outputs=outputs)
    gr.Examples(
        examples=feature_data.head(5).values.tolist(),
        inputs=components,
        outputs=outputs,
        fn=predict,
        cache_examples=False,
        label="Try example customers from the training dataset",
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)

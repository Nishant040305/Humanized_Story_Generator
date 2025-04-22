from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_path = "../models/transformer_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
if torch.cuda.is_available():
    model = model.to("cuda")


def tranformer_model(prompt):
    def generate_story(prompt, max_length=300):
        input_text = f"{prompt} : "
        inputs = tokenizer(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    story = generate_story(prompt)
    # print(story)
    return story

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
from pprint import pprint

dataset_id = 'hipml/survivor-subtitles-cleaned'
train_dataset = load_dataset(dataset_id)

checkpoint = 'answerdotai/ModernBERT-base'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForMaskedLM.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

text = "The capital of France is [MASK]."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print(outputs)

# print(inputs['input_ids'])
# print(inputs['input_ids'][0].tolist())
masked_index = inputs['input_ids'][0].tolist().index(tokenizer.mask_token_id)

predicted_token_id = outputs.logits[0, masked_index].argmax(axis=-1)
predicted_token = tokenizer.decode(predicted_token_id)
print(predicted_token)


pipe = pipeline(
    "fill-mask",
    model="answerdotai/ModernBERT-base",
    torch_dtype=torch.bfloat16,
)

input_text = "He walked to the [MASK]."
results = pipe(input_text)
pprint(results)


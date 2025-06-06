pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets scikit-learn

from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification

#carga del dataset
dataset = load_dataset("empathetic_dialogues")
#Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=32)

#procesamiemnto de dialogos
def tokenize(batch):
    return tokenizer(batch['utterance'], padding=True, truncation=True)

tokenized = dataset.map(tokenize, batched=True, batch_size=32)
tokenized = tokenized.rename_column("emotion", "labels")
tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

#entrenamiento
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./roberta_ed",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"]
)

trainer.train()

#Evaluar
trainer.evaluate()
model.save_pretrained("./roberta_ed_classif")
tokenizer.save_pretrained("./roberta_ed_classif")



#Adaptabilidad a modelo LLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel

gen_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gen_model = GPT2LMHeadModel.from_pretrained("gpt2")

#Fine tuning con el modelo RoBERTa
from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer

# Suponiendo que has creado un archivo de texto con diálogos concatenados
dataset_file = "empathetic_dialogues.txt"

train_dataset = TextDataset(
    tokenizer=gen_tokenizer,
    file_path=dataset_file,
    block_size=128
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=gen_tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir="./gpt2_ed",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8
)

trainer = Trainer(
    model=gen_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

trainer.train()

#(clasificador + generador)
def responder(texto_usuario):
    # Clasificar emoción
    inputs = tokenizer(texto_usuario, return_tensors="pt")
    outputs = model(**inputs)
    emotion_id = outputs.logits.argmax().item()
    emotion = id2label[emotion_id]  # Diccionario de id a etiqueta de emoción

    # Generar respuesta condicionada
    prompt = f"[EMOCION: {emotion}] {texto_usuario}\nRespuesta:"
    gen_inputs = gen_tokenizer(prompt, return_tensors="pt")
    gen_outputs = gen_model.generate(**gen_inputs, max_length=60)
    respuesta = gen_tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
    return respuesta









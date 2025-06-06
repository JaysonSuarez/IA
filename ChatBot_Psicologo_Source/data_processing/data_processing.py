from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BSC-TeMU/roberta-base-bne')
    return tokenizer(
        df['text'].tolist(), padding='max_length', truncation=True, max_length=128, return_tensors='pt'
    )
# AutoTokenizer: Carga un tokenizador pre-entrenado para RoBERTa.
# tokenize(): Aplica el tokenizador al texto de los DataFrames, convirtiéndolo en secuencias numéricas y añadiendo padding para que todas las secuencias tengan la misma longitud.

# ## Paso 6: Fine-Tuning del modelo RoBERTa

# In[9]:


    enc = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    # Move the encoded input to the same device as the model
    enc = {k: v.to(model.device) for k, v in enc.items()}
    logits = model(**enc).logits
    return le.inverse_transform([logits.argmax().item()])[0]


while True:
  user_input = input("Introduce el texto para predecir el sentimiento: ")
  if user_input.lower() in ["basta", "stop", "salir", "exit", "fin", "end"]:
    break

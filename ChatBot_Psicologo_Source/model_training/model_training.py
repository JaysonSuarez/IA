# Debe descargar el Dataset del siguiente enlace: http://tass.sepln.org/2020/?page_id=74. El archivo .zip llamado: Task 1 - train and dev sets.
# 
# Y para correr en colab, debe usarse la version que usa GPU para que pueda funcionar.

# ## Paso 1: Instalación de librerías necesarias

# In[1]:


get_ipython().system('pip install transformers scikit-learn pandas -q')


# In[2]:


get_ipython().system('pip install --upgrade transformers')


zip_path = '/content/Task1-train-dev.zip'
extract_dir = 'intertass'

with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(extract_dir)

df_train = load_split('train')
df_dev   = load_split('dev')

print(f"Train: {df_train.shape[0]} muestras")
print(f"Dev:   {df_dev.shape[0]} muestras")


# Carga el conjunto de datos InterTASS, que contiene texto etiquetado con emociones.
# 
# Descarga un archivo zip que contiene los datos de entrenamiento y desarrollo (train/dev).
# Extrae los archivos del zip.
# Lee los archivos TSV (texto separado por tabulaciones) y los combina en un DataFrame de pandas.
# El resultado son dos DataFrames: df_train para entrenamiento y df_dev para desarrollo.

# ## Paso 3: Exploración de datos

# In[4]:


print(df_train.head())
print(df_train['label'].value_counts())


# df_train.head(): Muestra las primeras filas del DataFrame de entrenamiento para ver su estructura y contenido.
# df_train['label'].value_counts(): Cuenta la frecuencia de cada etiqueta de emoción en el conjunto de datos, lo que ayuda a comprender la distribución de las clases.

# ## Paso 4: Preprocesamiento de texto y etiquetas

# In[5]:


df_train['label_enc'] = le.fit_transform(df_train['label'])
df_dev['label_enc']   = le.transform(df_dev['label'])
print(dict(zip(le.classes_, le.transform(le.classes_))))


train_enc = tokenize(df_train)
dev_enc   = tokenize(df_dev)
print(train_enc['input_ids'].shape, dev_enc['input_ids'].shape)


# Convierte el texto en una secuencia de tokens (unidades básicas de significado) que el modelo RoBERTa pueda entender.
# 
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
model = AutoModelForSequenceClassification.from_pretrained(
    'BSC-TeMU/roberta-base-bne', num_labels=len(le.classes_)
)
train_dataset = torch.utils.data.TensorDataset(
    train_enc['input_ids'], train_enc['attention_mask'], torch.tensor(df_train['label_enc'].values)
)
train_dataset = IntertassDataset(train_enc, df_train['label_enc'].values)
dev_dataset = IntertassDataset(dev_enc, df_dev['label_enc'].values)

training_args = TrainingArguments(
    output_dir='./results',
    do_train=True,
    do_eval=True,
    logging_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=2e-5,
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics
)
trainer.train()


# In[10]:


# 1. Distribution of Labels in the Training Set
plt.figure(figsize=(10, 5))
sns.countplot(x='label', data=df_train)
plt.title('Distribution of Labels in Training Set')
plt.xlabel('Sentiment Label')
plt.ylabel('Number of Samples')
plt.show()


# 2. Distribution of Labels in the Development Set
plt.figure(figsize=(10, 5))
sns.countplot(x='label', data=df_dev)
plt.title('Distribution of Labels in Development Set')
plt.xlabel('Sentiment Label')
plt.ylabel('Number of Samples')
plt.show()


# 3. Text Length Analysis (optional - requires text length calculation)
# Calculate the length of each text in the training set
df_train['text_length'] = df_train['text'].apply(len)

# Create the plot
plt.figure(figsize=(10, 5))
sns.histplot(df_train['text_length'], kde=True)
plt.title('Distribution of Text Lengths in the Training Set')
plt.xlabel('Text Length (Characters)')
plt.ylabel('Number of Samples')
plt.show()


# Entrena el modelo RoBERTa para la tarea específica de análisis de sentimientos en el dataset InterTASS.
# 
# AutoModelForSequenceClassification: Carga un modelo RoBERTa pre-entrenado y lo adapta para la clasificación de secuencias (en este caso, clasificar emociones).
# Trainer: Se encarga del proceso de entrenamiento, incluyendo la optimización de los parámetros del modelo.
# compute_metrics: Define las métricas que se usarán para evaluar el rendimiento del modelo durante el entrenamiento, como la precisión y la puntuación F1.

# ## Paso 7: Evaluación y prueba de ejemplo

# In[12]:


    "neu": ["confianza"]    # Neutral is harder to map; 'confianza' might fit some neutral contexts
    # You could add 'none': [] or similar if your dataset includes 'NONE'
}


#!/usr/bin/env python
# coding: utf-8

# # RoBERTa con InterTASS
# transformers: Proporciona acceso a modelos pre-entrenados como RoBERTa, utilizados para el procesamiento del lenguaje natural.
# scikit-learn: Ofrece herramientas para el aprendizaje automático, incluyendo la codificación de etiquetas.
# pandas: Permite la manipulación y análisis de datos con estructuras de datos como DataFrames.
# --upgrade transformers: actualiza la librería transformers para que tengas la versión mas nueva

# ## Paso 2: Carga del dataset InterTASS

# In[3]:


import zipfile, glob, pandas as pd

def load_split(split):
    files = glob.glob(f"{extract_dir}/{split}/*.tsv")
    dfs = []
    for path in files:
        country = path.split('/')[-1].split('.')[0]
        df = pd.read_csv(path, sep='\t', header=None, names=['id','text','label'])
        df['country'] = country
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
# Prepara los datos para ser utilizados por el modelo de aprendizaje automático.
# 
# LabelEncoder: Convierte las etiquetas de texto (como "joy", "sadness") a valores numéricos (0, 1, 2, etc.). Esto es necesario porque muchos modelos de aprendizaje automático trabajan con datos numéricos.

# ## Paso 5: Tokenización con RoBERTa

# In[6]:


def tokenize(df):
import torch
import os
os.environ["WANDB_DISABLED"] = "true"

dev_dataset   = torch.utils.data.TensorDataset(
    dev_enc['input_ids'], dev_enc['attention_mask'],   torch.tensor(df_dev['label_enc'].values)
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    from sklearn.metrics import f1_score
    return {
        'accuracy': (preds == labels).mean(),
        'f1_macro': f1_score(labels, preds, average='macro')
    }

class IntertassDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import random

        "Entiendo que te sientas enojado/a. Es importante reconocer esa emoción.",
        "La ira puede ser una señal de que algo no está bien. ¿Quieres hablar sobre ello?",
        "Tomarse un momento para respirar puede ayudar a manejar la ira.",
        "Estoy aquí para escucharte sin juzgar. Cuéntame qué te ha molestado.",
        "La ira es válida, pero encontrar formas saludables de expresarla es crucial."
    ],
    "miedo": [
        "El miedo es una emoción natural ante lo desconocido. Estoy aquí contigo.",
        "Hablar sobre lo que te asusta puede disminuir su poder. ¿Quieres compartirlo?",
        "Recuerda que eres fuerte y capaz de enfrentar tus temores.",
        "No estás solo/a en esto. Juntos podemos encontrar maneras de afrontarlo.",
        "Es valiente reconocer que sientes miedo. Estoy aquí para apoyarte."
    ],
    "alegría": [
        "¡Me alegra saber que te sientes bien! Comparte más sobre eso.",
        "Es maravilloso que experimentes alegría. Disfruta el momento.",
        "La alegría es contagiosa. Gracias por compartirla conmigo.",
        "Celebremos juntos tu felicidad. ¿Qué te ha hecho sentir así?",
        "Es genial escuchar noticias positivas. ¡Sigue compartiéndolas!"
    ],
    "sorpresa": [
        "¡Vaya, eso suena inesperado! ¿Cómo te sientes al respecto?",
        "Las sorpresas pueden ser emocionantes o desconcertantes. ¿Cuál fue tu reacción?",
        "Cuéntame más sobre lo que te sorprendió. Estoy interesado/a en saber.",
        "A veces, las sorpresas nos sacan de la rutina. ¿Fue una sorpresa agradable?",
        "Es interesante cómo ocurren cosas inesperadas. ¿Cómo lo estás manejando?"
    ],
    "desagrado": [
        "Entiendo que algo te ha causado desagrado. ¿Quieres hablar sobre ello?",
        "Es importante reconocer lo que no nos agrada. Estoy aquí para escucharte.",
        "Compartir lo que te molesta puede ayudarte a procesarlo.",
        "Todos experimentamos desagrado en algún momento. No estás solo/a.",
        "Hablar sobre lo que te incomoda puede ser liberador. ¿Te gustaría compartirlo?"
    ],
    "confianza": [
        "Es genial que te sientas confiado/a. Esa es una emoción poderosa.",
        "La confianza en uno mismo es clave para el bienestar. ¡Sigue así!",
        "Me alegra saber que te sientes seguro/a. ¿Qué ha contribuido a eso?",
        "La confianza puede abrir muchas puertas. ¿Hay algo que te gustaría compartir?",
        "Es inspirador ver tu confianza. ¿Qué te ha llevado a sentirte así?"
    ]
}

      import traceback
      traceback.print_exc()
  else:
    print("Por favor, ingresa algo sobre cómo te sientes.")


# In[35]:


get_ipython().system('pip install cohere -q')


# In[43]:


import cohere
import random # Ensure random is imported if needed for the second loop (which it is for random.choice)
import traceback # Ensure traceback is imported for error handling


co = cohere.ClientV2('uGJ50Z2ilig2S60S5GbsIcIEwp1FdzosdkXkDsm8')

print("""-Recuerda que este chatbot es solo una herramienta de apoyo y no un sustituto de la ayuda profesional.
def generate_cohere_response(emotion, user_text):
# # Este código ya utiliza una forma Prompt Chaining
# El segundo bucle while utiliza la polaridad_prevista del modelo RoBERTa

# ChatBot Psicólogo - RoBERTa + Generative Model

Este proyecto implementa un asistente psicológico usando modelos de lenguaje RoBERTa y generación de texto. Está organizado como un paquete ejecutable de Python con módulos bien separados para facilitar su uso y mantenimiento.

## Estructura del Proyecto

```
ChatBot_Psicologo_Source/
├── data_processing/
│   └── data_processing.py
├── model_training/
│   └── model_training.py
├── inference/
│   └── inference.py
├── utils/
│   └── utils.py
├── main.py
└── README.md
```

## Requisitos

- Python 3.8+
- Transformers
- Datasets
- Scikit-learn
- Pandas
- Torch

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Uso

Ejecutar el archivo principal:

```bash
python main.py
```

Este script coordina la carga de datos, entrenamiento del modelo y la inferencia de respuestas empáticas.

## Créditos

Este proyecto fue desarrollado como parte de un sistema de chatbot psicológico usando técnicas de NLP modernas.

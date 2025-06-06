"""
Script principal para entrenar y ejecutar el chatbot psicológico usando RoBERTa y generación empática.
"""

from data_processing.data_processing import *
from model_training.model_training import *
from inference.inference import *
from utils.utils import *

def main():
    print("Iniciando el ChatBot Psicológico...")

    # Cargar y procesar datos
    print("Cargando y procesando datos...")
    # TODO: Llamar funciones de data_processing

    # Entrenar el modelo
    print("Entrenando modelo...")
    # TODO: Llamar funciones de model_training

    # Ejecutar inferencia
    print("Generando respuesta empática...")
    # TODO: Llamar funciones de inference

if __name__ == "__main__":
    main()

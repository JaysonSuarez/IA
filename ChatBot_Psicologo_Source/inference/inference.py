def predict(text):
  prediction = predict(user_input)
  print(f"El sentimiento predicho es: {prediction}")


#  Toma un texto como entrada, lo tokeniza, lo pasa por el modelo y devuelve la etiqueta de emoción predicha.

# Ya que RoBERTa no es un LLM sino un Classifier, generamos un diccionario de posibles respuestas a las preguntas del user con el fin de:
# 
# **1. Clasificar la emoción**
# 
# **2. Analizar la emoción**
# 
# **3. "Generar" respuestas segun la emoción clasificada**

# In[26]:


# Diccionario de respuestas según emoción
respuestas = {
    "tristeza": [
        "Lamento que te sientas así. Estoy aquí para escucharte.",
        "Es normal sentirse triste a veces. ¿Quieres hablar más al respecto?",
        "Tus emociones son válidas. No estás solo/a en esto.",
        "Estoy aquí para ti. Cuéntame más si lo deseas.",
        "La tristeza es una emoción humana. Permítete sentirla y expresarla."
    ],
    "ansiedad": [
        "Parece que estás experimentando ansiedad. Respirar profundamente puede ayudarte a calmarte.",
        "La ansiedad puede ser abrumadora. Estoy aquí para apoyarte.",
        "Recuerda que no estás solo/a. Hablar sobre lo que sientes puede aliviar la carga.",
        "¿Hay algo específico que te preocupa? Compartirlo puede ayudarte a procesarlo.",
        "La ansiedad es una respuesta natural al estrés. Juntos podemos encontrar formas de manejarla."
    ],
    "ira": [
# Función para obtener una respuesta empática
def obtener_respuesta_emocional(emocion):
    return random.choice(respuestas.get(emocion, ["Estoy aquí para ti. Cuéntame más."]))


# In[42]:


polarity_to_possible_emotions = {
    "p": ["alegría","sorpresa"],       # Positive could relate to joy or surprise
    "n": ["tristeza", "miedo", "ansiedad", "ira", "desagrado", "frustrado"], # Negative could relate to several negative emotions
# Permitir al usuario ingresar texto y obtener una respuesta basada en la polaridad detectada
while True:
  user_input = input("¿Cómo te sientes hoy? (escribe 'salir' para terminar): ")
  if user_input.lower() in ["salir", "terminar", "adiós"]:
    print("¡Hasta luego!")
    break
  if user_input:
    try:
      # Usar la función predict para obtener la POLARITY (P, N, NEU)
      predicted_polarity = predict(user_input)
      print(f"La polaridad predicha es: {predicted_polarity.capitalize()}")

      # Convert the predicted polarity to lowercase
      predicted_polarity_lower = predicted_polarity.lower()

      # Get the list of possible emotions for this polarity
      possible_emotions = polarity_to_possible_emotions.get(predicted_polarity_lower, []) # Default to empty list if polarity not found

      respuesta = "Entiendo. Estoy aquí para escucharte si quieres compartir más." # Default response if no matching emotion found

      if possible_emotions:
          # Pick one emotion randomly from the list of possible emotions for this polarity
          # Ensure the chosen emotion is a key in the 'respuestas' dictionary before using it
          valid_possible_emotions = [emo for emo in possible_emotions if emo in respuestas]

          if valid_possible_emotions:
              chosen_emotion = random.choice(valid_possible_emotions)
              # Get a response for the chosen emotion using the original function
              respuesta = obtener_respuesta_emocional(chosen_emotion)
          # If possible_emotions list was not empty but none were valid keys in 'respuestas',
          # we keep the default response defined above.

      print(respuesta)

    except Exception as e:
      print(f"Ocurrió un error al procesar tu entrada: {e}")
      # Print the traceback to help diagnose the error if needed
Las respuestas que se generen son con fines de mejora y no deben ser consideradas como un diagnóstico o tratamiento profesional.-""")


# Function to generate text using Cohere based on the detected emotion
    prompt = f"El usuario expresó un sentimiento de {emotion} con el texto: \"{user_text}\". Genera una respuesta empática y comprensiva en español basada en este sentimiento."

    try:
        response = co.chat(
             model='command-a-03-2025',
             messages=[
                 {'role': 'system', 'content': f'Eres un chat únicamente entrenado para dar respuestas de ayuda psicológica, ninguna otra. Solo debes dar respuestas mejorativas, y no debes dar diagnósticos o tratamientos, ya que no eres un profesional. Responde de forma empática y comprensiva al usuario que expresó un sentimiento de {emotion} con el texto: "{user_text}".'},
                 {'role': 'user', 'content': user_text}
             ]
         )
        # Extract the text content from the response message
        return response.message.content[0].text.strip()

    except Exception as e:
        print(f"Error generating Cohere response: {e}")
        return "No pude generar una respuesta en este momento."

# Modify the main loop to use Cohere for generation
while True:
  user_input = input("¿Cómo te sientes hoy? (escribe 'salir' para terminar): ")
  if user_input.lower() in ["salir", "terminar", "adiós"]:
    print("¡Hasta luego!")
    break
  if user_input:
    try:
      predicted_polarity = predict(user_input)
      print(f"La polaridad predicha es: {predicted_polarity.capitalize()}")

      # Convert the predicted polarity to lowercase
      predicted_polarity_lower = predicted_polarity.lower()

      # Get the list of possible emotions for this polarity
      possible_emotions = polarity_to_possible_emotions.get(predicted_polarity_lower, [])

      respuesta = "Entiendo. Estoy aquí para escucharte si quieres compartir más." # Default response

      if possible_emotions:
          # Pick one emotion randomly from the list of possible emotions for this polarity
          valid_possible_emotions = [emo for emo in possible_emotions if emo in respuestas]

          if valid_possible_emotions:
              chosen_emotion = random.choice(valid_possible_emotions)
              print(f"(El chatbot interpretará esto como {chosen_emotion})") # Optional: show the interpreted emotion
              # Use Cohere to generate a response based on the chosen emotion and user text
              respuesta = generate_cohere_response(chosen_emotion, user_input)
      print(respuesta)

    except Exception as e:
      print(f"Ocurrió un error al procesar tu entrada: {e}")
      traceback.print_exc()
  else:
    print("Por favor, ingresa algo sobre cómo te sientes.")


# (que está entrenado en el conjunto de datos TASS) para determinar un conjunto potencial de emociones. A continuación, utiliza una emoción elegida al azar de ese conjunto, junto con la entrada original del usuario, como parte de la solicitud de la API Cohere para generar una respuesta más matizada y contextualmente relevante.
# 
# # El código actual ya demuestra una cadena simple:
# Entrada del Usuario -> Predicción RoBERTa (Polaridad) -> Asignar Polaridad a Posibles Emociones -> Elegir Emoción -> Mensaje Cohere -> Respuesta Cohere.
# 

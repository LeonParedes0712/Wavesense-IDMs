import json
import sys
import os
from openai import OpenAI
from tools import ALL_TOOLS, TOOL_MAP

client = OpenAI()

distraction_counter = 0

# Modelo configurado
MODEL_NAME = "gpt-4.1-mini"

def handle_eeg_state(state: str):
    global distraction_counter

    if state == "focused":
        distraction_counter = 0
        print(f"[LOG]: Estado 'focused'. Contador de distracción reiniciado.")
        return

    if state == "distracted":
        distraction_counter += 1
        print(f"\n[LOG]: Distracción #{distraction_counter} detectada consecutivamente.")

        system_prompt = (
            "Eres el tutor del sistema Wavesense. "
            f"El estudiante lleva {distraction_counter} distracciones consecutivas. "
            "1. Si lleva 1 o 2 distracciones, activa únicamente `trigger_distraction_alert`. "
            "2. Si alcanza 3 o más distracciones, activa `trigger_distraction_alert` Y TAMBIÉN `offer_academic_help`."
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Estado del estudiante: {state}. Contador actual: {distraction_counter}"}
            ],
            tools=ALL_TOOLS,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if message.tool_calls:
            frustration_prompt = None

            # PASO 1: Ejecutar primero todas las alertas visuales (TikTok)
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                if func_name == "trigger_distraction_alert":
                    result = TOOL_MAP[func_name]()
                    print(f"[ACCION]: {result}")
                elif func_name == "offer_academic_help":
                    # Guardamos la pregunta para hacerla después de abrir el TikTok
                    frustration_prompt = TOOL_MAP[func_name](**func_args) if func_args else TOOL_MAP[func_name]()

            # PASO 2: Si era la 3ª distracción, interactuar con el usuario DESPUÉS del TikTok
            if frustration_prompt:
                process_frustration_interaction(frustration_prompt)

def process_frustration_interaction(tutor_question: str):
    global distraction_counter

    print("\n" + "="*60)
    print(f"🤖 TUTOR WAVESENSE:\n{tutor_question}")
    
    # Detiene la ejecución aquí hasta que el usuario responda
    user_input = input("\n[Tú]: ").strip()

    if user_input.lower() in ["no", "ninguna", "no tengo dudas", "nada", ""]:
        print("\n🤖 TUTOR: ¡Entendido! Tómate un momento y seguimos cuando estés listo.\n")
    else:
        explain_doubt(user_input)

    distraction_counter = 0

def explain_doubt(user_response: str):
    print("\n🤖 TUTOR: Procesando tu mensaje...")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un tutor adaptativo. El estudiante se ha distraído 3 veces seguidas. "
                    "Identifica el tema y la duda de su mensaje y explícaselo usando una "
                    "analogía cotidiana sencilla en máximo 2 párrafos."
                )
            },
            {"role": "user", "content": user_response}
        ]
    )

    explanation = response.choices[0].message.content
    print(f"\n💡 EXPLICACIÓN DEL TUTOR:\n{explanation}\n")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Redirigir errores de GTK del sistema
    sys.stderr = open(os.devnull, 'w')

    # Para probar el flujo progresivo real, en la simulación llamaremos la función evento por evento:
    print("--- SIMULACIÓN PROGRESIVA DE EEG ---")
    
    # Evento 1
    handle_eeg_state("distracted")
    
    # Evento 2
    handle_eeg_state("distracted")
    
    # Evento 3 (Abrirá el 3er TikTok e INMEDIATAMENTE pedirá la duda)
    handle_eeg_state("distracted")
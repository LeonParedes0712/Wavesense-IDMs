import numpy as np


def to_wavesense_probabilities(
    model_output: dict, signal_has_artifact: bool = False
) -> dict[str, float]:
    """
    Convierte la salida del modelo local (diccionario de etiquetas o probabilidades)
    a las 4 clases estandarizadas requeridas por Wavesense:
    REST, LOW_LOAD, HIGH_LOAD, ARTIFACT.
    """
    # 1. Si el control de calidad detecta ruido severo, se reporta ARTIFACT directo
    if signal_has_artifact:
        return {"REST": 0.0, "LOW_LOAD": 0.0, "HIGH_LOAD": 0.0, "ARTIFACT": 1.0}

    # 2. Extracción flexible de scores tolerando variaciones de nombres
    rest_score = float(
        model_output.get("REST", model_output.get("DEMOSTRACION_LUZMA", 0.0))
    )

    # SUBWAY_SURFERS y BRAWL_STARS corresponden a LOW_LOAD
    subway_score = float(model_output.get("SUBWAY_SURFERS", 0.0))
    brawl_score = float(model_output.get("BRAWL_STARS", 0.0))
    low_load_score = max(subway_score, brawl_score)

    # MATH_LOAD corresponde a HIGH_LOAD
    high_load_score = float(model_output.get("MATH_LOAD", 0.0))

    # DIBUJANDO o etiqueta explícita de movimiento
    artifact_score = float(
        model_output.get("DIBUJANDO", model_output.get("ARTIFACT", 0.0))
    )

    # 3. Normalización Softmax
    raw_scores = np.array(
        [rest_score, low_load_score, high_load_score, artifact_score],
        dtype=np.float64,
    )
    exp_scores = np.exp(raw_scores - np.max(raw_scores))
    probs = exp_scores / np.sum(exp_scores)

    # 4. Formateo a 4 decimales
    keys = ["REST", "LOW_LOAD", "HIGH_LOAD", "ARTIFACT"]
    wavesense_dict = {
        key: float(np.round(p, 4)) for key, p in zip(keys, probs)
    }

    # Ajuste decimal fino para sumar exactamente 1.0
    diff = round(1.0 - sum(wavesense_dict.values()), 4)
    wavesense_dict["HIGH_LOAD"] = round(wavesense_dict["HIGH_LOAD"] + diff, 4)

    return wavesense_dict


if __name__ == "__main__":
    # --- PRUEBA DE EJECUCIÓN ---
    ejemplo_entrada = {
        "DEMOSTRACION_LUZMA": 0.10,
        "SUBWAY_SURFERS": 0.45,
        "BRAWL_STARS": 0.30,
        "MATH_LOAD": 3.60,
        "DIBUJANDO": -0.75,
    }

    resultado = to_wavesense_probabilities(ejemplo_entrada)

    print("\n✅ Adaptador ejecutado con éxito:")
    print("-" * 40)
    for estado, prob in resultado.items():
        print(f"  {estado:10s} : {prob:.4f}")
    print("-" * 40)
    print(f"  Suma total : {sum(resultado.values()):.4f}\n")
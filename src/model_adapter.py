from typing import Mapping


def to_wavesense_probabilities(
    model_output: Mapping[str, float],
    signal_has_artifact: bool = False,
) -> dict[str, float]:
    """
    Convierte probabilidades del modelo EEG a estados Wavesense.
    Suma directamente las probabilidades de las clases sin aplicar Softmax.
    """
    if signal_has_artifact:
        return {
            "REST": 0.0,
            "LOW_LOAD": 0.0,
            "HIGH_LOAD": 0.0,
            "ARTIFACT": 1.0,
        }

    # Soporte para 'REST' y alias de origen 'DEMOSTRACION_LUZMA'
    rest = float(
        model_output.get("REST", model_output.get("DEMOSTRACION_LUZMA", 0.0))
    )
    low_load = float(model_output.get("DIBUJANDO", 0.0))
    high_load = (
        float(model_output.get("MATH_LOAD", 0.0))
        + float(model_output.get("BRAWL_STARS", 0.0))
        + float(model_output.get("SUBWAY_SURFERS", 0.0))
    )

    total = rest + low_load + high_load
    if total <= 0:
        raise ValueError("No se recibieron probabilidades válidas del modelo EEG.")

    return {
        "REST": round(rest / total, 4),
        "LOW_LOAD": round(low_load / total, 4),
        "HIGH_LOAD": round(high_load / total, 4),
        "ARTIFACT": 0.0,
    }


if __name__ == "__main__":
    ejemplo = {
        "REST": 0.05,
        "DIBUJANDO": 0.05,
        "MATH_LOAD": 0.70,
        "BRAWL_STARS": 0.10,
        "SUBWAY_SURFERS": 0.10,
    }
    print("Probabilidades procesadas:", to_wavesense_probabilities(ejemplo))
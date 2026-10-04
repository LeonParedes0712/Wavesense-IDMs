import os
import sys

UNICORN_PATH = r"C:\Users\alerm\Documents\gtec\Unicorn Suite\Hybrid Black\Unicorn Python\Lib"

# Permitir que Windows encuentre las DLL de Unicorn
os.add_dll_directory(UNICORN_PATH)

# Permitir que Python encuentre UnicornPy
sys.path.insert(0, UNICORN_PATH)

import UnicornPy
import numpy as np
import csv


def main():

    # ---------------------------------------------------------
    # CONFIGURACIÓN
    # ---------------------------------------------------------
    TestsignalEnabled = False
    FrameLength = 1
    AcquisitionDurationInSeconds = 25
    DataFile = "datos_subway.csv"

    print("Unicorn EEG - Adquisición CSV")
    print("-----------------------------")

    try:

        # ---------------------------------------------------------
        # BUSCAR DISPOSITIVOS
        # ---------------------------------------------------------
        deviceList = UnicornPy.GetAvailableDevices(True)

        if deviceList is None or len(deviceList) == 0:
            raise Exception("No se encontró ningún Unicorn. Verifica que esté emparejado.")

        print("\nDispositivos disponibles:")

        for i, deviceName in enumerate(deviceList):
            print(f"#{i}: {deviceName}")

        # Seleccionar dispositivo
        deviceID = int(input("\nSelecciona dispositivo por ID: "))

        if deviceID < 0 or deviceID >= len(deviceList):
            raise IndexError("ID de dispositivo no válido.")

        # ---------------------------------------------------------
        # CONECTAR
        # ---------------------------------------------------------
        print(f"\nConectando a {deviceList[deviceID]}...")

        device = UnicornPy.Unicorn(deviceList[deviceID])

        print("Conectado.")

        # ---------------------------------------------------------
        # INFORMACIÓN DEL DISPOSITIVO
        # ---------------------------------------------------------
        numberOfAcquiredChannels = device.GetNumberOfAcquiredChannels()

        samplingRate = UnicornPy.SamplingRate

        print("\nConfiguración:")
        print(f"Frecuencia de muestreo: {samplingRate} Hz")
        print(f"Número de canales: {numberOfAcquiredChannels}")
        print(f"Duración: {AcquisitionDurationInSeconds} segundos")

        # ---------------------------------------------------------
        # BUFFER
        # Cada dato es float32 = 4 bytes
        # ---------------------------------------------------------
        receiveBufferBufferLength = (
            FrameLength *
            numberOfAcquiredChannels *
            4
        )

        receiveBuffer = bytearray(receiveBufferBufferLength)

        # ---------------------------------------------------------
        # CREAR CSV
        # ---------------------------------------------------------
        with open(DataFile, "w", newline="") as csvFile:

            writer = csv.writer(csvFile)

            # Encabezados
            header = ["Tiempo_s"]

            for channel in range(numberOfAcquiredChannels):
                header.append(f"Canal_{channel + 1}")

            writer.writerow(header)

            # -----------------------------------------------------
            # INICIAR ADQUISICIÓN
            # -----------------------------------------------------
            device.StartAcquisition(TestsignalEnabled)

            print("\nAdquisición iniciada...\n")

            numberOfGetDataCalls = int(
                AcquisitionDurationInSeconds *
                samplingRate /
                FrameLength
            )

            sampleNumber = 0

            # -----------------------------------------------------
            # LOOP DE ADQUISICIÓN
            # -----------------------------------------------------
            for i in range(numberOfGetDataCalls):

                device.GetData(
                    FrameLength,
                    receiveBuffer,
                    receiveBufferBufferLength
                )

                # Convertir bytes -> float32
                data = np.frombuffer(
                    receiveBuffer,
                    dtype=np.float32
                )

                # Si FrameLength > 1
                data = data.reshape(
                    FrameLength,
                    numberOfAcquiredChannels
                )

                # Guardar cada muestra
                for frame in data:

                    timeSeconds = sampleNumber / samplingRate

                    row = [timeSeconds] + frame.tolist()

                    writer.writerow(row)

                    sampleNumber += 1

                # Mostrar progreso
                if i % int(samplingRate) == 0:
                    print(
                        f"Tiempo: {sampleNumber / samplingRate:.1f} s"
                    )

            # -----------------------------------------------------
            # DETENER
            # -----------------------------------------------------
            device.StopAcquisition()

            print("\nAdquisición terminada.")

        print(f"\nDatos guardados en:")
        print(DataFile)

        del receiveBuffer
        del device

        print("\nUnicorn desconectado.")

    except UnicornPy.DeviceException as e:
        print(f"Error del Unicorn: {e}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
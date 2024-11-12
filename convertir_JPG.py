import os
from PIL import Image

# Define la ruta de la carpeta principal
carpeta_principal = './data_numeros/Numeros'

# Recorre las subcarpetas
for i in range(10):
    subcarpeta = os.path.join(carpeta_principal, str(i))
    
    # Verifica si la subcarpeta existe
    if os.path.isdir(subcarpeta):
        for archivo in os.listdir(subcarpeta):
            if archivo.lower().endswith('.jpeg'):
                # Crea la ruta completa del archivo
                ruta_archivo = os.path.join(subcarpeta, archivo)
                # Abre la imagen
                with Image.open(ruta_archivo) as img:
                    # Cambia el nombre del archivo a .jpg
                    nuevo_nombre = archivo[:-5] + '.jpg'
                    nueva_ruta = os.path.join(subcarpeta, nuevo_nombre)
                    # Guarda la imagen como JPG
                    img.save(nueva_ruta, 'JPEG')
                # Elimina el archivo original si es necesario
                os.remove(ruta_archivo)

print("Conversión completada.")

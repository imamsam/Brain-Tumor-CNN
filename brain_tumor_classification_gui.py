import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageOps, ImageTk
import numpy as np
import tensorflow as tf

# Path ke model yang disimpan dalam format .keras
# ganti sesuai direktori file model
model_path = r'C:\Users\Imam Sam\Downloads\IPYNB TUMOR OTAK\Brain Tumor MRI Dataset\saved_model40.keras'
model = tf.keras.models.load_model(model_path)

# Fungsi untuk memproses gambar
def process_image(image_path):
    try:
        print(f"Opening image from path: {image_path}")
        image = Image.open(image_path).convert('L')  # Konversi gambar ke grayscale
        print("Image opened and converted to grayscale.")
        image = ImageOps.fit(image, (168, 168), Image.LANCZOS)
        print("Image resized to 168x168.")
        image = np.array(image)
        print(f"Image converted to numpy array with shape: {image.shape}")
        image = image.reshape(1, 168, 168, 1)  # Reshape gambar sesuai input model
        print(f"Image reshaped to: {image.shape}")
        image = image / 255.0  # Normalisasi
        print("Image normalized.")
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Fungsi untuk membuat prediksi
def predict(image_path):
    image = process_image(image_path)
    if image is None:
        return "Error processing image."
    try:
        print("Predicting image.")
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)
        class_names = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']  # nama kelas
        print(f"Prediction successful. Predicted class: {class_names[predicted_class[0]]}")
        return class_names[predicted_class[0]]
    except Exception as e:
        print(f"Error predicting image: {e}")
        return "Error predicting image."

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            image = Image.open(file_path).convert('L')
            image.thumbnail((200, 200))
            img = ImageTk.PhotoImage(image)
            panel.config(image=img)
            panel.image = img
            prediction = predict(file_path)
            prediction_label.config(text=f"Predicted class: {prediction}")
        except Exception as e:
            print(f"Error uploading image: {e}")

# Membuat GUI utama
root = tk.Tk()
root.title("Brain Tumor Classification")

# Menambahkan judul
title = tk.Label(root, text="CNN untuk Klasifikasi Multi Kelas Tumor Otak Menggunakan Citra MRI", font=("Helvetica", 16))
title.pack(pady=10)

# Menambahkan teks tambahan
subtitle = tk.Label(root, text="Imam Samsudin - imamsamsudin@mail.ugm.ac.id", font=("Helvetica", 12))
subtitle.pack(pady=5)

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

# Label untuk menampilkan gambar yang diunggah
panel = tk.Label(root)
panel.pack(pady=20)

# Label untuk menampilkan hasil prediksi
prediction_label = tk.Label(root, text="Predicted class: ", font=("Helvetica", 12))
prediction_label.pack(pady=20)

root.mainloop()

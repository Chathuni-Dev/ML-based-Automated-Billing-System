import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk, ImageGrab
import serial
import csv
import time
from datetime import datetime
import os

# --- Configurations ---
MODEL_PATH = "model.tflite"
LABELS_PATH = "labels.txt"
PRICE_FILE = "price.csv"
SERIAL_PORT = "COM4"

# --- Load model ---
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Load labels ---
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# --- Load price list ---
def load_price_map(file=PRICE_FILE):
    price_map = {}
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            price_map[row['item']] = float(row['price_per_kg'])
    return price_map

price_map = load_price_map()

# --- Arduino weight ---
def get_weight_from_arduino(port=SERIAL_PORT, baudrate=9600):
    try:
        ser = serial.Serial(port, baudrate)
        time.sleep(2)
        weight = 0
        for _ in range(5):
            line = ser.readline().decode().strip()
            weight += float(line)
        ser.close()
        return max(weight / 5, 0)
    except Exception as e:
        print("Error reading from Arduino:", e)
        return 0

# --- ML classification ---
def classify_image(image_np):
    img = cv2.resize(image_np, (224, 224))
    input_data = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = np.argmax(output)
    return labels[idx], output[idx]

# --- GUI setup ---
window = tk.Tk()
window.title("Automated Billing System")
window.geometry("850x750")

video_label = Label(window, width=360, height=270)
video_label.pack()

bill_label = Label(window, text="", font=("Consolas", 12), justify="left")
bill_label.pack(pady=10)

cap = cv2.VideoCapture(0)

# --- State holders ---
current_item = None
current_weight = None
bill_ready = False

# --- Webcam feed loop ---
def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (360, 270))  # Resize webcam feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    video_label.after(10, update_frame)

# --- Capture & classify item ---
def capture_item():
    global current_item, bill_ready
    ret, frame = cap.read()
    if not ret:
        bill_label.config(text="Error: Camera capture failed.")
        return
    item, confidence = classify_image(frame)
    current_item = item
    bill_ready = True
    generate_bill()

# --- Read weight ---
def read_weight():
    global current_weight, bill_ready
    current_weight = get_weight_from_arduino() / 1000  # grams → kg
    bill_ready = True
    generate_bill()

# --- Generate bill ---
def generate_bill():
    if current_item and current_weight is not None:
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        filename_stamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        price_per_kg = price_map.get(current_item, 0)
        total = price_per_kg * current_weight

        bill_text = f"""
AUTOMATED BILLING SYSTEM
Date & Time : {timestamp}
-------------------------------
Item        : {current_item}
Weight (kg) : {current_weight:.3f}
Price / kg  : Rs. {price_per_kg:.2f}
-------------------------------
Total Price : Rs. {total:.2f}
-------------------------------
Press RESET to begin a new scan.
"""
        bill_label.config(text=bill_text)
        window.update()

        save_bill_as_image(filename_stamp)
        save_to_csv(timestamp, current_item, current_weight, price_per_kg, total)
        show_buttons_for("reset_only")

# --- Save full window screenshot ---
def save_bill_as_image(filename_stamp):
    folder = "saved_bills"
    os.makedirs(folder, exist_ok=True)

    window.update_idletasks()
    time.sleep(0.3) 

    x = window.winfo_rootx()
    y = window.winfo_rooty()
    w = x + 1400
    h = y + 1000

    image = ImageGrab.grab(bbox=(x, y, w, h))
    image.save(os.path.join(folder, f"bill_{filename_stamp}.png"))
    print(f"Full bill screenshot saved to: {folder}/bill_{filename_stamp}.png")

def save_to_csv(timestamp, item, weight, price_per_kg, total):
    file_path = "billing_history.csv"
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["DateTime", "Item", "Weight (kg)", "Price/kg", "Total"])
        writer.writerow([timestamp, item, f"{weight:.3f}", f"{price_per_kg:.2f}", f"{total:.2f}"])
    print(f"📄 Bill saved to CSV: {file_path}")

def reset_all():
    global current_item, current_weight, bill_ready
    current_item = None
    current_weight = None
    bill_ready = False
    bill_label.config(text="System reset. Ready for new item.")
    show_buttons_for("initial")

def show_buttons_for(stage):
    if stage == "initial":
        btn_capture.grid()
        btn_weight.grid()
        btn_reset.grid()
    elif stage == "reset_only":
        btn_capture.grid_remove()
        btn_weight.grid_remove()
        btn_reset.grid()

btn_frame = tk.Frame(window)
btn_frame.pack(pady=10)

btn_capture = Button(btn_frame, text="Capture Item", command=capture_item,
       font=("Arial", 14), bg="green", fg="white", width=20)
btn_capture.grid(row=0, column=0, padx=5, pady=5)

btn_weight = Button(btn_frame, text="Read Weight", command=read_weight,
       font=("Arial", 14), bg="blue", fg="white", width=20)
btn_weight.grid(row=0, column=1, padx=5, pady=5)

btn_reset = Button(btn_frame, text="Reset", command=reset_all,
       font=("Arial", 14), bg="gray", fg="white", width=42)
btn_reset.grid(row=1, column=0, columnspan=2, pady=10)

show_buttons_for("initial")
update_frame()
window.mainloop()
cap.release()
cv2.destroyAllWindows()

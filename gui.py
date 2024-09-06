import tkinter as tk
from tkinter import scrolledtext, StringVar, OptionMenu
from threading import Thread
from audio_processor import process_live_audio
from speech_recognition import process_text_input
from gesture_recognition import process_continuous_gesture_input

def start_gui():
    # Main window
    window = tk.Tk()
    window.title("AI-Powered Multimodal Assistant")

    # Dropdown for selecting input method
    input_method_var = StringVar(window)
    input_method_var.set("Text Input")  # Default value
    tk.Label(window, text="Select Input Method:").grid(row=0, column=0)
    input_methods = ["Text Input", "Voice Input", "Gesture Recognition"]
    input_menu = OptionMenu(window, input_method_var, *input_methods)
    input_menu.grid(row=0, column=1)

    # Text input and output
    text_input = tk.Entry(window, width=50)
    text_input.grid(row=1, column=0, columnspan=2)

    output_text = scrolledtext.ScrolledText(window, width=50, height=10)
    output_text.grid(row=2, column=0, columnspan=2)

    def on_submit():
        selected_input = input_method_var.get()
        if selected_input == "Text Input":
            user_input = text_input.get()
            process_text_input(user_input, output_text)
        elif selected_input == "Voice Input":
            Thread(target=process_live_audio, args=(output_text,), daemon=True).start()
        elif selected_input == "Gesture Recognition":
            Thread(target=process_continuous_gesture_input, args=(output_text,), daemon=True).start()

    submit_button = tk.Button(window, text="Submit", command=on_submit)
    submit_button.grid(row=1, column=2)

    # Main loop
    window.mainloop()

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import joblib
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from Scripts.customTools import Image as CustomImage
from Scripts.customTools import Perceptron

class FaceGenderClassifierApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Face Gender Classifier")
        self.root.geometry("1200x800")
    
        self.original_image = None
        self.current_image = None
        self.rect_start = None
        self.rect_end = None
        self.drawing = False
        self.face_image = None
        self.preview_photo = None

        self.default_color = "#1f538d" 
        self.active_color = "#2ecc71"   
        self.hover_color = "#1a4674"   
        self.active_model = None

        self.load_models()
   
        self.create_gui()

    def load_models(self):
        try:
            self.svm_model = joblib.load('Models/best_svm_model.pkl')
            self.svm_scaler = joblib.load('Models/svm_scaler.pkl')
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device for Perceptron: {self.device}")
            
            checkpoint = torch.load(
                'Models/best_perceptron_model.pth',
                map_location=self.device,
                weights_only=False  
            )
            self.perceptron_model = Perceptron(input_size=checkpoint['input_size'])
            self.perceptron_model.load_state_dict(checkpoint['model_state_dict'])
            self.perceptron_scaler = joblib.load('Models/perceptron_scaler.pkl')
            self.perceptron_model.to(self.device)
            self.perceptron_model.eval()
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")

    def create_gui(self):
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.left_frame = ctk.CTkFrame(self.main_frame)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.left_frame, bg="gray")
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.canvas.bind("<ButtonPress-1>", self.start_rect)
        self.canvas.bind("<B1-Motion>", self.draw_rect)
        self.canvas.bind("<ButtonRelease-1>", self.end_rect)
        
        self.right_frame = ctk.CTkFrame(self.main_frame)
        self.right_frame.pack(side="right", fill="y", padx=5, pady=5)
        
        self.preview_frame = ctk.CTkFrame(self.right_frame)
        self.preview_frame.pack(pady=10, padx=5)
        
        self.preview_label = ctk.CTkLabel(
            self.preview_frame,
            text="Selected Area Preview",
            font=("Arial", 12)
        )
        self.preview_label.pack(pady=5)
        
        self.preview_canvas = tk.Canvas(
            self.preview_frame,
            width=150,
            height=150,
            bg="gray"
        )
        self.preview_canvas.pack(pady=5)
        
        self.model_info_frame = ctk.CTkFrame(self.right_frame)
        self.model_info_frame.pack(pady=5)
        
        self.model_info_label = ctk.CTkLabel(
            self.model_info_frame,
            text="Model: None",
            font=("Arial", 12)
        )
        self.model_info_label.pack(pady=5)
        
        self.load_button = ctk.CTkButton(
            self.right_frame, 
            text="Load Image", 
            command=self.load_image,
            fg_color=self.default_color,
            hover_color=self.hover_color
        )
        self.load_button.pack(pady=10)
        
        self.clear_selection_button = ctk.CTkButton(
            self.right_frame, 
            text="Clear Selection", 
            command=self.clear_selection,
            fg_color=self.default_color,
            hover_color=self.hover_color
        )
        self.clear_selection_button.pack(pady=10)
        
        self.predict_svm_button = ctk.CTkButton(
            self.right_frame, 
            text="Predict (SVM)", 
            command=lambda: self.predict('svm'),
            fg_color=self.default_color,
            hover_color=self.hover_color
        )
        self.predict_svm_button.pack(pady=10)
        
        self.predict_perceptron_button = ctk.CTkButton(
            self.right_frame, 
            text="Predict (Perceptron)", 
            command=lambda: self.predict('perceptron'),
            fg_color=self.default_color,
            hover_color=self.hover_color
        )
        self.predict_perceptron_button.pack(pady=10)

        self.result_frame = ctk.CTkFrame(
            self.right_frame,
            corner_radius=15,
            fg_color=("gray90", "gray25"),
            border_width=2,
            border_color= "white"
        )
        self.result_frame.pack(pady=20, padx=10, expand=True, fill="x")

        self.result_title = ctk.CTkLabel(
            self.result_frame,
            text="Result",
            font=("Arial", 12, "bold"),
            text_color=("gray50", "gray70")
        )
        self.result_title.pack(pady=(10, 0))

        self.result_label = ctk.CTkLabel(
            self.result_frame, 
            text="None",
            font=("Arial", 20, "bold"),
            padx=30,
            pady=15
        )
        self.result_label.pack(pady=(5, 15))
        
        self.predict_svm_button.bind('<Enter>', lambda e: self.show_tooltip(
            "Support Vector Machine classifier"
        ))
        self.predict_svm_button.bind('<Leave>', lambda e: self.hide_tooltip())
        
        self.predict_perceptron_button.bind('<Enter>', lambda e: self.show_tooltip(
            "Neural Network Perceptron classifier"
        ))
        self.predict_perceptron_button.bind('<Leave>', lambda e: self.hide_tooltip())

    def show_tooltip(self, text):
        x = self.root.winfo_pointerx() + 10
        y = self.root.winfo_pointery() + 10
        
        self.tooltip = tk.Toplevel()
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.geometry(f"+{x}+{y}")
        
        label = tk.Label(self.tooltip, text=text, bg="yellow", relief="solid", borderwidth=1)
        label.pack()

    def hide_tooltip(self):
        if hasattr(self, 'tooltip'):
            self.tooltip.destroy()

    def reset_button_colors(self):
        self.predict_svm_button.configure(fg_color=self.default_color)
        self.predict_perceptron_button.configure(fg_color=self.default_color)

    def update_result(self, text, model_type):
        self.result_label.configure(text=text)
        if "Man" in text:
            self.result_frame.configure(
                border_color="#3498db",  
                fg_color=("#e8f4f8", "#1a2530")  
            )
        elif "Woman" in text:
            self.result_frame.configure(
                border_color="pink", 
                fg_color=("#fce8e6", "#301a1a")  
            )
        else:
            self.result_frame.configure(
                border_color=self.default_color,
                fg_color=("gray90", "gray25")
            )
        self.result_title.configure(
            text=f"Gender"
        )

    def predict(self, model_type='svm'):
        if self.face_image is None:
            self.update_result("Please select a face first!", "")
            return
                    
        try:
            self.reset_button_colors()
            
            if model_type == 'svm':
                self.predict_svm_button.configure(fg_color=self.active_color)
                self.active_model = 'svm'
            else:
                self.predict_perceptron_button.configure(fg_color=self.active_color)
                self.active_model = 'perceptron'
            
            face_array = np.array(self.face_image)
            face_processor = CustomImage(face_array)
            features = face_processor.featureVector.reshape(1, -1)
            
            if model_type == 'svm':
                features_scaled = self.svm_scaler.transform(features)
                prediction = self.svm_model.predict(features_scaled)[0]
            else:
                features_scaled = self.perceptron_scaler.transform(features)
                features_tensor = torch.FloatTensor(features_scaled).to(self.device)
                with torch.no_grad():
                    prediction = self.perceptron_model(features_tensor)
                    prediction = (prediction >= 0.5).float().cpu().numpy()[0]
            
            result_text = "Man" if prediction == 0 else "Woman"
            self.update_result(result_text, model_type)
            
            self.model_info_label.configure(
                text=f"{model_type.upper()}"
            )
                
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            self.update_result(f"Error in prediction: {str(e)}", "")

    def clear_selection(self):
        self.canvas.delete("rect")
        self.face_image = None
        self.rect_start = None
        self.rect_end = None
        self.update_result("None", "")
        self.preview_canvas.delete("all")
        self.preview_label.configure(text="Selected Area Preview")
        self.reset_button_colors()
        self.active_model = None
        self.model_info_label.configure(text="Active Model: None")

    def draw_rect(self, event):
        if self.drawing:
            self.canvas.delete("rect")
            rect_coords = (
                self.rect_start[0], 
                self.rect_start[1], 
                event.x, 
                event.y
            )
            self.canvas.create_rectangle(
                *rect_coords,
                outline="yellow",
                width=2,
                tags="rect"
            )
            
            width = abs(event.x - self.rect_start[0])
            height = abs(event.y - self.rect_start[1])
            self.canvas.create_text(
                min(event.x, self.rect_start[0]) + width/2,
                min(event.y, self.rect_start[1]) - 10,
                text=f"{width}x{height}",
                fill="yellow",
                tags="rect"
            )
    
    def end_rect(self, event):
        self.drawing = False
        self.rect_end = (event.x, event.y)
        
        rect_coords = (
            self.rect_start[0], 
            self.rect_start[1], 
            event.x, 
            event.y
        )
        self.canvas.delete("rect")
        self.canvas.create_rectangle(
            *rect_coords,
            outline="green",
            width=2,
            tags="rect"
        )
        
        width = abs(event.x - self.rect_start[0])
        height = abs(event.y - self.rect_start[1])
        self.canvas.create_text(
            min(event.x, self.rect_start[0]) + width/2,
            min(event.y, self.rect_start[1]) - 10,
            text=f"{width}x{height}",
            fill="green",
            tags="rect"
        )
        
        self.crop_face()

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.show_image(self.original_image)
            self.clear_selection()
    
    def show_image(self, image):
        self.original_image = image
        
        height, width = image.shape[:2]
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        ratio = min(canvas_width/width, canvas_height/height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        resized = cv2.resize(image, (new_width, new_height))
        self.current_image = Image.fromarray(resized)
        self.photo = ImageTk.PhotoImage(self.current_image)
        
        self.image_info = {
            'width': new_width,
            'height': new_height,
            'offset_x': (canvas_width - new_width) // 2,
            'offset_y': (canvas_height - new_height) // 2,
            'ratio': ratio
        }
        
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width//2, 
            canvas_height//2, 
            image=self.photo, 
            anchor="center"
        )

    def update_preview(self):
        if self.face_image:
            preview_size = (150, 150)
            preview_image = self.face_image.copy()
            preview_image.thumbnail(preview_size, Image.Resampling.LANCZOS)
            
            self.preview_photo = ImageTk.PhotoImage(preview_image)
            
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(
                75, 75,
                image=self.preview_photo,
                anchor="center"
            )
            
            original_size = self.face_image.size
            self.preview_label.configure(
                text=f"Selected Area: {original_size[0]}x{original_size[1]} px"
            )

    def crop_face(self):
        if self.current_image and self.rect_start and self.rect_end:
            try:
                x1, y1 = min(self.rect_start[0], self.rect_end[0]), min(self.rect_start[1], self.rect_end[1])
                x2, y2 = max(self.rect_start[0], self.rect_end[0]), max(self.rect_start[1], self.rect_end[1])
                
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                img_width, img_height = self.current_image.size

                offset_x = (canvas_width - img_width) // 2
                offset_y = (canvas_height - img_height) // 2

                x1 = max(0, min(x1 - offset_x, img_width))
                y1 = max(0, min(y1 - offset_y, img_height))
                x2 = max(0, min(x2 - offset_x, img_width))
                y2 = max(0, min(y2 - offset_y, img_height))
                
                if x2 > x1 and y2 > y1:
                    self.face_image = self.current_image.crop((x1, y1, x2, y2))
                    self.update_preview()
                else:
                    print("Invalid selection area")
                    self.face_image = None
                    
            except Exception as e:
                print(f"Error in crop_face: {str(e)}")
                self.face_image = None

    def start_rect(self, event):
        self.rect_start = (event.x, event.y)
        self.drawing = True

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = FaceGenderClassifierApp()
    app.run()
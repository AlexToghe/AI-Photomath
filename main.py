import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow
from GUI_Slider import Ui_MainWindow
from PyQt5.QtCore import QTimer
import cv2
from PyQt5.QtGui import QImage, QPixmap
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import Net

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Initialize camera
        self.cam = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update frame every 30 milliseconds
        
        # Connect buttons to functions
        self.captureButton.clicked.connect(self.on_capture_button_clicked)
        self.clearButton.clicked.connect(self.clear_button_clicked)
        
        # Connect slider valueChanged signal to update_threshold_value slot
        self.verticalSlider.valueChanged.connect(self.update_threshold_value)
        # Set initial threshold value
        self.threshold_value = 127
        
        # Flag to track if camera feed is paused
        self.paused = False
        
        # Load PyTorch model
        self.network = Net()
        self.network.load_state_dict(torch.load("C:/Users/danie/OneDrive/WayneWinter2024/ECE4600/FullWithSlider/Models/07_pytorch_wrapper/model.pt"))
        self.network.eval() # set to evaluation mode

    def update_frame(self):
        ret, frame = self.cam.read()
        if ret and not self.paused:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            
            # Calculate new width based on available space in the GUI
            available_width = self.cameraFeed.width()
            new_width = available_width
            
            # Calculate new height maintaining aspect ratio
            new_height = int(h * (new_width / w))
            
            # Resize frame
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
            
            bytes_per_line = ch * new_width
            q_img = QImage(frame_resized.data, new_width, new_height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.cameraFeed.setPixmap(pixmap)
        else:
            pass
        
    def pause_camera_feed(self):
        self.paused = True
        
    def resume_camera_feed(self):
        self.paused = False

    def on_capture_button_clicked(self):
        print("Capture pressed") # verification that it works
        # self.equationOutput.setText("")
        # self.solutionOutput.setText(" ")
        self.pause_camera_feed()  # Pause camera feed
        
        # Save frame to a JPG file
        ret, frame = self.cam.read()
        if ret:
            cv2.imwrite("captured_frame.jpg", frame)
            print("Frame saved as captured_frame.jpg")
            # Perform image processing and inference
            self.process_and_infer_image("captured_frame.jpg")
        else:
            pass
        
    def update_threshold_value(self, value):
        self.threshold_value = value

    def process_and_infer_image(self, image_path):
        # Read the image
        image = cv2.imread(image_path, 0)  # Read as grayscale

        # Apply Gaussian Blur
        gblur_im = cv2.GaussianBlur(image, (11, 11), 0)

        # Thresholding
        ret1, thresh1 = cv2.threshold(gblur_im, self.threshold_value, 255, cv2.THRESH_BINARY)    # Has issues creating the rectangles when inverting at this step, so invert later

        # Find contours
        contours, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Drawing rectangles around contours
        padding = 10
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x - padding, y - padding), (x + w + padding, y + h + padding), (0, 255, 0), 3)

        # Display processed image with contours and rectangles
        plt.imshow(image, cmap='gray')
        plt.title("Image with contours and rectangles")
        plt.show()

        # Saving cropped characters
        i = 0
        
        contours_sorted = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])        # Sort the rectangles from left to right
        
        for cnt in contours_sorted:
            x, y, w, h = cv2.boundingRect(cnt)
            if (w > 5 and h > 70) or (w > 100 and h < 20):
                if (i == 0):
                    cv2.imwrite(str(i) + ".jpg", thresh1[y:y + h, x:x + w])
                else:
                    cv2.imwrite(str(i) + ".jpg", thresh1[(y - padding):(y + h + padding), (x - padding):(x + w + padding)])
                i = i + 1

        # Create a blank image to draw contours
        contour_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw contours on a blank image
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

        # Display the image with contours
        plt.imshow(contour_img)
        plt.title("Image with contours")
        plt.show()
        
        solution = []
        # Perfrom prediction from model
        for i, filename in enumerate(sorted(os.listdir())):
            if filename.endswith(".jpg") and filename != "captured_frame.jpg" and filename != "0.jpg":
                digit_resized = cv2.imread(filename, 0)  # Read as grayscale
                digit_resized = 255 - digit_resized         # Invert to white on black
                digit_resized = cv2.resize(digit_resized, (28, 28))
                digit_tensor = torch.from_numpy(digit_resized).unsqueeze(0).unsqueeze(0).float() / 255.0
                output = self.network(digit_tensor)
                pred = output.argmax(dim=1, keepdim=True)
                prediction = str(pred.item())
                print(f"Prediction for digit {i}: {prediction}")
                solution.append(prediction)
                # self.solutionOutput.append(prediction)
        print(f"Solution: {solution}")
        sol_oneNum = ''.join(solution)
        self.equationOutput.setText(f"{sol_oneNum}")

    def clear_button_clicked(self):
        print("Cleared")
        self.equationOutput.setText("") # Clearing Text box
        #self.solutionOutput.setText("") # Clearing Text box
        self.resume_camera_feed()  # Resume camera feed
        
        # Delete the .jpgs to avoid loading old numbers (if the previous number had more digits)
        for filename in os.listdir():
            if filename.endswith(".jpg"):
                try:
                    os.remove(filename)
                    print(f"Deleted file: {filename}")
                except Exception as e:
                    print(f"Error deleting file {filename}: {e}")
       
                
    def closeEvent(self, event):
        # Override closeEvent to release camera when the window is closed
        self.cam.release()
        event.accept()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())

import tkinter as tk
import torch
import numpy as np
from model import RawCNN

digit_model = RawCNN(10)
digit_model.load_state_dict(torch.load("./saved_model.pt"))
digit_model.eval()

img_dims = (28, 28)
dim_multiplier = 15
canvas_width = img_dims[0]*dim_multiplier
canvas_height = img_dims[1]*dim_multiplier

class App(tk.Tk):
    def __init__(self):
        self.win = tk.Tk.__init__(self)
        self.previous_x = self.previous_y = 0
        self.x = self.y = 0
        self.points_recorded = []
        self.canvas = tk.Canvas(self, width=canvas_width, height=canvas_height, bg = "black", cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)
        # self.button_print = tk.Button(self, text = "Display points", command = self.print_points)
        # self.button_print.pack(side="top", fill="both", expand=True)
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        self.button_clear.pack(side="top", fill="both", expand=True)
        self.button_clear = tk.Button(self, text = "Prediction", command = self.prediction)
        self.button_clear.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<Motion>", self.tell_me_where_you_are)
        self.canvas.bind("<B1-Motion>", self.draw_from_where_you_are)

    def clear_all(self):
        self.canvas.delete("all")
        self.points_recorded[:] = []

    def print_points(self):
        # if self.points_recorded:
        #     self.points_recorded.pop()
        #     self.points_recorded.pop()
        self.canvas.create_line(self.points_recorded, fill = "yellow")
        

    def tell_me_where_you_are(self, event):
        self.previous_x = event.x
        self.previous_y = event.y

    def draw_from_where_you_are(self, event):
        # if self.points_recorded:
        #     self.points_recorded.pop()
        #     self.points_recorded.pop()

        self.x = event.x
        self.y = event.y
        self.canvas.create_line(self.previous_x, self.previous_y, 
                                self.x, self.y,fill="white", width=3)
        self.points_recorded.append((self.previous_x, self.previous_y))
        # self.points_recorded.append(self.previous_y)
        # self.points_recorded.append(self.x)     
        # self.points_recorded.append(self.x)        
        self.previous_x = self.x
        self.previous_y = self.y

    def prediction(self):
        img_arr = torch.tensor([[-1.0 for x in range(img_dims[0])] for i in range(img_dims[1])])
        for point in self.points_recorded:
            img_arr[point[1] // dim_multiplier][point[0] // dim_multiplier] = 1.0
        img_arr = img_arr.reshape((1, img_dims[1], img_dims[0]))
        pred = digit_model(img_arr)
        _, predicted = torch.max(pred.data, 1)
        self.open_popup(predicted.sum())
        self.clear_all()

    def open_popup(self, prediction):
        top= tk.Toplevel(self.win)
        top.geometry(f"{int(canvas_height // 1.5)}x{int(canvas_width // 1.5)}")
        top.title("Prediction")
        tk.Label(top, text=f"The prediction is: {prediction}", font=('Calibri 12 bold')).place(x=(canvas_width // 2 // 3),y=(canvas_height // 2 // 2))


if __name__ == "__main__":
    app = App()
    app.mainloop()
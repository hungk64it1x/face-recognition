from ast import arg
from cProfile import label
from json.tool import main
import os 
import argparse
import check_camera
import capture_image
from capture_image import *
import train_image
import recognize
import update_db
import numpy as np
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from config import USE_MTCNN

window = tk.Tk()

window.title("Hệ thống nhận diện khuôn mặt")
window.geometry('800x500')

bg = PhotoImage(file = r"background\background.png")
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
label1 = Label(window, image=bg)
label1.place(x = 0, y = 0)

def clear():
    std_name.delete(0, 'end')
    res = ""
    label4.configure(text=res)


def clear2():
    std_number.delete(0, 'end')
    res = ""
    label4.configure(text=res)

path2image = r'TrainingImage'
    
list_ids = os.listdir(path2image)
try:
    ID = np.max([int(i) for i in list_ids])
except:
    ID = 1

def checkCamera():
    label4.configure(text="Camera hoạt động tốt")
    if USE_MTCNN:
        check_camera.mtcnn()
    else:
        check_camera.haar()

def CaptureFaces():
    label4.configure(text="")
    name = (std_name.get())
    Id = (std_number.get())
    if is_number(Id) and name.isalpha():
        Id, name = capture_image.takeImages(Id, name)
        label4.configure(text="Lưu thông tin {} với ID: {} thành công!".format(name, Id))
    else:
        label4.configure(text="Chưa nhập thông tin hoặc thông tin không hợp lệ!")
    return Id


def Trainimages():
    label4.configure(text="")
    train_image.TrainImages()
    label4.configure(text="Huấn luyện toàn bộ dữ liệu thành công!".format(ID))

def UpdateDB():
    label4.configure(text="")
    list_ids = os.listdir(path2image)
    # ID = np.max([int(i) for i in list_ids])
    ID = std_number.get()
    update_db.UpdateDB(ID)
    label4.configure(text="Cập nhật dữ liệu cho ID {} thành công!".format(ID))

def RecognizeFaces():
    label4.configure(text="")
    recognize.recognize_attendence()

label1 = tk.Label(window, fg="black", text="Họ tên :", width=10, height=1,
                  font=('Helvetica', 16))
label1.place(x=120, y=110)
std_name = tk.Entry(window, background="white", fg="black", width=25, font=('Helvetica', 14))
std_name.place(x=280, y=110)
label2 = tk.Label(window, fg="black", text="Id :", width=10, height=1,
                  font=('Helvetica', 16))
label2.place(x=120, y=150)
std_number = tk.Entry(window, background="white", fg="black", width=25, font=('Helvetica', 14))
std_number.place(x=280, y=150)

clearBtn1 = tk.Button(window, background="blue", command=clear, fg="white", text="Xóa", width=8, height=1,
                      activebackground="red", font=('Helvetica', 10))
clearBtn1.place(x=580, y=110)
clearBtn2 = tk.Button(window, background="blue", command=clear2, fg="white", text="Xóa", width=8,
                      activebackground="red", height=1, font=('Helvetica', 10))
clearBtn2.place(x=580, y=150)

label4 = tk.Label(window, background="yellow", fg="black", width=55, height=6, font=('Helvetica', 14, 'italic'))
label4.place(x=95, y=340)

takeImageBtn = tk.Button(window, command=checkCamera, background="white", fg="black", text="Kiểm tra camera",
                         activebackground="red",
                         width=14, height=3, font=('Helvetica', 9))
takeImageBtn.place(x=50, y=240)
trainImageBtn = tk.Button(window, command=CaptureFaces, background="white", fg="black", text="Chụp ảnh",
                          activebackground="red",
                          width=14, height=3, font=('Helvetica', 9))
trainImageBtn.place(x=200, y=240)
trackImageBtn = tk.Button(window, command=Trainimages, background="white", fg="black", text="Huấn luyện từ đầu", width=14,
                          activebackground="red", height=3, font=('Helvetica', 9))
trackImageBtn.place(x=350, y=240)
trackImageBtn = tk.Button(window, command=UpdateDB, background="white", fg="black", text="Cập nhật DB", width=14,
                          activebackground="red", height=3, font=('Helvetica', 9))
trackImageBtn.place(x=500, y=240)
trackImageBtn = tk.Button(window, command=RecognizeFaces, background="white", fg="black", text="Nhận diện", width=14,
                          activebackground="red", height=3, font=('Helvetica', 9))
trackImageBtn.place(x=650, y=240)

window.mainloop()
# Wala' Essam Ashqar || ولاء عصام أشقر
# 12027854

import cv2
import numpy as np
from tkinter import filedialog
from tkinter import * 
import tkinter as Tk
from PIL import Image, ImageTk

original_img = None
output_img = None
imagefinal = None
grayfinal = None

def openimage():
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
    if file_path:
        global original_img
        original_img = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        label_width, label_height = int(label1.cget("width")), int(label1.cget("height"))
        resized_image = cv2.resize(image_rgb, (label_width, label_height))
        global imagefinal
        imagefinal = ImageTk.PhotoImage(Image.fromarray(resized_image))
        label1.config(image=imagefinal)
        label1.image = imagefinal
        label2.config(image=imagefinal)
        label2.image = imagefinal
        
def convertgrayscale():
    if hasattr(label1, 'image'):
        image_rgb = np.array(original_img)
        if image_rgb.dtype != np.uint8:
            image_rgb = np.array(image_rgb, dtype=np.uint8)
        grayscale_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        label_width, label_height = int(label2.cget("width")), int(label2.cget("height"))
        resizedgrayscale = cv2.resize(grayscale_image, (label_width, label_height))
        global grayfinal
        global output_img
        grayfinal = ImageTk.PhotoImage(Image.fromarray(resizedgrayscale))
        output_img = resizedgrayscale
        label2.config(image=grayfinal)
        label2.image = grayfinal

def saveimage():
    savepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if savepath:
        imgsave = Image.fromarray(output_img)
        imgsave.save(savepath)
        print(f"Processed image saved to {savepath}")
       
def applyfilter(x):
    global output_img
    filteredimage = cv2.filter2D(output_img, -1, x)
    output_img = filteredimage
    filteredfinal = ImageTk.PhotoImage(Image.fromarray(filteredimage))
    label2.config(image=filteredfinal)
    label2.image = filteredfinal

def pointdetect():
    x = np.array([[-1,-1,-1],
                  [-1, 8,-1],
                  [-1,-1,-1]], dtype=np.float32)
    x = x / (x.sum() + 1)
    applyfilter(x)

def Hline():
    x = np.array([[-1,-1,-1],
                  [ 2, 2, 2],
                  [-1,-1,-1]], dtype=np.float32)
    x = x / (x.sum() + 1)
    applyfilter(x)

def Vline():
    x = np.array([[-1, 2,-1],
                  [-1, 2,-1],
                  [-1, 2,-1]], dtype=np.float32)    
    x = x / (x.sum() + 1)
    applyfilter(x)

def line45m():
    x = np.array([[2, -1, -1],
                  [-1, 2, -1],
                  [-1, -1, 2]], dtype=np.float32)
    x = x / (x.sum() + 1)
    applyfilter(x)
    
def line45p():
    x = np.array([[-1,-1,  2],
                  [-1, 2, -1],
                  [ 2,-1, -1]], dtype=np.float32)
    x = x / (x.sum() + 1)
    applyfilter(x)

def Hedge():
     x = np.array([[-1,-2,-1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]], dtype=np.float32)
     x = x / (x.sum() + 1)
     applyfilter(x)

def Vedge():
     x = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
     x = x / (x.sum() + 1)
     applyfilter(x)

def edge45p():
     x = np.array([[-2,-1, 0],
                   [-1, 0, 1],
                   [ 0, 1, 2]], dtype=np.float32)
     x = x / (x.sum() + 1)
     applyfilter(x)

def edge45m():
     x = np.array([[ 0, 1, 2],
                   [-1, 0, 1],
                   [-2,-1, 0]], dtype=np.float32)
     x = x / (x.sum() + 1)
     applyfilter(x)

def laplace():
     x = np.array([[ 0,-1, 0],
                   [-1, 4,-1],
                   [ 0,-1, 0]], dtype=np.float32)
     x = x / (x.sum() + 1)
     applyfilter(x)

def log():
     x = np.array([[ 0, 0,-1, 0, 0],
                   [ 0,-1,-2,-1, 0],
                   [-1,-2,16,-2,-1],
                   [ 0,-1,-2,-1, 0],
                   [ 0, 0,-1, 0, 0]], dtype=np.float32)
     x = x / (x.sum() + 1)
     applyfilter(x)

def zerocrossing():
    global output_img
    if output_img is not None:
        if len(output_img.shape) == 3:
            outputgray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        else:
            outputgray = output_img
        zerocrossings = np.zeros_like(outputgray, dtype=np.uint8)
        for i in range(1, outputgray.shape[0] - 1):
            for j in range(1, outputgray.shape[1] - 1):
                n = [outputgray[i - 1, j], outputgray[i + 1, j],
                              outputgray[i, j - 1], outputgray[i, j + 1]]
                if np.any(np.diff(np.sign(n))):
                    zerocrossings[i, j] = 255
        output_img = zerocrossings
        zerocrossingfinal = ImageTk.PhotoImage(Image.fromarray(zerocrossings))
        label2.config(image=zerocrossingfinal)
        label2.image = zerocrossingfinal
    else:
        print("Please open an image first.")

def threshold(threshold_value):
    global output_img
    if output_img is not None:
        if len(output_img.shape) == 3:
            outputgray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        else:
            outputgray = output_img
        _, thresholded = cv2.threshold(outputgray, threshold_value, 255, cv2.THRESH_BINARY)
        output_img = thresholded
        thresholdedfinal = ImageTk.PhotoImage(Image.fromarray(thresholded))
        label2.config(image=thresholdedfinal)
        label2.image = thresholdedfinal
    else:
        print("Please open an image first.")

def adaptivethreshold(size, c):
    global output_img
    if output_img is not None:
        if len(output_img.shape) == 3:
            outputgray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        else:
            outputgray = output_img
        adaptivethresholded = cv2.adaptiveThreshold(
            outputgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, size, c)
        output_img = adaptivethresholded
        adaptivethresholdedfinal = ImageTk.PhotoImage(Image.fromarray(adaptivethresholded))
        label2.config(image=adaptivethresholdedfinal)
        label2.image = adaptivethresholdedfinal
    else:
        print("Please open an image first.")

def userdefined():
    global output_img
    if output_img is not None:
        size = int(txt2.get("1.0", "end-1c"))  
        coefficientsstring = txt1.get("1.0", "end-1c")
        coefficients = [float(coeff.strip()) for coeff in coefficientsstring.split(",")]
        filtersize = int(np.sqrt(len(coefficients)))
        if filtersize * filtersize != len(coefficients):
            print("Error: The number of coefficients is nor correct.")
            return
        userdefined = np.array(coefficients).reshape((filtersize, filtersize))
        userdefined = userdefined / (userdefined.sum() + 1)
        filteredimage = cv2.filter2D(output_img, -1, userdefined)
        output_img = filteredimage
        filteredfinal = ImageTk.PhotoImage(Image.fromarray(filteredimage))
        label2.config(image=filteredfinal)
        label2.image = filteredfinal
    else:
        print("Please open an image first.")

root = Tk.Tk()
root.geometry('1100x800')
root.title('Filter App')

imagef  = Tk.Frame(root,highlightbackground='black',highlightthickness=1)
label1 = Label(imagef,width='450', height= '700',highlightbackground='black',highlightthickness=1)
label2 = Label(imagef,width='450', height= '700',highlightbackground='black',highlightthickness=1)

labelsf  = Tk.Frame(root,highlightbackground='black',highlightthickness=1)
inputl = Label(labelsf,text='input image',width='50',highlightbackground='black',highlightthickness=1, height= '100',font=('bold',12))
outputl = Label(labelsf,text='output image',width='50',highlightbackground='black',highlightthickness=1, height= '100',font=('bold',12))
note1 = Label(labelsf,text='Please be sure that the image in grayscale before everytime you change the filter.',width='68', height= '1',font=('bold',8))

optionsf  = Tk.Frame(root, bg= '#c3c3c3',highlightbackground='black',highlightthickness=2)
btn1 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Open image",command=openimage)
btn2 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Grayscale", command=convertgrayscale)
btn3 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Point detection",command=pointdetect)
btn4 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Horizontal edge detection",command=Hedge)
btn5 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Horizontal line detection",command=Hline)
btn6 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Vertical edge detection",command=Vedge)
btn7 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Vertical line detection",command=Vline)
btn8 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="+45 line detection",command=line45p)
btn9 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="-45 line detection", command=line45m)
btn10 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="+45 edge detection",command=edge45p)
btn11 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="-45 edge detection",command=edge45m)
btn12 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Laplacian filter",command=laplace)
btn13 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Laplacian of Gaussian",command=log)
btn14 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Zero crossing",command=zerocrossing)
btn15 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Threshold",command=lambda: threshold(128))
btn16 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Adaptive threshold",command=lambda:adaptivethreshold(11, 2))
btn17 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="User-defined filter",command=userdefined)
btn18 = Button(optionsf, width= "19", height= "1", font=('bold',9),text="Save", command=saveimage)
txt1 = Text(optionsf, width= "12", height= "1", font=('bold',8))
txt2 = Text(optionsf, width= "12", height= "1", font=('bold',8))
placeholder1 = Tk.Label(optionsf,width= "10", height= "1", text="coefficients:", fg='gray',font=('bold',8))
placeholder2 = Tk.Label(optionsf,width= "10", height= "1", text="size:", fg='gray',font=('bold',8))
note2 = Tk.Label(optionsf,width= "30", height= "1", text="Put comma , between coefficients 1 or 0",font=('bold',7))

optionsf.pack(side= LEFT)
optionsf.pack_propagate(False)
optionsf.configure(width=200, height=800) 

labelsf.pack(side=TOP)
labelsf.pack_propagate(False)
labelsf.configure(width=900, height=100) 
inputl.pack(side= LEFT)
outputl.pack(side=RIGHT)

imagef.pack(side=BOTTOM)
imagef.pack_propagate(False)
imagef.configure(width=900, height=700) 
label1.pack(side = LEFT)
label2.pack(side = RIGHT)
note1.pack()
note1.place(x=480,y=64)

btn1.pack(pady=4)
btn2.pack(pady=3)
btn3.pack(pady=3)
btn4.pack(pady=3)
btn5.pack(pady=3)
btn6.pack(pady=3)
btn7.pack(pady=3)
btn8.pack(pady=3)
btn9.pack(pady=3)
btn10.pack(pady=3)
btn11.pack(pady=3)
btn12.pack(pady=3)
btn13.pack(pady=3)
btn14.pack(pady=3)
btn15.pack(pady=3)
btn16.pack(pady=3)
btn17.pack(pady=3)
btn18.pack(pady=3)
txt1.pack(pady=3)
txt2.pack(pady=3)
note2.pack(pady=3)

placeholder1.place(x=23,y=580)
placeholder2.place(x=23,y=600)
txt1.place(x=94, y=580)
txt2.place(x=94, y=600)
note2.place(x=6,y=623)

root.mainloop() 

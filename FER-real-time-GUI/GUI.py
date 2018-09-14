from Tkinter import*
import time
import sys
from PIL import ImageTk
from PIL import Image
import real_time_class
import sub_inter_class

gui=Tk()
gui.geometry('520x350')
gui.title('Facial Expression Recognition')

def exit_gui():
    gui.destroy()
    exit()

def data_training():
    print 'data_training'

def Interaction():
    interaction=sub_inter_class.sub_in()
    #camera1=camera.camera_on()

# resize and add images
image_path='/home/liutao/landmark-FER-SVM/reactions.png'
img=Image.open(image_path)
imsize=img.size
new_width=520
new_height=new_width*imsize[1]/imsize[0]
img=img.resize((new_width,new_height))
img=ImageTk.PhotoImage(img)
panel=Label(gui,image=img)
panel.place(x=0,y=0)

# title
label_1=Label(gui,text='Facial Expression Recognition', bg= 'gray', fg='white', font= 'non 22 bold')
label_1.place(x=0,y=new_height,width=new_width)

# buttons
button_1=Button(gui,text='Model-Training',width=20,command=data_training)
button_1.place(x=40,y=220)

button_2=Button(gui,text='Real-Time Interaction',width=20,command=Interaction)
button_2.place(x=290,y=220)

button_3=Button(gui,text='Exit',width=10,command=exit_gui)
button_3.place(x=205,y=315)

# label
label_2=Label(gui,text='Copyright by Xiangyi Cheng', fg='gray', font= 'non 9 ')
label_2.place(x=340,y=330)

label_3=Label(gui,text='CWRU, Cleveland', fg='gray', font= 'non 9 ')
label_3.place(x=10,y=330)

gui=mainloop()


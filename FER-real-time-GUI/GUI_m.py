from Tkinter import*
import time
import sys
from PIL import ImageTk
from PIL import Image
import sub_inter_class
import landmark_class
import landmark_VL_class
import landmark_AU_class


class SeaofBTCapp(Tk):

    def __init__(self, *args, **kwargs):
        
        Tk.__init__(self, *args, **kwargs)
        container = Frame(self)

        container.pack(side="top", fill="both", expand = True)

        self.geometry('520x350')
        self.title('Facial Expression Recognition')

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self,parent)

        #label = Label(self, text="Start Page", font=LARGE_FONT)
        #label.pack(pady=10,padx=10)

        image_path='/home/leo/woody_vision/Woody-FEM-SVM-master/emoji.png'
        img=Image.open(image_path)
        imsize=img.size
        new_width=520
        new_height=new_width*imsize[1]/imsize[0]
        img=img.resize((new_width,new_height))
        self.img=ImageTk.PhotoImage(img)
        panel=Label(self,image=self.img)
        panel.place(x=0,y=0)

        # title
        label_1=Label(self,text='Facial Expression Recognition', bg= 'gray', fg='white', font= 'non 22 bold')
        label_1.place(x=0,y=new_height,width=new_width)

        # buttons
        button_1=Button(self,text='Model-Training',width=20,command=lambda:controller.show_frame(PageOne))
        button_1.place(x=40,y=220)

        button_2=Button(self,text='Real-Time FER',width=20,command=self.Interaction)
        button_2.place(x=290,y=220)

        button_3=Button(self,text='Exit',width=10,command=self.exit_gui)
        button_3.place(x=205,y=315)

        # label
        label_2=Label(self,text='Copyright by Xiangyi Cheng', fg='gray', font= 'non 9 ')
        label_2.place(x=340,y=330)

        label_3=Label(self,text='CWRU, Cleveland', fg='gray', font= 'non 9 ')
        label_3.place(x=10,y=330)

    def exit_gui(self):
        self.destroy()
        exit()

    def Interaction(self):
        interaction=sub_inter_class.sub_in()


class PageOne(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        #label = Label(self, text="Page One!!!", font=LARGE_FONT)
        #label.pack(pady=10,padx=10)

        label1_1 = Label(self, text='Model Setting',font= 'non 15 bold')
        label1_1.pack(pady=(8,0))

        label_frame1 = LabelFrame(self,text='Feature Extraction', width=247,heigh=275,bd=3)
        label_frame1.place(x=8,y=40)

        label_frame2 = LabelFrame(self,text='Training Setting',width=247,heigh=275,bd=3)
        label_frame2.place(x=262,y=40)




        ############################# Frame 1################################################################
        label2_1 = Label(label_frame1, text='Feature Types')
        label2_1.pack(padx=(5,10),pady=(10,5))

        self.v1=IntVar()

        R11= Radiobutton(label_frame1,text='Vecterized Landmark Only (VL)',value=11,variable=self.v1)
        R11.pack(anchor=W)
        R12= Radiobutton(label_frame1,text='Landmark Curvature (LC)',value=12,variable=self.v1)
        R12.pack(anchor=W)
        R13= Radiobutton(label_frame1,text='Both Two (VL+LC)',value=13,variable=self.v1)
        R13.pack(anchor=W,pady=(0,10))
        R13.invoke() # default choice

        label2_2 = Label(label_frame1,text='The Number of Landmarks for LC')
        label2_2.pack(padx=(5,0),pady=(10,5))


        self.tkvar = StringVar()
 
        # Dictionary with options
        choices = { '13','15','28','39','44','N/A'}
        self.tkvar.set('44') # set the default option
 
        popupMenu = OptionMenu(label_frame1, self.tkvar, *choices)
        popupMenu.pack(pady=(3,0))

        label2_8 = Label(label_frame1, text='L =')
        label2_8.place(x=42,y=150)

        label2_3 = Label(label_frame1,text='Weight Distribution')
        label2_3.pack(padx=(5,0),pady=(10,5))

        label2_4 = Label(label_frame1,text='VL for')
        label2_4.pack(side= LEFT,padx=(5,0),pady=(10,5))

        self.entry11 = Entry(label_frame1,width=5)
        self.entry11.pack(side=LEFT)
        self.entry11.insert(0, '75')

        label2_5 = Label(label_frame1,text='%')
        label2_5.pack(side= LEFT,pady=(10,5))

        label2_6 = Label(label_frame1,text='%')
        label2_6.pack(side=RIGHT,pady=(10,5))

        self.entry12 = Entry(label_frame1,width=5)
        self.entry12.pack(side=RIGHT)
        self.entry12.insert(0, '25')

        

        label2_7 = Label(label_frame1,text='LC for')
        label2_7.pack(side=RIGHT,padx=(5,0),pady=(10,5))


        



        ################################ Frame 2############################################################# 
        label1_2 = Label(label_frame2,text='Data Distribution')
        label1_2.pack(padx=(5,10),pady=(10,5))

        self.v2=IntVar()

        R21= Radiobutton(label_frame2,text='Training 90%, Prediction 10%',value=21,variable=self.v2)
        R21.pack(anchor=W)
        R22= Radiobutton(label_frame2,text='Training 80%, Prediction 20%',value=22,variable=self.v2)
        R22.pack(anchor=W)
        R23= Radiobutton(label_frame2,text='Training 70%, Prediction 30%',value=23,variable=self.v2)
        R23.pack(anchor=W,pady=(0,10))
        R22.invoke() # default choice

        label1_3 = Label(label_frame2,text='Penalty Parameter of the Error Term')
        label1_3.pack(padx=(5,0),pady=(10,5))

        label1_7 = Label(label_frame2,text='n =')
        label1_7.pack(side=BOTTOM,padx=(5,0),pady=(11,5),anchor=W)

        label1_6 = Label(label_frame2,text='The Number of the Training Set')
        label1_6.pack(side=BOTTOM,padx=(5,0),pady=(13,4))

        label1_4 = Label(label_frame2,text='c =')
        label1_4.pack(side=LEFT,padx=(55,0),pady=(6,3))

        self.tkvar1 = StringVar()
 
        # Dictionary with options
        choices1 = { '1','3.16','10','31.62','100'}
        self.tkvar1.set('100') # set the default option
 
        popupMenu1 = OptionMenu(label_frame2, self.tkvar1, *choices1)
        popupMenu1.place(x=95,y=143)

        self.entry2 = Entry(label_frame2,width=5)
        self.entry2.place(x=33,y=215)
        self.entry2.insert(END, '1')

        label1_8 = Label(label_frame2,text='(1~10 is recommended)')
        label1_8.place(x=80,y=215)



        ####################################### Buttons###################################################### 
        button1 = Button(self, text="Back to Home", width=10,
                            command=lambda: controller.show_frame(StartPage))
        button1.place(x=404,y=309)


        button3 = Button(self, text="Start Training", width=10,
                            command=self.start_t)
        button3.place(x=276,y=309)

        #self.tkvar2 = StringVar()
        self.label_reminer = Label(self, text='')
        self.label_reminer.place(x=6,y=309)




    def start_t(self):
        #self.label_reminer.config(text='')

        if self.v1.get()==11:
            
            self.tkvar.set('N/A')
            self.entry11.delete(0, 'end')
            self.entry12.delete(0, 'end')
            self.entry11.insert(0, 'N/A')
            self.entry12.insert(0, 'N/A')

            if self.entry2.get():
                n_training_set = int(self.entry2.get())
                penalty_c = int(self.tkvar1.get()) 

                if self.v2.get()==21:
                    training_percentage=0.9

                if self.v2.get()==22:
                    training_percentage=0.8

                if self.v2.get()==23:
                    training_percentage=0.7

                training = landmark_VL_class.training_model(n_training_set,penalty_c,training_percentage)
                accuracy=round(training.p_pred_lin,4)
                textvar='Trained, accuracy = '+str(accuracy)
                self.label_reminer.config(text=textvar,fg='green')

            else:
                self.label_reminer.config(text='Not Completed',fg='red')



        if self.v1.get()==12:
            self.entry11.delete(0, 'end')
            self.entry12.delete(0, 'end')
            self.entry11.insert(0, 'N/A')
            self.entry12.insert(0, 'N/A')
            if self.entry2.get() and self.tkvar.get()!='N/A':
                
                n_landmark = int(self.tkvar.get())

                n_training_set = int(self.entry2.get())
                penalty_c = int(self.tkvar1.get()) 

                if self.v2.get()==21:
                    training_percentage=0.9

                if self.v2.get()==22:
                    training_percentage=0.8

                if self.v2.get()==23:
                    training_percentage=0.7

                training = landmark_AU_class.training_model(n_training_set,penalty_c,training_percentage,n_landmark)
                accuracy=round(training.p_pred_lin,4)
                textvar='Trained, accuracy = '+str(accuracy)
                self.label_reminer.config(text=textvar,fg='green')

            else:
                self.label_reminer.config(text='Not Completed',fg='red')


        if self.v1.get()==13:

            if self.entry11.get() and self.entry12.get() and self.entry2.get() and self.tkvar.get()!='N/A':

                n_landmark = int(self.tkvar.get())

                AL_percentage = int(self.entry11.get())
                AU_percentage = int(self.entry12.get())

                if AL_percentage+AU_percentage==100:

                    w1 = 0.01 * AL_percentage
                    n_training_set = int(self.entry2.get())
                    penalty_c = int(self.tkvar1.get()) 

                    if self.v2.get()==21:
                        training_percentage=0.9

                    if self.v2.get()==22:
                        training_percentage=0.8

                    if self.v2.get()==23:
                        training_percentage=0.7

                    training = landmark_class.training_model(w1,n_training_set,penalty_c,training_percentage,n_landmark)
                    accuracy=round(training.p_pred_lin,4)
                    textvar='Trained, accuracy = '+str(accuracy)
                    self.label_reminer.config(text=textvar,fg='green')


                else:

                    self.label_reminer.config(text='Wrong Percentage',fg='red')

            else:

                self.label_reminer.config(text='Not Completed',fg='red')

app = SeaofBTCapp()
app.mainloop()
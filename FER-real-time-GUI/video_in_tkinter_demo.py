from Tkinter import*
import cv2
import PIL.Image, PIL.ImageTk
import time
 
class App:
    def __init__(self):
        self.window = Tk()
        self.window.title('Real Time Detection')
  
 
         # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture()
 
         # Create a canvas that can fit the above video source size
        self.canvas = Canvas(self.window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
 
         # Button that lets the user take a snapshot
        self.btn_snapshot=Button(self.window, text="Detection", width=30, command=self.detection)
        self.btn_snapshot.pack(anchor=CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def detection(self):
        print 'hi'
 
    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        #ret, frame = self.vid.read()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self):
        # Open the video source
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                            # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
App()
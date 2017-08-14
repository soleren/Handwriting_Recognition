# -*- coding:utf-8 -*-

from tkinter import *
from PIL import ImageGrab,Image
import random,os
import numpy as np
import NeuralNet


class Paint(Frame):
    width = 200
    height = 200
    # alphabet = ['А','Б','В','Г','Д','Е','Ё','Ж','З','И','К','Л','М','Н','О','П','Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Щ','Ъ','Ь','Ы','Э','Ю','Я','0','1','2','3','4','5','6','7','8','9']
    alphabet = ['0','1','2','3','4']
    learned = []

    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.hnodes = 400
        self.learning_rate = 0.001
        self.nnet = NeuralNet.NeuralNet((self.width * self.height), self.hnodes,len(self.alphabet),self.learning_rate)

        self.setUI()
        self.brush_size = 8
        self.brush_color="black"


    def setUI(self):

        self.parent.title("Handwriting recognition")
        self.pack(side=RIGHT,fill=BOTH)
        self.canv = Canvas(self, bg="white", width=self.width, height=self.height)
        self.canv.grid(row=0, rowspan=10,columnspan=5)
        self.canv.bind("<B1-Motion>", self.draw)
        self.canv.bind("<Button-3>", self.clear)

        learn_label = Label(self, text="-" * 10 + " Learn " + "-" * 10)
        learn_label.grid(row=0, column=5, columnspan=5)

        learn_first = Label(self, text="1. Write a letter       ")
        learn_first.grid(row=1, column=5,columnspan=2)
        learn_first = Label(self, text="rclick to clear")
        learn_first.grid(row=1, column=7,columnspan=2)

        learn_second = Label(self, text="2. Select the letter   ")
        learn_second.grid(row=2, column=5,columnspan=2)
        self.var = StringVar(self)
        learn_second_input = OptionMenu(self, self.var, *self.alphabet,command=self.get_value)
        learn_second_input.grid(row=2, column=7,columnspan=3)

        learn_third_label = Label(self, text="3. Push the button  ")
        learn_third_label.grid(row=3, column=5,columnspan=2)
        learn_third_btn = Button(self, text="Learn", width=8, command=lambda: self.learn_btn(self))
        learn_third_btn.grid(row=3, column=7, padx=5,columnspan=3)

        recognize_label = Label(self, text="-" * 10 + " Recognize " + "-" * 10)
        recognize_label.grid(row=4, column=5, columnspan=5)

        recognize_first = Label(self, text="4. Write a letter        ")
        recognize_first.grid(row=5, column=5,columnspan=2)

        recognize_second_label = Label(self, text="5. Push the button  ")
        recognize_second_label.grid(row=6, column=5,columnspan=2)

        recognize_second_btn = Button(self, text="Recognize", width=8,command=lambda: self.recognize(self))
        recognize_second_btn.grid(row=6, column=7, padx=5,columnspan=3)

        learned_label = Label(self, text="-" * 32 + " Learned " + "-" * 32)
        learned_label.grid(row=11, column=0, columnspan=10)

        result_label = Label(self, text="-" * 33 + " Result " + "-" * 33)
        result_label.grid(row=20, column=0, columnspan=10)

        col = 0
        row = 0
        for i in range(len(self.alphabet)):
            if i % 6 == 0 and i != 0:
                col+=1
                row=0
            Label(self, text="%s:%1.3f" % (self.alphabet[i],0)).grid(row=21+row, column=col+1,sticky=W)
            row+=1

    def draw(self,event):
        self.canv.create_oval(event.x - self.brush_size,event.y - self.brush_size,
                              event.x + self.brush_size,event.y + self.brush_size,
                              fill=self.brush_color, outline=self.brush_color)

    def clear(self, event):
        self.canv.delete("all")

    def learn_btn(self, widget):

        x = self.winfo_rootx()
        y = self.winfo_rooty()
        x1 = x + self.width
        y1 = y + self.height
        # self.li = ImageGrab.grab().crop((x, y, x1, y1)).save(os.getcwd()+"\\img\\" + str(random.randint(1, 1000000000))+".JPG")
        im = ImageGrab.grab().crop((x, y, x1, y1))

        numimage = np.dot(np.array(im),[0.299, 0.587, 0.114]).ravel()
        # im = Image.fromarray(numimage)
        # im.show()
        im.save(os.getcwd()+"\\img\\" + str(random.randint(1, 1000000000))+".JPG")
        scaled_image = (numimage / 255 * 0.98) + 0.01
        # print(list(scaled_image))
        # im = Image.fromarray(scaled_image)
        # im.show()
        if self.curval not in self.learned:
            self.learned.append(self.curval)

        output = [0.99 if self.curval == x else 0.01 for x in self.alphabet ]
        print(output)
        for i in range(40):
            self.nnet.train(scaled_image,output)
        r = 12
        c = 0
        for i in range(len(self.learned)):
            if i % 10 == 0 and i != 0:
                r += 1
                c = 0
            Label(self, text="%s " % self.learned[i]).grid(row=r, column=c)
            c+=1


    def recognize(self,widget):
        x = self.winfo_rootx()
        y = self.winfo_rooty()
        x1 = x + self.width
        y1 = y + self.height
        # self.rl = ImageGrab.grab().crop((x, y, x1, y1)).save(os.getcwd()+"\\img\\" + str(random.randint(1, 1000000000))+".JPG")
        # if self.li == self.rl:
        #     print("!!!!")
        im = ImageGrab.grab().crop((x, y, x1, y1))
        # im.show()
        numimage = np.dot(np.array(im),[0.299, 0.587, 0.114]).ravel()
        scaled_image = (numimage / 255 * 0.98) + 0.01
        result = self.nnet.work(scaled_image)
        print(result)
        col = 0
        row = 0
        for i in range(len(result)):
            if i % 6 == 0 and i != 0:
                col+=1
                row=0
            Label(self, text="%s:%3.2f " % (self.alphabet[i],result[i]*100)).grid(row=21+row, column=col+1,sticky=W)
            row+=1


    def get_value(self,value):
        self.curval = value

def main():
    root = Tk()
    root.geometry("390x475")
    root.resizable(width=FALSE, height=FALSE)
    app = Paint(root)
    root.mainloop()


if __name__ == "__main__":
    main()

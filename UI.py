import shutil
import tkinter
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from tkinter.messagebox import showerror

import cv2 as cv
import self
from PIL import Image, ImageTk
import numpy as np
import sys, random, datetime, os, winreg, getpass, time, threading
import os
from matplotlib import pyplot as plt

import predict
import split2
import plate_locate

# PicPath: Save the last picture file path, this picture should be opened successful.
class LPRPath:
    def getPath(self):
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\{}\LPR".format(getpass.getuser()))
            self.path = winreg.QueryValueEx(key, "LPR")
        except:
            self.path = None
        return self.path

    def setPath(self, path):
        key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\{}\LPR".format(getpass.getuser()))
        winreg.SetValueEx(key, "LPR", 0, winreg.REG_SZ, path)
        self.path = path

# Main LPR surface
class LPRSurface(Tk):
	#设置窗体大小
	labelPicWidth  	= 700
	labelPicHeight 	= 550
	buttonWidth 	= 200
	buttonHeight 	= 50
	textWidth 		= 10
	textHeight 		= 50
	tkWidth 		= labelPicWidth*2-200
	tkHeigth 		= labelPicHeight + buttonHeight * 4
	isPicProcessing = False
	root 			= None

	def resizePicture(self, imgCV):
		if imgCV is None:
			print("Read Fail!")
			return None

		imgCVRGB = cv.cvtColor(imgCV, cv.COLOR_BGR2RGB)
		img = Image.fromarray(imgCVRGB)
		imgTK = ImageTk.PhotoImage(image=img)

		picWidth = imgTK.width()
		picHeight = imgTK.height()
		# print("Picture Size:", picWidth, picHeight)
		if picWidth <= self.labelPicWidth and picHeight <= self.labelPicHeight:
			return imgTK

		widthScale = 1.0*self.labelPicWidth/picWidth
		heightScale = 1.0*self.labelPicHeight/picHeight

		scale = min(widthScale, heightScale)

		resizeWidth = int(picWidth*scale)
		resizeHeight = int(picHeight*scale)

		img = img.resize((resizeWidth, resizeHeight), Image.Resampling.LANCZOS)
		imgTK = ImageTk.PhotoImage(image=img)

		return imgTK

	# Load picture
	def loadPicture(self):
		#Get Picture Path
		if True == self.isPicProcessing:
			print("Please wait until previous picture process finish!!!")
			messagebox.showerror(title="PROCESSING", message="Please wait until previous picture process finish!!!")
			return

		LPRPic = LPRPath()
		if None == LPRPic.getPath():
			initPath = ""
		else:
			initPath = LPRPic.path

		# fileName = None
		fileName = filedialog.askopenfilename(title='Load Picture', \
											  filetypes=[('Picture File', '*.jfif *.jpg *.png *.gif'), ('All Files', '*')], \
											  initialdir=initPath)

		#print(fileName)
		if not os.path.isfile(fileName):
			print("Please input correct filename!")
			return False
		# Read Picture File.
		try:
			imgCV = cv.imread(fileName)
		except:
			print("Open file faile!")
			return False

		LPRPic.setPath(fileName)
		self.imgOri = self.resizePicture(imgCV)
		if self.imgOri is None:
			print("Load picture fail!")
			return False

		# self.imgOri = ImageTk.PhotoImage(self.imgOri)
		self.labelPic.configure(image = self.imgOri, bg="#e9defa")

		#print(type(fileName))

		dst = plate_locate.plate_locate(fileName)
		basename = os.path.basename(fileName)  # 从全路径获取文件名
		path='ccpd-roii'

		if not os.path.exists(path):
			os.makedirs(path)

		if dst is None:
			showerror(title="error", message="sorry,license plate recognition failed!")  # 信息错误弹窗，点击确定返回值为 o
			return
		else:
			cv.imwrite(path+'/'+basename,dst)

			print(fileName)
			split2.run(fileName)#字符分割
			name1,name=split2.getName(fileName)
			pathSplit='ture_car_characters/'+name
			print(pathSplit)

			if not os.path.exists(pathSplit):
				showerror(title="error",message="sorry,license plate recognition failed!")# 信息错误弹窗，点击确定返回值为 o
				return
			else:
				# realpath='D:/MyCode/Coding/car-plate/'
				# shutil.copy(fileName,realpath)
				result,acc=predict.predict(name)
				acc=str(round(acc * 100, 2)) + '%'
				self.entryPlateNumList =list(result)#字符识别
				#self.entryPlateNumList = ["皖","A","W","X","E","9","1"]
				for index in range(7):
					entryPlateNum = Entry(self,font=("Arial", 25))
					entryPlateNum.place(x=self.textWidth * index * 6, y=self.labelPicHeight + self.textHeight * 2, \
										width=self.textWidth * 5, height=self.textHeight)
					# entryPlateNum.append(self.entryPlateNumList[index])
					entryPlateNum.insert('1', self.entryPlateNumList[index])

				#车牌定位结果
				global imgLic  # 函数运行结束就被回收了，会显示的是空白
				imgLic_open = Image.open(path+'/'+basename)
				#imgLic_open = Image.open('D:/MyCode/Coding/ccpd-roii/01-90_87-241&550_440&622-431&621_237&617_239&552_433&556-0_0_21_30_32_32_0-155-35.jpg')
				imgLic = ImageTk.PhotoImage(imgLic_open)
				label_img = Label(self, image=imgLic)
				label_img.place(x=700, y=55, \
											 width=500, height=200)

				#字符切割结果
				global imageSe  # 函数运行结束就被回收了，会显示的是空白
				imageSe = []
				for i in range(7):
					imageSe.append(ImageTk.PhotoImage(Image.open('car_characters/'+name+f'/image_{i}.jpg')))
					#imageSe.append(ImageTk.PhotoImage(Image.open(f'D:/MyCode/Coding/car_characters/0_0_23_20_32_29_29/image_{i}.jpg')))  # 分别打开并显示
					imageSe_img = Label(self, image=imageSe[i])
					imageSe_img.place(x=700 +50 + 50*i, y=285, \
									width=50, height=100)

				#车牌识别准确率
				entryPlateNum = Entry(self, font=("Arial", 20))
				entryPlateNum.place(x=700 + 50, y=455, \
									width=100, height=50)
				entryPlateNum.insert('1', acc)

	def __init__(self, *args, **kw):
		super().__init__()
		self.title("LPR Surface")
		self.geometry(str(self.tkWidth) + "x" + str(self.tkHeigth))
		self.resizable(0, 0)

		def labelInit():
			# Picture Label:
			self.labelPic = Label(self, text="Show Picture Area", font=("Arial", 24), bg="#c2e9fb")
			self.labelPic.place(x=0, y=0, width=self.labelPicWidth, height=self.labelPicHeight)

			# Vehicle Plate Number Label:
			self.labelPlateNum = Label(self, text="Vehicle License Plate Number:", anchor=SW,font=("Arial", 15))
			self.labelPlateNum.place(x=0, y=self.labelPicHeight+10, \
									 width=self.textWidth * 40, height=self.textHeight)
			# Vehicle Plate Location Label:
			self.labelPlateLocation = Label(self, text="Vehicle License Plate Location:", anchor=SW,font=("Arial", 15))
			self.labelPlateLocation.place(x=self.labelPicWidth+50, y=0, \
									 width=300, height=50)
			# Vehicle Character Segmentation Label:
			self.labelPlateNum = Label(self, text="Vehicle Character Segmentation:", anchor=SW,font=("Arial", 15))
			self.labelPlateNum.place(x=700+50, y=233, \
									 width=300, height=50)
			# Vehicle Accuracy Label:
			self.labelPlateNum = Label(self, text="Vehicle Character Identification Accuracy :", anchor=SW, font=("Arial", 15))
			self.labelPlateNum.place(x=700 + 50, y=400, \
									 width=400, height=50)

		def buttonInit():
			# Picture Button
			self.buttonPic = Button(self, text="Load Picture", command=self.loadPicture,font=("Arial", 15))
			self.buttonPic.place(x=self.tkWidth - 3 * self.buttonWidth / 2,
								 y=self.labelPicHeight + self.buttonHeight / 0.6, \
								 width=self.buttonWidth, height=self.buttonHeight)

		labelInit()
		buttonInit()

		#当点击窗口x退出时，执行的程序
		def on_closing():
			#self.destroy()
			self.quit()

		# WM_DELETE_WINDOW 不能改变，这是捕获命令
		self.protocol('WM_DELETE_WINDOW', on_closing)
		self.mainloop()


if __name__ == '__main__':
	LS = LPRSurface()
# 车牌识别
import shutil
import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
import plate_locate
# 限制图像大小（车牌）
def Limit(image):
    height, width, channel = image.shape
    # 设置权重
    weight = width / 300
    # 计算输出图像的宽和高
    last_width = int(width / weight)
    last_height = int(height / weight)
    image = cv.resize(image, (last_width, last_height))
    return image

# 二-5、统计白色像素点（分别统计每一行、每一列）
def White_Statistic(image):
    ptx = []  # 每行白色像素个数
    pty = []  # 每列白色像素个数
    height, width = image.shape
    # 逐行遍历
    for i in range(height):
        num = 0
        for j in range(width):
            if (image[i][j] == 255):
                num = num + 1
        ptx.append(num)

    # 逐列遍历
    for i in range(width):
        num = 0
        for j in range(height):
            if (image[j][i] == 255):
                num = num + 1
        pty.append(num)

    return ptx, pty

# 二-6、绘制直方图
def Draw_Hist(ptx, pty):
    # 依次得到各行、列
    rows, cols = len(ptx), len(pty)
    row = [i for i in range(rows)]
    col = [j for j in range(cols)]
    # 横向直方图
    plt.barh(row, ptx, color='black', height=1)
    #       纵    横
    #plt.show()
    # 纵向直方图
    plt.bar(col, pty, color='black', width=1)
    #       横    纵
    #plt.show()

# 二-7-2、横向分割：上下边框
def Cut_X(ptx, rows):
    # 横向切割（分为上下两张图，分别找其波谷，确定顶和底）
    # 1、下半图波谷
    min, r = 300, 0
    for i in range(int(rows / 2)):
        if ptx[i] < min:
            min = ptx[i]
            r = i
    h1 = r  # 添加下行（作为顶）

    # 2、上半图波谷
    min, r = 300, 0
    for i in range(int(rows / 2), rows):
        if ptx[i] < min:
            min = ptx[i]
            r = i
    h2 = r  # 添加上行（作为底）

    return h1, h2

# 二-7-3、纵向分割：分割字符
def Cut_Y(pty, cols, h1, h2, binary):
    WIDTH = 32  # 经过测试，一个字符宽度约为32
    w = w1 = w2 = 0  # 前谷 字符开始 字符结束
    begin = False  # 字符开始标记
    last = 10  # 上一次的值
    con = 0  # 计数
    # 纵向切割（正式切割字符）
    for j in range(int(cols)):
        if con==7:
            break
        # 0、极大值判断
        if pty[j] == max(pty):
            if j < 30:  # 左边（跳过）
                w2 = j
                if begin == True:
                    begin = False
                continue

            elif j > 280:  # 右边（直接收尾）
                if begin == True:
                    begin = False
                w2 = j
                b_copy = binary.copy()
                b_copy = b_copy[h1:h2, w1:w2]
                #cv.imshow('binary%s-%d' % (name, con), b_copy)
                dirs = 'car_characters\\%s' % (name)
                if not os.path.exists(dirs):  # 如果不存在路径，则创建这个路径
                    os.makedirs(dirs)
                path = dirs + '\\' + 'image_%d.jpg' % (con)
                cv.imwrite(path, b_copy)
                con += 1
                break

        if con==7:
            break

        # 1、前谷（前面的波谷）
        if pty[j] < 3 and begin == False:  # 前谷判断：像素数量<12
            last = pty[j]
            w = j

        # 2、字符开始（上升）
        elif last < 3 and pty[j] > 12:
            last = pty[j]
            w1 = j
            begin = True

        # 3、字符结束
        elif pty[j] < 5 and begin == True:
            begin = False
            last = pty[j]
            w2 = j
            width = w2 - w1
            # 3-1、分割并显示（排除过小情况）
            if 10 < width < WIDTH + 3:  # 要排除掉干扰，又不能过滤掉字符”1“
                b_copy = binary.copy()
                b_copy = b_copy[h1:h2, w1:w2]
                #cv.imshow('binary%s-%d' % (name, con), b_copy)
                dirs = 'car_characters\\%s' % (name)
                if not os.path.exists(dirs):  # 如果不存在路径，则创建这个路径
                    os.makedirs(dirs)
                path = dirs + '\\' + 'image_%d.jpg' % (con)
                cv.imwrite(path, b_copy)
                con += 1
                if con == 7:
                    break
            # 3-2、从多个贴合字符中提取单个字符
            elif width >= WIDTH + 3:
                # 统计贴合字符个数
                num = int(width / WIDTH + 0.5)  # 四舍五入
                for k in range(num):
                    # w1和w2坐标向后移（用w3、w4代替w1和w2）
                    w3 = w1 + k * WIDTH
                    w4 = w1 + (k + 1) * WIDTH
                    b_copy = binary.copy()
                    b_copy = b_copy[h1:h2, w3:w4]
                    #cv.imshow('binary%s-%d' % (name, con), b_copy)
                    dirs = 'car_characters\\%s' % (name)
                    if not os.path.exists(dirs):  # 如果不存在路径，则创建这个路径
                        os.makedirs(dirs)
                    path = dirs + '\\' + 'image_%d.jpg' % (con)
                    cv.imwrite(path, b_copy)
                    con += 1
                    if con == 7:
                        break

        if con==7:
            break

        # 4、分割尾部噪声（距离过远默认没有字符了）
        elif begin == False and (j - w2) > 30:
            break

    # 最后检查收尾情况
    if begin == True and con<=6:
        w2 = 295
        b_copy = binary.copy()
        b_copy = b_copy[h1:h2, w1:w2]
        #cv.imshow('binary%s-%d' % (name, con), b_copy)
        dirs = 'car_characters\\%s' % (name)
        if not os.path.exists(dirs):  # 如果不存在路径，则创建这个路径
            os.makedirs(dirs)
            path = dirs + '\\' + 'image_%d.jpg' % (con)
            cv.imwrite(path, b_copy)

# 二-7、分割车牌图像（根据直方图）
def Cut_Image(ptx, pty, binary):
    # 1、依次得到各行、列
    rows, cols = len(ptx), len(pty)

    # 2、横向分割：上下边框
    h1, h2 = Cut_X(ptx, rows)

    # 3、纵向分割：分割字符
    Cut_Y(pty, cols, h1, h2, binary)

# 一、形态学提取车牌
def Get_Licenses(image):
    # 1、转灰度图
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    #cv.imshow('gray', gray)

    # 2、顶帽运算
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (17, 17))
    tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    #cv.imshow('tophat', tophat)

    # 4、自适应二值化（阈值自己可调）
    ret, binary = cv.threshold(tophat, 75, 255, cv.THRESH_BINARY)
    #cv.imshow('binary', binary)

    # 5、开运算分割（纵向去噪，分隔）
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 15))
    Open = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    #cv.imshow('Open', Open)

    # 6、闭运算合并，把图像闭合、揉团，使图像区域化，便于找到车牌区域，进而得到轮廓
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (41, 15))
    close = cv.morphologyEx(Open, cv.MORPH_CLOSE, kernel)
    #cv.imshow('close', close)

    # 7、膨胀/腐蚀（去噪得到车牌区域）
    kernel_x = cv.getStructuringElement(cv.MORPH_RECT, (25, 7))
    kernel_y = cv.getStructuringElement(cv.MORPH_RECT, (1, 11))
    # 7-1、腐蚀、膨胀（去噪）
    erode_y = cv.morphologyEx(close, cv.MORPH_ERODE, kernel_y)
    #cv.imshow('erode_y', erode_y)
    dilate_y = cv.morphologyEx(erode_y, cv.MORPH_DILATE, kernel_y)
    #cv.imshow('dilate_y', dilate_y)
    # 7-1、膨胀、腐蚀（连接）（二次缝合）
    dilate_x = cv.morphologyEx(dilate_y, cv.MORPH_DILATE, kernel_x)
    #cv.imshow('dilate_x', dilate_x)
    erode_x = cv.morphologyEx(dilate_x, cv.MORPH_ERODE, kernel_x)
    #cv.imshow('erode_x', erode_x)

    # 8、腐蚀、膨胀：去噪
    kernel_e = cv.getStructuringElement(cv.MORPH_RECT, (25, 9))
    erode = cv.morphologyEx(erode_x, cv.MORPH_ERODE, kernel_e)
    #cv.imshow('erode', erode)
    kernel_d = cv.getStructuringElement(cv.MORPH_RECT, (25, 11))
    dilate = cv.morphologyEx(erode, cv.MORPH_DILATE, kernel_d)
    #cv.imshow('dilate', dilate)

    # 9、获取外轮廓
    img_copy = image.copy()
    # 9-1、得到轮廓
    contours, hierarchy = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 9-2、画出轮廓并显示
    cv.drawContours(img_copy, contours, -1, (255, 0, 255), 2)
    #cv.imshow('Contours', img_copy)

    # 10、遍历所有轮廓，找到车牌轮廓
    for contour in contours:
        # 10-1、得到矩形区域：左顶点坐标、宽和高
        rect = cv.boundingRect(contour)
        # 10-2、判断宽高比例是否符合车牌标准，截取符合图片
        if rect[2] > rect[3] * 3 and rect[2] < rect[3] * 7:
            # 截取车牌并显示
            #print(rect)
            image = image[(rect[1]):(rect[1] + rect[3]), (rect[0]):(rect[0] + rect[2])]  # 高，宽
            # 限制大小（按照比例限制）
            #image = Limit(image)
    return image

def con(img):          #增强对比度
    h, w, ch = img.shape
    src2 = np.zeros([h, w, ch], img.dtype)
    con = cv.addWeighted(img, 1.2, src2, 1 - 1.2, 0)
    #cv.imshow('con', con)
    return con
def rui(img):          #锐化
    fil = np.array([[-1, -1, -1], [-1, 9, -1], [-1, 1, -1]])
    res=cv.filter2D(img,-1,fil)
    #cv.imshow('res', res)
    return res
def get_BluePlate_bin(pai_src):
    # 锐化 取 channl_h
    pai_rui = rui(pai_src)          #锐化
    img_hsv = cv.cvtColor(pai_rui, cv.COLOR_BGR2HSV)  # 转换成HSV颜色空间
    h, s, v = cv.split(img_hsv)   # 分离 H、S、V 通道
    _, rui_otsu = cv.threshold(h, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # 二值化
    #cv.imshow('rui_otsu', rui_otsu)
    # 增强对比度 取 channl_s
    pai_con = con(pai_src)
    img_hsv = cv.cvtColor(pai_con, cv.COLOR_BGR2HSV)  # 转换成HSV颜色空间
    h, s, v = cv.split(img_hsv)   # 分离 H、S、V 通道
    _, con_otsu = cv.threshold(s, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # 二值化
    #cv.imshow('con_otsu', con_otsu)
    con_rui =rui_otsu & con_otsu  #与操作 清晰，去噪
    bin = cv.inRange(con_rui, 200, 255)  # 二值化
    #cv.imshow('bin', bin)
    return bin

# 二、直方图提取字符
def Get_Character(image):
    # 增强对比度并二值化
    binary=get_BluePlate_bin(image)
    # -------------------------------跳变次数去掉铆钉和边框----------------------------------
    LICENSE_HIGH, LICENSE_WIDTH = image.shape[:2]
    times_row = []  # 存储哪些行符合跳变次数的阈值
    for row in range(LICENSE_HIGH):  # 按行检测 白字黑底
        pc = 0
        for col in range(LICENSE_WIDTH):
            if col != LICENSE_WIDTH - 1:
                if binary[row][col + 1] != binary[row][col]:
                    pc = pc + 1
        times_row.append(pc)
    #print("每行的跳变次数:", times_row)
    # 找车牌的下边缘-从下往上扫描
    row_end = 0
    row_start = 0
    for row in range(LICENSE_HIGH - 2):
        if times_row[row] < 16:
            continue
        elif times_row[row + 1] < 16:
            continue
        elif times_row[row + 2] < 16:
            continue
        else:
            row_end = row + 2
    #print("row_end", row_end)
    # 找车牌的上边缘-从上往下扫描
    i = LICENSE_HIGH - 1
    row_num = []  # 记录row_start可能的位置
    while i > 1:
        if times_row[i] < 16:
            i = i - 1
            continue
        elif times_row[i - 1] < 16:
            i = i - 1
            continue
        elif times_row[i - 2] < 16:
            i = i - 1
            continue
        else:
            row_start = i - 2
            row_num.append(row_start)
            i = i - 1
    #print("row_num", row_num)
    # 确定row_start最终位置
    for i in range(len(row_num)):
        if i != len(row_num) - 1:
            if abs(row_num[i] - row_num[i + 1]) > 3:
                row_start = row_num[i]
    #print("row_start", row_start)
    times_col = [0]
    for col in range(LICENSE_WIDTH):
        pc = 0
        for row in range(LICENSE_HIGH):
            if row != LICENSE_HIGH - 1:
                if binary[row, col] != binary[row + 1, col]:
                    pc = pc + 1
        times_col.append(pc)
    #print("每列的跳变次数", times_col)
    # 找车牌的左右边缘-从左到右扫描
    col_start = 0
    col_end = 0
    for col in range(len(times_col)):
        if times_col[col] > 2:
            col_end = col
    #print('col_end', col_end)
    j = LICENSE_WIDTH - 1
    while j >= 0:
        if times_col[j] > 2:
            col_start = j
        j = j - 1
    #print('col_start', col_start)
    # 将车牌非字符区域变成纯黑色
    for i in range(LICENSE_HIGH):
        if i > row_end or i < row_start:
            binary[i] = 0
    for j in range(LICENSE_WIDTH):
        if j < col_start or j > col_end:
            binary[:, j] = 0
    # plate_binary = gray.copy()
    for i in range(LICENSE_WIDTH - 1, LICENSE_WIDTH):
        binary[:, i] = 0
    #cv.imshow("res2", binary)
    # 字符细化操作
    kernel = np.ones((2, 2), np.uint8)
    specify = cv.erode(binary,kernel, iterations=2)
    #cv.imshow("specify", specify)
    #4、膨胀（粘贴横向字符）
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 3))  # 横向连接字符
    dilate = cv.dilate(specify, kernel)
    #cv.imshow('dilate', dilate)
    # 5、统计各行各列白色像素个数（为了得到直方图横纵坐标）
    ptx, pty = White_Statistic(dilate)
    # 6、绘制直方图（横、纵）
    Draw_Hist(ptx, pty)
    # 7、分割（横、纵）（横向分割边框、纵向分割字符）
    Cut_Image(ptx, pty, binary)
    # cv.waitKey(0)

def getName(path):
    filepath = path
    filename = os.path.basename(filepath)  # 从全路径获取文件名
    countnum1 = 0
    countnum2 = 0
    countnum3 = 0
    #print(filename)
    for i in range(len(filename)):
        if(filename[i]=="-"):
            countnum1+=1
            if(countnum1==4):
                countnum1=i
                break
    for j in range(len(filename)):
        if(filename[j]=="-"):
            countnum2+=1
            if(countnum2==5):
                countnum2=j
                break
    for k in range(countnum1,len(filename)):
        if (filename[k] == "_"):
            countnum3 = k
            break
    filename1=filename[countnum1+1:countnum2]
    # filename2=filename[countnum1+1:countnum3]
    # filename3=filename[countnum3+1:countnum2]
    # return filename1,filename2,filename3
    return filename,filename1

def remove(name):
    path2 = 'car_characters/'+name
    if not os.path.exists(path2):
        return
    path3 = 'ture_car_characters'
    files22 = os.listdir(path2)
    num22 = len(files22)
    if (num22 == 7):
        for j in range(num22):
            if j == 0:
                path221 = path3 + '/' + name + '/zh/1'
                if not os.path.exists(path221):  # 如果不存在路径，则创建这个路径
                    os.makedirs(path221)
                shutil.copy(path2 + '/' + files22[j], path221 + '/')
            elif j > 0:
                path222 = path3 + '/' + name + '/letter/1'
                if not os.path.exists(path222):  # 如果不存在路径，则创建这个路径
                    os.makedirs(path222)
                shutil.copy(path2 + '/' + files22[j], path222 + '/')
    else:
        return

def run(fileName):
    global name
    name = ""
    # 1、获取路径
    #path1 = 'D:\MyCode\Coding\ccpd-base\\00-84_95-252&501_443&593-439&565_255&588_262&515_446&492-12_11_14_29_24_33_25-59-57.jpg'
    path1 = fileName
    #image=plate_locate.plate_locate(path1)
    name1,name = getName(path1)
    path2='ccpd-roii/'+name1
    # 2、获取图片
    img = cv.imread(path2)
    image = img.copy()
    if image is None:
        return
    #cv.imshow('image', image)
    # 3、提取车牌
    image = Get_Licenses(image)  # 形态学提取车牌
    # 4、提取字符
    Get_Character(image)
    remove(name)
    cv.waitKey(0)

# run()
#run("D:/MyCode/Coding/ccpd-car/1/013287835249-89_91-307&460_501&538-509&541_306&533_309&462_512&470-1_5_22_33_30_24_26-101-28.jpg")
import cv2
import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

model_filename = 'svm_model.pkl'
#加载SVM模型
with open(model_filename, 'rb') as f:
    svm_model = pickle.load(f)


def get_BlueImg_bin(img):
    # cv2.imshow('ori',img)
    # 掩膜：BGR通道，若像素B分量在 100~255 且 G分量在 0~190 且 G分量在 0~140 置255（白色） ，否则置0（黑色）
    mask_gbr=cv2.inRange(img,(120,0,0),(255,190,140))
    # cv2.imshow('mask_gbr',mask_gbr)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转换成 HSV 颜色空间
    h, s, v = cv2.split(img_hsv)                # 分离通道  色调(H)，饱和度(S)，明度(V)
    mask_s = cv2.inRange(s, 80, 255)                # 取饱和度通道进行掩膜得到二值图像
    # cv2.imshow('mask_s',mask_s)
    rgbs= mask_gbr & mask_s                # 与操作，两个二值图像都为白色才保留，否则置黑
    #cv2.imshow('rgbs',rgbs)
    # 核的横向分量大，使车牌数字尽量连在一起
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 3))
    img_rgbs_dilate = cv2.morphologyEx(rgbs,cv2.MORPH_CLOSE, kernel,iterations= 3)   # 膨胀 ，减小车牌空洞
    # cv2.imshow('img_rgbs_dilate',img_rgbs_dilate)

    return img_rgbs_dilate




def get_EdgeImg_bin(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    mask_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # 计算x方向梯度
    img_x = cv2.convertScaleAbs(mask_x)  # 取绝对值
    # kernel 横向分量大，使车牌数字尽量连在一起
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 3))
    closing_img = cv2.morphologyEx(img_x, cv2.MORPH_CLOSE, kernel, iterations=3)#闭运算

    img_edge = cv2.inRange(closing_img,120,255)
    # cv2.imshow('img_edge',img_edge)
    return img_edge

def get_first_roi(img,mask):#第一次筛选车牌区域
    contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for c in range(len(contours)):
        roi = get_any_roi(img,contours[c])
        if roi is not None:
            rois.append(roi)
    return rois

def get_second_roi(rois):#第二次筛选车牌区域
    t_rois =[]
    rrois = []
    for roi in rois:
        if(roi.shape[0]>0 and roi.shape[1]>0):
            tr_roi = cv2.cvtColor(roi.copy(), cv2.COLOR_BGR2GRAY)
            tr_roi = cv2.resize(tr_roi, (350,120)) #转换成和训练集同样尺寸
            tr_roi = np.array(tr_roi).flatten()
            t_rois.append(tr_roi)
            rrois.append(roi)
    # 获取距离超平面距离，1是所求车牌图片，距离越大，概率越大
    dest = svm_model.decision_function(t_rois)

    return rrois[np.argmax(np.array(dest))]

def get_any_roi(img,contour):
    roi = None
    min = 150 * 35
    max = 420 * 110
    # 获取轮廓的外接矩形
    rect = cv2.minAreaRect(contour)

    w, h = rect[1]
    if w < h:
        w, h = h, w
    angle = rect[2]
    area = w * h
    if min < area and area < max:
        wh = float(w)/ h
        if wh > 2 and wh < 5:
            # 画出来
            # cv2.drawContours(dst, [box1], 0, (0, 0, 255), 1)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            flag = False
            # print('first角度为:'+str(angle))
            if angle > 60:
                if angle < 87:
                    angle = -(90 - angle)
                    flag = True
            elif angle < 30:
                if angle > 3:
                    angle = angle
                    flag = True
            # print('second角度为:'+str(angle))
            if (flag):
                H,W = img.shape[:2]
                # 根据 ROI 倾斜角计算旋转矩阵，旋转中心就是 ROI 的中心
                M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])

                nW = int((H * sin) + (W * cos))
                nH = int((H * cos) + (W * sin))
                # print(h,w)
                # print(nH,nW)

                # 调整旋转矩阵
                M[0, 2] += (nW - W) / 2
                M[1, 2] += (nH - H) / 2

                # 旋转图像
                image = cv2.warpAffine(img, M, (nW, nH))
                # cv2.imshow('rotate',image)

                # 对box内坐标进行变换
                # print(box)
                box_new = cv2.transform(np.array([box]), M)[0].astype(int)
                # print(box_new)
                left_point_x = np.min(box_new[:, 0])
                right_point_x = np.max(box_new[:, 0])
                top_point_y = np.min(box_new[:, 1])
                bottom_point_y = np.max(box_new[:, 1])
                # print(left_point_x,right_point_x,top_point_y,bottom_point_y)
                # # 使用切片操作截取矩形部分图像
                roi = image[top_point_y:bottom_point_y, left_point_x:right_point_x]
            else:
                x, y, w, h = cv2.boundingRect(contour)
                roi = img[y:y + h, x:x + w]
    return roi

def roi_correct(img):
    if(img.shape[0]>0 and img.shape[1]>0):
        # 将图像转换成灰度图像

        # dst2 = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 对灰度图像进行边缘检测，以便霍夫变换可以检测边缘
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # cv2.imshow('edge',edges)

        # 对边缘图像应用霍夫变换来检测直线
        lines = cv2.HoughLines(edges, 2, np.pi / 180, 130)

        # 计算所有直线的平均角度
        angles = []
        if(lines is not None):
            for line in lines:
                for rho, theta in line:
                    if abs(theta-np.pi/2) < np.pi / 23:  # 小于8度
                        # print(theta)
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        # cv2.line(dst2, (x1, y1), (x2, y2), (0, 0, 255),1)
                        angles.append(theta)

            # 计算角度的平均值
            avg_angle = np.mean(angles)
            # print(avg_angle/np.pi)
            # cv2.imshow('img',img)
            # 使用平均角度旋转图像并矫正斜率
            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), (avg_angle-np.pi/2) * 180 / np.pi, 1)
            dst = cv2.warpAffine(img, M, (cols, rows))
            # cv2.imshow('img',dst2)
            return dst
        else:
            return img
    else:
        return None

# def train_ccpd():
#     path = 'ccpd-car'
#     dirs = os.listdir(path)
#     for i in dirs:
#         dirpath = os.path.join(path,i)
#         filedirs = os.listdir(dirpath)
#         todirpath = os.path.join('ccpd-plate-roiii', i)
#         for j in filedirs:
#             img = cv2.imread(str(os.path.join(dirpath,j)))
#             img_rgbs_dilate = get_BlueImg_bin(img)
#             edge_img = get_EdgeImg_bin(img)
#             hsv_rois = get_first_roi(img,img_rgbs_dilate)
#             edge_rois= get_first_roi(img,edge_img)
#             rois = []
#             if hsv_rois is not None and edge_rois is not None:
#                 rois = hsv_rois+edge_rois
#             elif hsv_rois is not None:
#                 rois = hsv_rois
#             elif edge_rois is not None:
#                 rois = edge_rois
#
#             if(len(rois)>0):
#                 roi = get_second_roi(rois)
#                 dst = roi_correct(roi)
#                 if(dst is not None):
#                     cv2.imwrite(os.path.join(todirpath,j), dst)

def train_ccpd():
    path = 'ccpd-car'
    dirs = os.listdir(path)
    for i in dirs:
        dirpath = os.path.join(path,i)
        filedirs = os.listdir(dirpath)
        todirpath = os.path.join('ccpd-plate-roiii', i)
        for j in filedirs:
            dst = plate_locate(str(os.path.join(dirpath,j)))
            if(dst is not None):
                cv2.imwrite(os.path.join(todirpath,j), dst)
    return

def plate_locate(filename):
    img = cv2.imread(filename)
    img_rgbs_dilate = get_BlueImg_bin(img)
    edge_img = get_EdgeImg_bin(img)
    hsv_rois = get_first_roi(img,img_rgbs_dilate)
    edge_rois = get_first_roi(img,edge_img)
    rois = []
    if hsv_rois is not None and edge_rois is not None:
        rois = hsv_rois+edge_rois
    elif hsv_rois is not None:
        rois = hsv_rois
    elif edge_rois is not None:
        rois = edge_rois
    dst = None
    if (len(rois) > 0):
        roi = get_second_roi(rois)
        dst = roi_correct(roi)
    return dst

# dst = plate_locate('ccpd-car/0/028125-86_90-193&505_470&600-477&583_200&603_196&522_473&502-0_0_10_9_25_33_29-129-80.jpg')
# cv2.imshow('dst',dst)
# cv2.waitKey(-1)







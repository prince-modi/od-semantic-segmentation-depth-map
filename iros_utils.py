import numpy as np
from ultralytics import YOLO
from depth import Depth
import glob,os
import pdb
import pandas as pd
import cv2
import math
import matplotlib.pyplot as plt
import time
from supervision.tools.detections import Detections, BoxAnnotator
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from panoptic import PanopticSeg

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from pickle import dump, load
from sklearn.metrics import mean_squared_error

pss=PanopticSeg()
dpt = Depth()

def depthwrapper():
    depth_img_folder = "/home/sumanraj/IROS_official/results_ABS"
    try:
        i = 0
        while True:
            t0 = time.time()
            frame_path=f"finale/{i}.jpg"
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, (960,720), interpolation=cv2.INTER_AREA)
            mask = pss.panoptic_pred(frame=frame,id=0)
            mask=mask[0].cpu().numpy()
            # print(mask[0].cpu().numpy().shape)
            depth = dpt.inference_depth(frame_path, save=False, show=False, send_depth=False)
            # print(depth.shape)
            ## need to change some things
            xyxy = [0,0,320,720]
            depth[mask!=0]=0
            # print(depth)
            
            free = {}
            for j in range(3):
                median,mean,test=depth_pixels_bbox(xyxy,depth)
                free[j]=mean
                xyxy[0]+=320
                xyxy[2]+=320
            i+=1
            maxy=min(free, key=free.get)
            print("Free space",maxy,free[maxy])
            # if maxy==0:
            #     depth = cv2.rectangle(frame, (0,0), (320,720), (255, 0, 0), thickness=3)
            # elif maxy==1:
            #     depth = cv2.rectangle(frame, (320,0), (640,720), (255, 0, 0), thickness=3)
            # elif maxy==2:
            #     depth = cv2.rectangle(frame, (640,0), (960,720), (255, 0, 0), thickness=3)
            cv2.imshow('depth', depth)
            cv2.imwrite(f"finale/depth_black_masks/{i}_maskless.png",depth)
            cv2.waitKey(1)
    except Exception as e:
        print(e)
        

def depth_pixels_bbox(xyxy=None, depth=None, csv=False, median=True, mean=True):
    if csv:
        #xyxy = [x1(topleft),y1(topleft),x2(bottomright),y2(bottomright)]
        xyxy = xyxy.split()
        xyxy = [xyxy[0][1:-3], xyxy[1][:-3], xyxy[2][0:-3], xyxy[3][0:-3]]
        xyxy = [int(i) for i in xyxy]
        tempx = (xyxy[2]-xyxy[0])/2
        tempy = (xyxy[3]-xyxy[1])/2
        print("Midpoint",tempx,tempy)
    # print(xyxy)
    xyxy = [int(i) for i in xyxy]
    bbox_pxls = depth[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
    median_dpt, mean_dpt = 0,0
    if median: median_dpt = np.median(bbox_pxls)  
    if mean: mean_dpt = np.average(bbox_pxls) 
    # print(depth.shape, bbox_pxls.shape)
    # print(np.sort(bbox_pxls.flatten()), bbox_pxls.flatten().shape)
    print(median_dpt, mean_dpt)

    return median_dpt, mean_dpt, bbox_pxls
     

def calculate_depth(vest_obj, obs_obj, depth):
    obs_detections = [obs_obj[0].boxes.xyxy.cpu().numpy(), obs_obj[0].boxes.conf.cpu().numpy(), obs_obj[0].boxes.cls.cpu().numpy().astype(int)] #boxes,conf,id
    vest_detections = [vest_obj[0].boxes.xyxy.cpu().numpy(), vest_obj[0].boxes.conf.cpu().numpy(), vest_obj[0].boxes.cls.cpu().numpy().astype(int)] #boxes,conf,id
    print('obs_detection:', obs_detections, '\n', 'vest_detection: ', vest_detections)
    obs_depth_arr = []
    for box in obs_detections[0]:
        print(box)
        _,mean_dpt,_ = depth_pixels_bbox(box, depth)
    #     obs_depth_arr.append(median_dpt)
    # obs_detections.append(obs_depth_arr)

    # _,vest_dpt,_ = depth_pixels_bbox(vest_detections[0], depth) if vest_detections[0] is not None else math.nan

    # print(obs_detections[-1], vest_dpt)
    return 1


def vip_vest_inter(vest_obj, obs_obj):
    if len(vest_obj) != 0:
        obs_detections = [obs_obj[0].boxes.xyxy.cpu().numpy(), obs_obj[0].boxes.conf.cpu().numpy(), obs_obj[0].boxes.cls.cpu().numpy().astype(int)] #boxes,conf,id
        vest_bbox = vest_obj[0].boxes.xyxy.cpu().numpy().squeeze()
        print(len(vest_bbox), len(obs_detections[0]))
    
        # print("chalna bhai")
        vest_bbox = [int(i) for i in vest_bbox]
        # print("chalna bhai pls")

        # point of interest vest_bbox[0],vest_bbox[2]
        for j,box in enumerate(obs_detections[0]):
            print("chalna bhai ESA KYA KARA")
            xtop = [i for i in range(vest_bbox[0]-10,vest_bbox[0]+10)]
            xbot = [i for i in range(vest_bbox[2]-10,vest_bbox[2]+10)]
            box = [int(i) for i in box]
            fail_box = box
            print(xtop,xbot,box)
            if box[0] in xtop and box[2] in xbot:
                if box[1] < vest_bbox[1] and box[3] > vest_bbox[3]:
                    print('returning logical output')
                    fail_box=box
                    return box
                else:
                    return fail_box
            if j == len(obs_detections[0])-1:
                return [0,0,0,0]  
            
    else:
        print('kuch toh gadbad kia')
        return [0,0,480,360]


def write_results(frame, frame_path, model,dpt_model):
    results = model(frame)
    # print(model.model.names)
    dixy = {}
    for j, result in enumerate(results):
        print(len(result))
        xyxy=result.boxes.xyxy.cpu().numpy()
        print(xyxy)
        depth=dpt_model.inference_depth(frame_path, save=True, show=False)
        print(depth)
        # depth_pixels_bbox(xyxy,depth)
        print('before probs')
        probs=result.boxes.conf.cpu().numpy()
        print(result.boxes.cls.cpu().numpy())
        class_id=result.boxes.cls.cpu().numpy().astype(int)
        class_name=[]
        print(1)
        for i in class_id:
            class_name.append(str(i)+"="+model.model.names[i])
            print(class_name)
        for k,temp in enumerate(result):
            #  dixy[f'{frame_path[24:]}_{k}']=abs_csv(show=True, real_inf=True)
            if class_name[k]=='0=person':
                dixy[f"{frame_path[29:]}_{k}"]={f"{k}":str({"bb":str([p for p in xyxy[k]]),"prob":probs[k],"name":class_name[k]})}

    # if 0:
    #     model = load(open('median_dpt.pkl', 'rb'))
    #     x_seq = np.linspace(-10,10)
    #     y = model.predict(x_seq.reshape(-1,1))
    #     plt.plot(x_seq,y,color="black")
    #     plt.show()
    #     {k:str({"bb":str([p for p in xyxy[k]]),"prob":probs[k],"name":class_name[k]})}
    #         #  print([i for i in xyxy[k]])
    with open("REV_results.txt",'a') as f:

        # f.write(f"{str(results)}\n")
        # print(f"{frame_path} \n bbox: {xyxy} \n probs: {probs} \n class_names: {class_name}\n \n")
        # f.write(f"{frame_path} \n bbox: {xyxy} \n probs: {probs} \n class_names: {class_name}\n \n")
        f.write(f'{str(dixy)}\n')


def lft(df):
    x1=np.array(df['Mean'])
    x1=x1.reshape(-1,1)
    reg1 = LinearRegression().fit(x1, df['ABS'])
    ypred1=reg1.predict(x1)
    x2=np.array(df['Median'])
    x2=x2.reshape(-1,1)
    reg2 = LinearRegression().fit(x2, df['ABS'])
    ypred2=reg2.predict(x2)
    # print(ypred1,ypred2)
    return ypred1, ypred2


def interpolation(df):
    X_seq = np.linspace(df['Median'].min(),df['Median'].max()).reshape(-1,1)
    coefs = np.polyfit(df['Median'].values.flatten(), df['ABS'].values.flatten(), 3)
    return X_seq, coefs


def polyreg(df):
    degree=2
    x1=np.array(df['Median'])
    x1=x1.reshape(-1,1)
    polyreg_median=make_pipeline(PolynomialFeatures(degree),LinearRegression())
    polyreg_median.fit(x1,df['abs'])
    x2=np.array(df['Mean'])
    x2=x2.reshape(-1,1)
    polyreg_mean=make_pipeline(PolynomialFeatures(degree),LinearRegression())
    polyreg_mean.fit(x2,df['abs'])
    dump(polyreg_median, open('median_dpt.pkl', 'wb'))
    dump(polyreg_mean, open('mean_dpt.pkl', 'wb'))
    return polyreg_median, polyreg_mean


def plot_abs_rev(df, rev_median_arr, rev_mean_arr, abs_arr, mode="log"):
    if mode=="log":
        rev_mean_arr =  np.log([int(i) for i in rev_mean_arr])
        rev_median_arr = np.log([int(i) for i in rev_median_arr])
        df['Median'] = rev_median_arr
        df['Mean'] = rev_mean_arr
    else:
        rev_mean_arr =  np.array([int(i) for i in rev_mean_arr])
        rev_median_arr = np.array([int(i) for i in rev_median_arr])
    
    #accesing values from loaded csv
    # rev_mean_arr = df['Mean'].to_numpy()
    # rev_median_arr = df['Median'].to_numpy()
    # abs_arr = df['abs'].tolist()

    # df1 = pd.DataFrame(list(zip(abs_arr, rev_mean_arr, rev_median_arr)), columns=['ABS', 'Mean', 'Median'])
    # print(df1, '\n\n')

    with open('abs_results.txt','w') as f:
        f.write(f'abs_arr: {abs_arr} \n\n')
        f.write(f'rev_median_arr: {rev_median_arr} \n\n')
        f.write(f'rev_mean_arr: {rev_mean_arr} \n\n')
        f.write(f'Dataframe:  {df}')

    # ypred1,ypred2=lft(df)
    # X_seq, coefs = interpolation(df)
    pmed,pmean = polyreg(df)
    plt.rcParams["figure.figsize"] = (10,6) 

    #RMSE calculation
    rev_median_pred = pmed.predict((rev_median_arr).reshape(-1,1))
    rev_mean_pred = pmean.predict((rev_mean_arr).reshape(-1,1))
    df['rev_median'] = rev_median_pred
    df['rev_mean'] = rev_mean_pred
    median_rmse = mean_squared_error(abs_arr, rev_median_pred, squared=False)
    mean_rmse = mean_squared_error(abs_arr, rev_mean_pred, squared=False)
    median_accuracy = np.mean(np.divide(rev_median_pred, abs_arr)) 
    mean_accuracy = np.mean(np.divide(rev_mean_pred, abs_arr))
    print(f"The RMSE of the median model is {median_rmse:.2f}, accuracy: {median_accuracy*100}")
    print(f"The RMSE of the mean model is {mean_rmse:.2f}, accuracy: {mean_accuracy*100}")

    #log data:
    # The RMSE of the median model is 1.41, accuracy: 112.3712163869083
    # The RMSE of the mean model is 1.44, accuracy: 113.9741021824302

    #normal data
    # The RMSE of the median model is 1.61, accuracy: 111.28014550817396
    # The RMSE of the mean model is 1.64, accuracy: 111.66929148655913

    #plotting
    plt.subplot(1,2,1)
    plt.scatter(rev_median_arr,abs_arr,c='r') #plot points
    X_seq = np.linspace(df['Median'].min(),df['Median'].max(),300).reshape(-1,1)
    plt.plot(X_seq,pmed.predict(X_seq),color="black") #plot curve
    # plt.plot(df['Median'],ypred2)
    plt.title('Median vs ABS') # set the title
    plt.xlabel('Median')
    plt.ylabel('ABS')
    plt.grid()

    plt.subplot(1,2,2)
    plt.scatter(rev_mean_arr,abs_arr,c='b') #plot points
    X_seq = np.linspace(df['Mean'].min(),df['Mean'].max(),300).reshape(-1,1)
    plt.plot(X_seq,pmean.predict(X_seq),color="black") #plot curve
    # plt.plot(df['Mean'],ypred2)
    plt.title('Mean vs ABS') # set the title
    plt.xlabel('Mean')
    plt.ylabel('ABS')
    plt.grid()

    # plt.savefig('/home/sumanraj/IROS_official/abs_vs_rev_depth.jpg')
    plt.savefig(f'/home/sumanraj/IROS_official/{mode}_abs_vs_rev_raw_depth.jpg')
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    return df



def abs_csv(show=False, real_inf=False, mode="poly"): #mode=log, poly
    filename = '/home/sumanraj/IROS_official/ABS_REV.xlsx'
    depth_img_folder = "/home/sumanraj/IROS_official/results_ABS"
    # depth_img_folder = "/home/sumanraj/IROS_official/output/depth"
    print(1)
    df = pd.read_excel(filename)
    df.reset_index() #make sure indices pair with number of rows
    print(df)
    
    #renaming image names 1 -> 01
    img = list(df['img'])
    for i in range(0, len(img)):
        img[i] = str(int(img[i])).zfill(2)
    
    df['img'] = img
    
    # initialising abs and rev arrays
    abs_arr = []
    rev_median_arr = []
    rev_mean_arr = []

    if real_inf:
        dpt = Depth()

    for index, row in df.iterrows():
        print(f'index: {index}')
        t0 = time.time()
        # if row['class'] == 'person':
        # if index == 19:
        #     break
        try: 
            if not pd.isna(row['class']):
                label, img_name, bbox, abs = row['class'], row['img'], row['bbox'], row['abs']
                # print(label, img_name, bbox, abs)
                t1 = time.time()
                #image processing
                if real_inf:
                    img_name = os.path.join(depth_img_folder, img_name + ".jpg") #jpg images
                    depth, raw_depth = dpt.inference_depth(img_name, save=False, show=False, send_depth=True)
                    t2 = time.time()
                    raw_depth = cv2.resize(raw_depth, (960, 720), interpolation = cv2.INTER_AREA)
                else:
                    img_name = os.path.join(depth_img_folder, img_name + "-dpt_swin2_large_384.png") #depth maps
                    depth = cv2.imread(img_name, 0)
                    depth = cv2.resize(depth, (960, 720), interpolation = cv2.INTER_AREA)
                
                t3 = time.time()
                median_dpt, mean_dpt, bbox = depth_pixels_bbox(bbox, depth, csv=True)

                if show:
                    cv2.imshow('depth', bbox)
                    cv2.waitKey(1)

                rev_median_arr.append(median_dpt)
                rev_mean_arr.append(mean_dpt)
                abs_arr.append(abs)
                print(f'index: {index}, loop time: {time.time()-t0}, pandas: {t1-t0}, inf: {t2-t1}, resize:{t3-t2}, maths: {time.time()-t3}\n\n')
                
        except Exception as e:
            print(e)
    
    #writing results into csv file
    # df1 = pd.DataFrame(list(zip(abs_arr, rev_mean_arr, rev_median_arr)), columns=['ABS', 'Mean', 'Median'])
    
    df['Median'] = (rev_median_arr)
    df['Mean'] = (rev_mean_arr)
     
    df = plot_abs_rev(df, rev_median_arr, rev_mean_arr, abs_arr, mode)
    df.to_excel(filename)
    print(df, '\n\n')    
   

if __name__=="__main__":
    # abs_csv(show=True, real_inf=True, mode="poly")

    # if 0:
    #     model = load(open('median_dpt.pkl', 'rb'))
    #     x_seq = np.linspace(-10,10)
    #     y = model.predict(x_seq.reshape(-1,1))
    #     plt.plot(x_seq,y,color="black")
    #     plt.show()

    depthwrapper()



####################################data points
#      Median          Mean    ABS
# 0   31384.0  39547.059532   1.25
# 1   23061.0  24064.190281   2.22
# 2    7688.5   7883.824291   4.34
# 3    3043.0   3342.738244   5.21
# 4    1774.0   2542.510706   7.83
# 5    1608.5   2214.573455   8.90
# 6    1043.5   1685.800058  10.27
# 7    4367.5   4763.027160  11.21
# 8   53110.0  38386.885390   1.00
# 9   17979.0  17369.893176   2.07
# 10  10007.0  10398.674792   3.00
# 11  20551.0  19545.704898   4.07
# 12  19310.0  18890.725937   5.22
# 13  19070.0  18925.306540   6.35
# 14  16350.0  16172.051009   7.50
# 15  16424.0  16200.055125   8.72
# 16  20862.5  20872.335443   9.88
# 17  10386.0  10551.226667  11.11
# 18  18015.0  17758.473684  12.30
# 19  51475.5  36468.975211   0.70
# 20   7491.0   8169.055059   1.50
# 21  21090.0  24879.926603   1.08
# 22  62376.0  61501.341022   1.70
# 23  37345.5  37346.317977   2.37
# 24  32656.0  29786.080973   2.05
# 25  14312.0  20072.523158   4.00
# 26  19515.5  20081.530398   5.74
# 27  47185.0  37058.573618   1.40
# 28  14223.0  15369.013396   3.67
# 29   7027.0   7005.298035   6.47
# 30   8556.5   9700.251152   9.45
# 31  20621.0  25001.975238   1.95
# 32  22871.0  22533.085661   3.39
# 33  15695.0  16706.093835   6.49
# 34  20157.0  21980.382797   3.23
# 35  16563.5  16991.102176   7.85
# 36  24637.0  25742.601822   2.81
# 37  13388.0  14816.965493   4.52
# 38  33891.0  30098.469763   2.60
# 39  21403.0  20761.861026   4.65
# 40  24408.0  25569.291152   6.93
# 41  13931.0  22204.496880   1.90
# 42  47337.5  37699.102578   5.70
# 43  36171.0  36212.489008   3.28
# 44   5942.0  13663.183025   2.78
# 45  20281.5  23716.175225   6.66
# 46  38734.0  36389.333193   2.67
# 47  56361.0  53987.960107   1.05
# 48   9307.0  15794.394760   0.99
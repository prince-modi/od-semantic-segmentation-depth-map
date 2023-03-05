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
from supervision.draw.color import ColorPalette
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
        # print("Midpoint",tempx,tempy)
    # print(xyxy)
    xyxy = [int(i) for i in xyxy]
    bbox_pxls = depth[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
    median_dpt, mean_dpt = 0,0
    # if median: median_dpt = np.median(bbox_pxls)
    if median: median_dpt = np.percentile(bbox_pxls, 95)  
    if mean: mean_dpt = np.average(bbox_pxls) 
    # print(depth.shape, bbox_pxls.shape)
    # print(np.sort(bbox_pxls.flatten()), bbox_pxls.flatten().shape)
    # print(median_dpt, mean_dpt)

    return median_dpt, mean_dpt, bbox_pxls
     

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
    print("SAVED models!!")
    return polyreg_median, polyreg_mean


def calculate_depth(vest_obj, obs_obj, labels, dist, depth):
    #data preparation
    track_results = []
    for obj in [vest_obj, obs_obj]:
        boxes = obj[0].boxes.xyxy.cpu().numpy().tolist()  
        conf = obj[0].boxes.conf.cpu().numpy().tolist() 
        cls_id = obj[0].boxes.cls.cpu().numpy().astype(int).tolist()  
        # boxes = obs_obj[0].boxes.xyxy.cpu().numpy().tolist() + vest_obj[0].boxes.xyxy.cpu().numpy().tolist()
        # conf = obs_obj[0].boxes.conf.cpu().numpy().tolist() + vest_obj[0].boxes.conf.cpu().numpy().tolist()
        # cls_id = obs_obj[0].boxes.cls.cpu().numpy().astype(int).tolist() + vest_obj[0].boxes.cls.cpu().numpy().astype(int).tolist()
        depth_arr = []
        dist_arr = []
        detections = [boxes, conf, cls_id, labels]
        
        #for yolo & vest
        for i, box in enumerate(detections[0]):
            # print(box)
            if len(box)!=0:
                median_dpt,_,_ = depth_pixels_bbox(box, depth)
                depth_arr.append(median_dpt)
                pred_dist = dist.predict(np.array(median_dpt).reshape(-1, 1))
                dist_arr.append(round(pred_dist.item(), 2))
            else:
                depth_arr.append(math.nan)
                dist_arr.append(math.nan) 

        detections.append(depth_arr)
        detections.append(dist_arr)
        track_results.append(detections)

        # for x in detections:
        #     print(x)

   

    # for i in range(len(track_results[0][-1])):
    #     print(str(track_results[0][-1][i]))

    return detections


def box_annotator(frame, vest_detections, obs_detections, yolo_classes_id_dict, vest_class_id_dict, track_results):
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=2, text_thickness=1, text_scale=0.75, text_padding=3)
    # print(yolo_classes_id_dict, vest_class_id_dict)

    print('vest results:', track_results[0][-1])
    print('obs results:', track_results[1][-1])
    
    if len(vest_detections.tracker_id) != 0:
        labels1 = [
            f"#{tracker_id} {vest_class_id_dict[class_id]} "  #tracker_id 
            # f"#{random.randint(1,30)} vest {confidence:0.2f}"
            for _, _, class_id, tracker_id #confidence
            in vest_detections
        ]
        print('vest label:', labels1)
        for i, x in enumerate(labels1):
            print(i,x)
            x = x + str(track_results[0][-1][i])
            print(i,x)
        # print(labels1)
        frame = box_annotator.annotate(frame=frame, detections=vest_detections, labels=labels1)
           
    if len(obs_detections.tracker_id) != 0:
        labels2 = [
            f"#{tracker_id} {yolo_classes_id_dict[class_id]} "  #tracker_id 
            # f"#{random.randint(1,30)} vest {confidence:0.2f}"
            for _, _, class_id, tracker_id #confidence
            in obs_detections
        ]
        print('obs label:', labels2)
        # for i, x in enumerate(labels2):
        #     x = x + str(track_results[1][-1][i])
        frame = box_annotator.annotate(frame=frame, detections=obs_detections, labels=labels2)



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
    filename = '/home/sumanraj/IROS_official/New_ABS_REV.xlsx'
    depth_img_folder = "/home/sumanraj/IROS_official/results_ABS_2/depth" #real_inf=False
    if real_inf:
        depth_img_folder = "/home/sumanraj/IROS_official/ABS_calculation_dataset" #real_inf=True
    # depth_img_folder = "/home/sumanraj/IROS_official/output/depth"
    print(1)
    df = pd.read_excel(filename)
    df.reset_index() #make sure indices pair with number of rows
    # print(df)
    
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

    #filtering conditions
    # df = df[df['abs'] <= 6]
    df = df[(df['class'] == 'person')] #| (df['class'] == 'bench')
    print(df)

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
                    print('real_inf:', depth)
                    t2 = time.time()
                    raw_depth = cv2.resize(raw_depth, (960, 720), interpolation = cv2.INTER_AREA)
                else:
                    img_name = os.path.join(depth_img_folder, img_name + "-dpt_beit_large_512.png") #depth maps "-dpt_swin2_large_384.png"
                    depth = cv2.imread(img_name, 0)
                    print('stored img inf:', depth)
                    depth = cv2.resize(depth, (960, 720), interpolation = cv2.INTER_AREA)
                    t2 = time.time()
                
                t3 = time.time()
                median_dpt, mean_dpt, bbox = depth_pixels_bbox(bbox, depth, csv=True)

                if show:
                    cv2.imshow('depth', bbox)
                    cv2.waitKey(1)

                if real_inf:
                    rev_median_arr.append(median_dpt) #int(median_dpt/65535 * 255)
                    rev_mean_arr.append(mean_dpt) #int(mean_dpt/65535 * 255)
                else:
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
    # df.to_excel(filename)
    print(df[['abs', 'rev_median', 'rev_mean', 'Median', 'Mean']], '\n\n')    
     

if __name__=="__main__":
    abs_csv(show=True, real_inf=True, mode="poly")

    # if 0:
    #     model = load(open('median_dpt.pkl', 'rb'))
    #     x_seq = np.linspace(-10,10)
    #     y = model.predict(x_seq.reshape(-1,1))
    #     plt.plot(x_seq,y,color="black")
    #     plt.show()

    # depthwrapper()



####################################data points
# SAVED models!!
# The RMSE of the median model is 0.96, accuracy: 106.52966662598725
# The RMSE of the mean model is 1.02, accuracy: 108.21938774071845 
#       abs  rev_median   rev_mean    Median          Mean
# 0    1.94    1.761903   1.926713  61773.00  48706.596377
# 1    3.62    4.179461   4.192681  35002.00  28683.403200
# 2    3.44    3.071391   3.078212  40734.00  33116.529098
# 3    6.07    7.095379   7.325111  24343.00  20136.916620
# 4    0.80    1.765237   1.823533  61854.00  43835.961926
# 5    1.30    1.786731   1.812932  62343.00  45663.508565
# 6    1.90    1.727593   1.903590  60823.00  48327.709347
# 7    2.50    1.677624   1.846742  57265.00  43037.170023
# 8    3.10    1.982121   1.986888  49716.00  40608.045259
# 9    3.80    2.746105   2.611969  42833.00  35565.900694
# 10   4.60    3.496945   3.283816  38330.00  32183.149667
# 11   5.40    4.892674   4.569296  31982.00  27435.102508
# 12   6.00    6.916095   6.735412  24898.00  21507.051622
# 13   6.70    7.367122   7.187546  23519.00  20449.990019
# 14   7.20    7.457274   7.517831  23250.00  19704.327418
# 15   8.10    8.267072   7.962149  20922.00  18735.251984
# 16   8.70    9.090641   9.137890  18697.00  16328.579030
# 17   9.30    9.159769   9.042866  18516.00  16515.096104
# 18  10.00    8.551000   8.231291  20140.95  18165.714002
# 19  10.50    9.776034   9.859917  16938.90  14945.795767
# 20  11.30   10.483832  10.723491  15198.60  13370.599730
# 21   1.05    1.718182   2.052970  54711.00  39840.281572
# 22  12.40   10.377836  10.389323  15454.10  13970.676736
# 23   2.30    1.678141   1.820009  57136.00  46145.723549
# 24   3.60    3.471360   3.407725  38466.00  31652.875548
# 25   5.50    6.150048   6.057477  27384.00  23189.271130
# 26   1.30    1.829610   1.833060  63192.00  46693.107057
# 27   3.00    4.268345   4.101702  34604.00  28999.821636
# 28   6.50    9.175473   9.502235  18475.00  15622.923113 



##########################################
# SAVED models!!
# The RMSE of the median model is 0.95, accuracy: 106.53904597573177
# The RMSE of the mean model is 1.02, accuracy: 108.69709236125478
#       abs  rev_median   rev_mean  Median        Mean
# 0    1.94    1.770392   1.914263   241.0  189.751927
# 1    3.62    4.185342   4.233384   136.0  111.527890
# 2    3.44    3.046457   3.125468   159.0  128.864915
# 3    6.07    7.064081   7.342645    95.0   78.136708
# 4    0.80    1.770392   1.827909   241.0  170.729082
# 5    1.30    1.793596   1.810922   243.0  177.870484
# 6    1.90    1.731744   1.899260   237.0  188.278691
# 7    2.50    1.677969   1.852618   223.0  167.611391
# 8    3.10    1.969833   1.989491   194.0  158.117659
# 9    3.80    2.730521   2.630615   167.0  138.425135
# 10   4.60    3.499585   3.296582   149.0  125.206643
# 11   5.40    4.915364   4.623148   124.0  106.639600
# 12   6.00    6.898431   6.790209    97.0   83.525759
# 13   6.70    7.403142   7.229834    91.0   79.371092
# 14   7.20    7.489525   7.571753    90.0   76.466726
# 15   8.10    8.296069   8.043913    81.0   72.680195
# 16   8.70    9.056976   9.174245    73.0   63.279070
# 17   9.30    9.155000   9.044004    72.0   64.027013
# 18  10.00    8.576558   8.286964    78.0   70.469464
# 19  10.50    9.756726   9.980087    66.0   57.883733
# 20  11.30   10.488166  10.827759    59.0   51.719568
# 21   1.05    1.717169   2.056030   213.0  155.128090
# 22  12.40   10.381735  10.398694    60.0   54.087153
# 23   2.30    1.677969   1.816526   223.0  179.749016
# 24   3.60    3.451362   3.416468   150.0  123.151355
# 25   5.50    6.185021   6.065600   106.0   90.092629
# 26   1.30    1.833254   1.826778   246.0  181.906417
# 27   3.00    4.242620   4.158917   135.0  112.791419
# 28   6.50    9.155000   9.571937    72.0   60.510252 




# index: 28
# real_inf: [[13058 12989 12942 ... 13024 11051  8745]
#  [13022 12971 12933 ... 12699 11940 10970]
#  [12975 12961 12940 ... 12502 12752 12950]
#  ...
#  [43292 43296 43274 ... 41684 41890 42088]
#  [42598 43060 43435 ... 41868 41998 42051]
#  [41521 42664 43612 ... 41897 41955 41984]]


# index: 28
# stored img inf: [[ 51  50  50 ...  50  43  34]
#  [ 50  50  50 ...  49  46  42]
#  [ 50  50  50 ...  48  49  50]
#  ...
#  [169 169 169 ... 162 163 164]
#  [166 168 169 ... 163 164 164]
#  [162 166 170 ... 163 163 164]]
import numpy as np
import cv2
def check_blocker(img_number,frame,track_results):
    track_results = np.array(track_results)
    if(track_results.size==0):
        free_coordinates=[(0,320),(320,640),(640,960)]
        width_fragments = [320,320,320]
        width_coverage = [320,320,320]
        spanning_width=[960]
        partition_fragment_dict = {0:([320],[(0,320)]),1:([320],[(320,640)]),2:([320],[(640,960)])}
    else:
        obs_bbox_arr=track_results[:,0]
        #obs_bbox_arr = np.array(track_results[1][0])
        #obs_bbox_arr = obs_results[0].boxes.xyxy.cpu().numpy()
        size = (720,960)
        #obs_bbox_arr =np.array([[5,10,40,50],[60,70,80,90],[10,45,30,80],[900,50,1000,70]])
        print(obs_bbox_arr)
        obs_bbox_arr = np.sort(obs_bbox_arr,axis =0)
        print(obs_bbox_arr)
        img = np.zeros(shape=(720,960), dtype="int")
        print(img)
        start = 0
        end = 960
        free_coordinates = []
        width_fragments = []
        width_coverage = []
        spanning_width=[]
        partition_fragment_dict = dict()
        n = 3
        if obs_bbox_arr[0][0] > start:
            free_tuple = (start,obs_bbox_arr[0][0])
            free_coordinates.append(free_tuple)
        #obs_bbox_arr = obs_bbox_arr.sort()
        temp_end = obs_bbox_arr[0][2]
        for i in range(len(obs_bbox_arr)-1): 
            if(obs_bbox_arr[i][2] > temp_end):
                temp_end =obs_bbox_arr[i][2]
            if obs_bbox_arr[i+1][0] > temp_end:
                free_tuple = (temp_end, obs_bbox_arr[i+1][0])
                free_coordinates.append(free_tuple)
            else:
                continue
        if obs_bbox_arr[-1][2] < end:
            if(obs_bbox_arr[-1][2] > temp_end):
                temp_end = obs_bbox_arr[-1][2]
            free_tuple = (temp_end,end)
            free_coordinates.append(free_tuple)

        print(free_coordinates)
        # for i in free_coordinates:
        #     cv2.line(frame, (int(i[0]),int(960/2)), (int(i[1]),int(960/2)), color = (0,255,0), thickness = 3)
        cv2.imwrite(f'outputRuchi/width/{img_number}.jpg', frame) 
        for i in range(n):
            width = 0
            start = i*(960/n)
            end = start + (960/n) 
            coords_list=[]
            frags_width=[]
            print(start,end)
            for j in range(len(free_coordinates)):
                if free_coordinates[j][0] >= start and free_coordinates[j][1] <= end:
                    fragment = free_coordinates[j][1] - free_coordinates[j][0] 
                    frags_width.append(fragment)
                    coords_list.append(free_coordinates[j])
                elif free_coordinates[j][0] >= start and free_coordinates[j][0] < end and free_coordinates[j][1] > end:
                    fragment = end - free_coordinates[j][0] 
                    frags_width.append(fragment)
                    coords_list.append((free_coordinates[j][0],end))
                    spanning_width.append(free_coordinates[j][1] - free_coordinates[j][0] )
                elif free_coordinates[j][0] < start and free_coordinates[j][1] <= end and free_coordinates[j][1] > start: 
                    fragment = free_coordinates[j][1] - start 
                    frags_width.append(fragment)
                    coords_list.append((start,free_coordinates[j][1]))
                    spanning_width.append(free_coordinates[j][1] - free_coordinates[j][0] )
                elif free_coordinates[j][0] < start and free_coordinates[j][1] > end:
                    fragment = end-start
                    frags_width.append(fragment)
                    coords_list.append((start,end))                   
                else: 
                    continue

                # print(free_coordinates[j])
                width += fragment
                width_fragments.append(fragment)       
            width_coverage.append(width)
            partition_fragment_dict[i]=(frags_width,coords_list)

    print(width_fragments)
    print(width_coverage)
    print(spanning_width)
    print(partition_fragment_dict)
    return partition_fragment_dict,width_coverage
    ##parttion dict = tuple (fragwidth array ,co-ord array)

#check_blocker(f,1)   
  

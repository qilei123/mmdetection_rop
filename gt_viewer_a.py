import os
import json
import cv2

save_folder = 'gt_viewer_a'
if not os.path.exists(save_folder):
    os.makedirs(save_folder+'/train')
    os.makedirs(save_folder+'/val')

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

def show_gts(anno_dir,t_v_folder):


    gt_json = json.load(open(anno_dir))

    images_dict = dict()

    for img in gt_json['images']:
        images_dict[img['id']] = img

    annos_by_image_id = dict()
    for anno in gt_json['annotations']:
        if not annos_by_image_id.has_key(anno['image_id']):
            annos_by_image_id[anno['image_id']]=[]
        annos_by_image_id[anno['image_id']].append(anno)

    for key in annos_by_image_id.keys():
        if True or (not ('train' in os.path.basename(images_dict[key]['file_name']) or 'val' in os.path.basename(images_dict[key]['file_name']))):
            image = cv2.imread(t_v_folder+'2014/'+images_dict[key]['file_name'])
            for anno in annos_by_image_id[key]:
                bbox = [int(anno['bbox'][0]),int(anno['bbox'][1]),int(anno['bbox'][2]),int(anno['bbox'][3])]
                cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,255,0),5)
                cv2.putText(image,str(anno['category_id']),(bbox[0]+bbox[2],bbox[1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
                polygon = anno['segmentation'][0]
                for i in range(len(polygon)/2-1):
                    cv2.line(image,(polygon[i*2],polygon[i*2+1]),(polygon[(i+1)*2],polygon[(i+1)*2+1]),(255,0,0),2)
                    #print(polygon[i*2],polygon[i*2+1])
            
            cv2.imwrite(save_folder+'/'+t_v_folder+'/'+os.path.basename(images_dict[key]['file_name']),image)
            print(images_dict[key]['file_name'])
    

def show_gt_with_result(anno_gt_dir,anno_result_dir,t_v_folder):
    missed =[0,0,0,0]
    totals = [0,0,0,0]
    results_json = json.load(open(anno_result_dir))

    result_annos_by_image_id = dict()
    for anno in results_json:
        if not anno['image_id'] in result_annos_by_image_id:
            result_annos_by_image_id[anno['image_id']]=[]
        result_annos_by_image_id[anno['image_id']].append(anno)

    gt_json = json.load(open(anno_gt_dir))

    images_dict = dict()

    for img in gt_json['images']:
        images_dict[img['id']] = img

    annos_by_image_id = dict()
    for anno in gt_json['annotations']:
        if not anno['image_id'] in annos_by_image_id:
            annos_by_image_id[anno['image_id']]=[]
        annos_by_image_id[anno['image_id']].append(anno)

    for key in annos_by_image_id.keys():
        #if True or (not ('train' in os.path.basename(images_dict[key]['file_name']) or 'val' in os.path.basename(images_dict[key]['file_name']))):
        #image = cv2.imread(t_v_folder+'2014/'+images_dict[key]['file_name'])
        '''
        for anno in annos_by_image_id[key]:
            bbox = [int(anno['bbox'][0]),int(anno['bbox'][1]),int(anno['bbox'][2]),int(anno['bbox'][3])]
            cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,255,0),5)
            cv2.putText(image,str(anno['category_id']),(bbox[0]+bbox[2],bbox[1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
            polygon = anno['segmentation'][0]
            #for i in range(len(polygon)/2-1):
            #    cv2.line(image,(polygon[i*2],polygon[i*2+1]),(polygon[(i+1)*2],polygon[(i+1)*2+1]),(255,0,0),2)
                #print(polygon[i*2],polygon[i*2+1])
        
        for anno in result_annos_by_image_id[key]:
            bbox = [int(anno['bbox'][0]),int(anno['bbox'][1]),int(anno['bbox'][2]),int(anno['bbox'][3])]
            cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(255,0,0),5)
            cv2.putText(image,str(anno['category_id']),(bbox[0]+bbox[2],bbox[1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
                #polygon = anno['segmentation'][0]            
        '''
        for anno_m in annos_by_image_id[key]:
            
            box_m = [int(anno_m['bbox'][0]),int(anno_m['bbox'][1]),int(anno_m['bbox'][0]+anno_m['bbox'][2]),int(anno_m['bbox'][1]+anno_m['bbox'][3])]
            Targeted = False
            if key in result_annos_by_image_id:
                for anno_n in result_annos_by_image_id[key]:
                    box_n = [int(anno_n['bbox'][0]),int(anno_n['bbox'][1]),int(anno_n['bbox'][0]+anno_n['bbox'][2]),int(anno_n['bbox'][1]+anno_n['bbox'][3])]
                    if anno_m['category_id']==anno_n['category_id'] and anno_n['score']>0.05:
                        iou_rate = bb_intersection_over_union(box_m,box_n)
                        if iou_rate>0.2:
                            Targeted=True
                            break
            if not Targeted:
                #cv2.rectangle(image,(box_m[0],box_m[1]),(box_m[2],box_m[3]),(0,0,255),5)
                #cv2.putText(image,str(anno_m['category_id']),(box_m[2],box_m[1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
                missed[int(anno_m['category_id'])-1]+=1
            else:
                #cv2.rectangle(image,(box_m[0],box_m[1]),(box_m[2],box_m[3]),(0,255,0),5)
                #cv2.putText(image,str(anno_m['category_id']),(box_m[2],box_m[1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)  
                pass   
            totals[int(anno_m['category_id'])-1]+=1                           
        #h = image.shape[0]
        #w = image.shape[1]
        #cv2.imshow('test',cv2.resize(image,(int(w*0.4),int(h*0.4))))
        #cv2.waitKey(0)
            #cv2.imwrite(save_folder+'/'+t_v_folder+'/'+os.path.basename(images_dict[key]['file_name']),image)
            #print(images_dict[key]['file_name'])
    print('----------------')
    print(missed)
    print(totals)
#show_gts('annotations/merged_reid_new_instances_train2014.json','train')
#show_gts('annotations/merged_reid_new_instances_val2014.json','val')

for i in range(8,15):
    print(i)
    show_gt_with_result('annotations/instances_val2014.json.original.opt_v2.hflip','work_dirs/faster_rcnn_r50_fpn_1x_with_focal_loss_smallset_advance_optdataset4_head_v1_with_pseudo_gt_v1_with_nms/e'+str(i)+'_results.pkl.json','val')
    show_gt_with_result('annotations/instances_val2014.json.original.opt_v2.hflip','work_dirs/faster_rcnn_r50_fpn_1x_with_focal_loss_smallset_advance_optdataset4_head_v1_with_pseudo_gt_v1_with_nms_2/e'+str(i)+'_results.pkl.json','val')
    show_gt_with_result('annotations/instances_val2014.json.original.opt_v2.hflip','work_dirs/faster_rcnn_r50_fpn_1x_with_focal_loss_smallset_advance_optdataset4_head_v1_with_pseudo_gt_v0/e'+str(i)+'_results.pkl.json','val')
    show_gt_with_result('annotations/instances_val2014.json.original.opt_v2.hflip','work_dirs/faster_rcnn_r50_fpn_1x_with_focal_loss_smallset_advance_optdataset4_deephead_v1/e'+str(i)+'_results.pkl.json','val')
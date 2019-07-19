import numpy as np
import matplotlib.pyplot as plt
import time

def parse_loss_record(loss_record_dir):
    loss_file = open(loss_record_dir)
    loss_records = dict()
    count=0
    for line in loss_file:
        if '- INFO - Epoch' in line:
            count+=1
            line_eles = line.split(',')
            for line_ele in line_eles:
                if 'loss' in line_ele:
                    line_ele.replace(' ','')
                    loss_name,loss = line_ele.split(':')
                    loss=float(loss)
                    if not loss_name in loss_records:
                        loss_records[loss_name]=[]
                        
                    loss_records[loss_name].append(loss)

    return count,loss_records


def draw_loss_plot(count,loss_records):
    plt.figure()
    for key in loss_records:
        x = range(count)
        y = loss_records[key]
        te, = plt.plot(x,y)
        te.set_label(key)
        plt.legend()
    plt.xlabel("count")
    plt.ylabel("loss")
    plt.title("A simple loss plot")
    #plt.savefig('easyplot.jpg')
    plt.show()

def start_loss_plot_server(loss_record_dir,updata_time):
    while True:
        time.sleep(updata_time)
        #loss_record_dir='/data0/qilei_chen/AI_EYE/BostonAI4DB7/work_dirs/faster_rcnn_r50_fpn_1x_2000_v2/20190719_114750.log'
        count,loss_records = parse_loss_record(loss_record_dir)
        draw_loss_plot(count,loss_records)

loss_record_dir='/data0/qilei_chen/AI_EYE/BostonAI4DB7/work_dirs/faster_rcnn_r50_fpn_1x_with_focal_loss_smallset_advance_optdataset4_head_v1_second_round_v2/20190717_022640.log'
count,loss_records = parse_loss_record(loss_record_dir)
draw_loss_plot(count,loss_records)
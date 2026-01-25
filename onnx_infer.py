import argparse
import os
import cv2
import numpy as np
import onnxruntime as ort
import torch
import yaml
from pathlib import Path
from moudle.Yolov8_onnx import YOLOv8

# 定義支援的副檔名
img_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
vid_formats = ['.mp4', '.avi', '.mov', '.mkv']

def get_source_file(source_path):
    """
    獲取來源檔案清單
    :param source_path: 資料夾或檔案路徑
    :param mode: 過濾模式，可選 'all', 'image', 'video'
    """
    p = Path(source_path).resolve()

    # 檢查路徑是否存在
    if not p.exists():
        print(f"Error: Path {source_path} does not exist.")
        return []

    files = []
    if p.is_dir():
        # 如果是資料夾，使用 rglob ('**/*') 進行遞迴取檔
        print(f"Scanning directory: {source_path}")
        for file in p.rglob('*'):
            if file.suffix.lower() in img_formats + vid_formats:
                files.append(file)
    elif p.is_file():
        # 如果是單一檔案，判斷副檔名
        if p.suffix.lower() in img_formats + vid_formats:
            files.append(p)
        else:
            print(f"Unsupported file format: {p.suffix}")
    else:
        print(f"Source path does not exist: {source_path}")
        
    return files



def parse_arguments():
    """
    解析命令列參數以配置 YOLOv8 ONNX 推論環境。

    此函式定義了模型路徑、資料來源及推論超參數。設定完成後，
    可透過傳回的 Namespace 物件存取各項設定。

    Args:
        --model (str): ONNX 模型檔案路徑。預設為 "yolov8n.onnx"。
        --img (str): 單張輸入影像的路徑。
        --img_folder (str): 包含多張待處理影像的資料夾路徑。
        --source (str): 通用輸入來源 (可為影像、資料夾、影片或串流位址)。
        --cfg (str): 模型配置檔 (.yaml) 路徑。
        --conf_thres (float): 信心度閾值 (Confidence Threshold)，預設為 0.5。
        --iou_thres (float): 非極大值抑制 (NMS) 的 IoU 閾值，預設為 0.5。
        --show (bool): 是否啟用結果視窗顯示。
        --task (str): 指定模型任務類型 (例如: 'detect', 'segment', 'classify')。

    Returns:
        argparse.Namespace: 包含所有解析參數的物件，可透過屬性方式存取（例如 args.model）。
    """
    parser = argparse.ArgumentParser(description="YOLOv8 ONNX Inference Script")
    parser.add_argument("--model", type=str,
                        default="yolov8n.onnx", help="Input your ONNX model.")
    parser.add_argument("--source", type=str,default=" ", help="Path to input source.")
    parser.add_argument('--cfg', type=str,default=" ", help='Path to config.')
    parser.add_argument("--conf_thres", type=float,
                        default=0.5, help="Confidence threshold")
    parser.add_argument("--iou_thres", type=float,
                        default=0.5, help="NMS IoU threshold")
    parser.add_argument("--show", type=bool,
                        default=False, help="show image switch")
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode switch.')
    return parser.parse_args()

if __name__ == "__main__":

    arge = parse_arguments()
    debug_mode = arge.debug
    # init onnx detector
    onnx_detector = YOLOv8(onnx_model=arge.model,yaml_file=arge.cfg,confidence_thres=arge.conf_thres,iou_thres=arge.iou_thres, debug=arge.debug)
    # get file list
    source_list = get_source_file(arge.source)

    for source in source_list:
        if source.suffix.lower() in img_formats:
            img_media = cv2.imread(source)
            if debug_mode:
                print("image shape:{}".format(img_media.shape))
            img_media_copy = img_media
            results = onnx_detector.detect(img_media)
            if debug_mode:
                print("result:{}".format(results))
                print("number of detections:{}".format(len(results)))
            for result in results:
                box,conf,cls = result
                x,y,w,h = box
                if debug_mode:
                    print("box:{},conf:{},cls:{}".format(box,conf,cls))
                    print("x:{},y:{},w:{},h:{}".format(x,y,w,h))
                
                if arge.show:
                    onnx_detector.draw_detections(img_media_copy,box,conf,cls)
                    cv2.namedWindow("result",cv2.WINDOW_NORMAL)
                    cv2.imshow("result",img_media_copy)
            cv2.waitKey(0)

        else:
            print(f"Processing Video: {source}")
            cap_media = cv2.VideoCapture(source,cv2.CAP_FFMPEG)
            if not cap_media.isOpened():
                print("Error: Could not open video.")
                exit()
            all_frame_count = int(cap_media.get(cv2.CAP_PROP_FRAME_COUNT))
            while True:
                ret, frame = cap_media.read()
                frame_id = int(cap_media.get(cv2.CAP_PROP_POS_FRAMES))
                print("processing video frame : {}-{}".format(frame_id,all_frame_count))
                if not ret:
                    print("End of video stream or cannot read the video.")
                    break
                if all_frame_count < frame_id:
                    print("Processing frame {}/{}".format(frame_id,all_frame_count))
                    break
                results = onnx_detector.detect(frame)
                for result in results:
                    box,conf,cls = result
                    x,y,w,h = box
                    # print("box:{},conf:{},cls:{}".format(box,conf,cls))
                
                if arge.show:
                    onnx_detector.draw_detections(frame,box,conf,cls)
                    cv2.namedWindow("result_frmae_id:{}".format(frame_id),cv2.WINDOW_NORMAL)
                    cv2.imshow("result_frmae_id:{}".format(frame_id),frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            cap_media.release()



    

    cv2.destroyAllWindows()
import argparse
import os
import cv2
from pathlib import Path
from urllib.parse import urlparse
from module.Yolov8_onnx import YOLOv8
from module.gst_moudle import create_gstreamerVideoCapture


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

def get_source_file(source_path, mode='all'):
    """
    獲取來源清單（支援檔案、資料夾、串流網址、Webcam）
    
    Args:
        source_path: str, 資料夾、檔案路徑、RTSP/RTMP 網址或 '0' (Webcam)
        mode: str, 過濾模式 'all', 'image', 'video'
    Returns:
        list: 包含 Path 物件或字串網址的清單
    """
    img_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    vid_formats = ['.mp4', '.avi', '.mkv', '.mov', '.flv']
    
    # 根據 mode 決定要過濾的副檔名
    target_formats = []
    if mode in ['all', 'image']: target_formats += img_formats
    if mode in ['all', 'video']: target_formats += vid_formats

    # 1. 處理串流或 Webcam (如: 'rtsp://...', 'http://...', '0')
    is_url = urlparse(str(source_path)).scheme in ['rtsp', 'rtmp', 'http', 'https']
    is_webcam = str(source_path).isnumeric() or str(source_path).startswith('/dev/video')

    if is_url or is_webcam:
        print(f"Detected stream source: {source_path}")
        return [str(source_path)]

    # 2. 處理本地檔案與資料夾
    p = Path(source_path).resolve()
    if not p.exists():
        print(f"Error: Path {source_path} does not exist.")
        return []

    files = []
    if p.is_dir():
        print(f"Scanning directory: {p}")
        # 使用 rglob 遞迴搜尋，並篩選符合條件的檔案
        for file in p.rglob('*'):
            if file.suffix.lower() in target_formats:
                files.append(file)
    elif p.is_file():
        if p.suffix.lower() in target_formats:
            files.append(p)
        else:
            print(f"Unsupported file format: {p.suffix}")
            
    return sorted(files)

def parse_arguments():
    """
    解析命令列參數以配置 YOLOv8 ONNX 推論環境。

    此函式定義了模型路徑、資料來源及推論超參數。設定完成後，
    可透過傳回的 Namespace 物件存取各項設定。

    Args:
        --model (str): ONNX 模型檔案路徑。預設為 "yolov8n.onnx"。
        --source (str): 通用輸入來源 (可為影像、資料夾、影片或串流位址)。
        --cfg (str): 模型配置檔 (.yaml) 路徑。
        --conf_thres (float): 信心度閾值 (Confidence Threshold)，預設為 0.5。
        --iou_thres (float): 非極大值抑制 (NMS) 的 IoU 閾值，預設為 0.5。
        --show (bool): 是否啟用結果視窗顯示。

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
    onnx_detector = YOLOv8(onnx_model=arge.model,yaml_file=arge.cfg,confidence_thres=arge.conf_thres,iou_thres=arge.iou_thres, debug_mode=arge.debug)
    # get file list
    source_list = get_source_file(arge.source)
    root_path = Path(__file__).resolve().parent


    for source in source_list:
        is_stream = str(source).startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        
        if is_stream:
            print(f"Processing Stream: {source}")
            cap_media = create_gstreamerVideoCapture(video_path= source)
            # cap_media = cv2.VideoCapture(source,cv2.CAP_FFMPEG)
            if not cap_media.isOpened():
                print("Error: Could not open video.")
                exit()
            while True:
                ret, frame = cap_media.read()
                if not ret:
                    print("End of video stream or cannot read the video.")
                    break
                frame_copy = frame.copy()
                results = onnx_detector.detect(frame)
                if not results:
                    # print("no detect target")
                    if arge.show:
                        cv2.namedWindow("result_frmae",cv2.WINDOW_NORMAL)
                        cv2.imshow("result_frmae",frame_copy)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    continue
                bboxes = results[0]
                confes = results[1]
                classes = results[2]
                if arge.show:
                    for i in range(len(bboxes)):
                        onnx_detector.draw_detections(frame_copy,bboxes[i],confes[i],classes[i])
                    cv2.namedWindow("result_frmae",cv2.WINDOW_NORMAL)
                    cv2.imshow("result_frmae",frame_copy)
                        
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap_media.release()
        elif source.suffix.lower() in img_formats:
            print(f"Processing Image: {source}")
            img_media = cv2.imread(source)
            img_media_copy = img_media
            results = onnx_detector.detect(img_media)
            if not results:
                if arge.show:
                    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
                    cv2.imshow("image",img_media_copy)
                    cv2.waitKey(0)
                continue
            bboxes = results[0]
            confes = results[1]
            classes = results[2]
            if debug_mode:
                print("result:{}".format(results))
                print("number of detections:{}".format(len(results)))
                print("bboxes:{}".format(bboxes))
                print("confes:{}".format(confes))
                print("classes:{}".format(classes))
            if arge.show:
                for i in range(len(bboxes)):
                    onnx_detector.draw_detections(img_media_copy, bboxes[i], confes[i], classes[i])
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.setWindowTitle("result", "result")
                cv2.moveWindow("result", 0, 0)
                # cv2.resizeWindow("result", img_media.shape[1], img_media.shape[0])
                cv2.imshow("result", img_media_copy)
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
                if not ret:
                    print("End of video stream or cannot read the video.")
                    break
                if all_frame_count < frame_id:
                    print("Processing frame {}/{}".format(frame_id,all_frame_count))
                    break
                frame_copy = frame.copy()
                results = onnx_detector.detect(frame)
                if not results:
                    # print("no detect target")
                    if arge.show:
                        cv2.namedWindow("result_frmae",cv2.WINDOW_NORMAL)
                        cv2.setWindowTitle("result_frmae","frame id : {:04d}".format(frame_id))
                        cv2.imshow("result_frmae",frame_copy)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    continue
                bboxes = results[0]
                confes = results[1]
                classes = results[2]
                if arge.show:
                    for i in range(len(bboxes)):
                        onnx_detector.draw_detections(frame_copy,bboxes[i],confes[i],classes[i])
                    cv2.namedWindow("result_frmae",cv2.WINDOW_NORMAL)
                    cv2.setWindowTitle("result_frmae","frame id : {:04d}".format(frame_id))
                    cv2.imshow("result_frmae",frame_copy)
                        
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap_media.release()

    
    cv2.destroyAllWindows()
  
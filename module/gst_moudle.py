import os
import platform
from itertools import product
import cv2



def is_x86PC():
    return platform.processor()=='x86_64'

def replace_gstStr_forPC(gst_str):
    gst_plugins= gst_str.split('!')

    new_plugins= []
    for plugin in gst_plugins:
        if 'nvv4l2decoder' in plugin:#x86上沒有max-perform等變數
            new_plugins.append('nvv4l2decoder')
        elif 'nvv4l2h264enc' in plugin:#x86上沒有max-perform等變數
            new_plugins.append('nvv4l2h264enc')
        elif 'nvvidconv' in plugin:#x86上使用nvvidconvert替代
            new_plugin= plugin.replace('nvvidconv', 'nvvideoconvert')
            new_plugins.append(new_plugin)
        elif 'nv3dsink' in plugin:#顯示部份Jetson: nv3dsink, x86: nveglglessink
            new_plugin= plugin.replace('nv3dsink', 'nveglglessink')
            new_plugins.append(new_plugin)
        else:
            new_plugins.append(plugin)
    
    new_plugins= [plugin for plugin in map(lambda x:x.strip(), new_plugins)]
    new_gst_str= ' ! '.join(new_plugins)
    return new_gst_str

class GstreamerPipeLineCreator_Nvidia:
    def __init__(self, video_path, limit_fps=None, latency=0, target_HW= None):
        """
        產生GStreamer decode video所需的Pipeline，會回傳一系列組合的pipelines，這些pipelines會依序去嘗試解碼，並且優先嘗試最省資源的解碼方法
        Args:
            video_path(str): 影片的來源(支援rtsp, webcam, video_path)。Ex: rtsp-->rtsp://192.168.1.1/mystream, webcam--> 0, video_path--> ./data/video.mp4
            limit_fps(int): 限制video輸出的fps
            latency(int): 用來控制rtsp的latency (msec)，當此數值為0每次都會近可能取得最新畫面(但有機會lag或封包丟失)，當此數值較大會與限制畫面差異latency msec，但封包較不易丟失。
            target_HW(None|tuple): 用來控制解碼完的video是否resize到指定大小target_HW。 Ex: target_HW=(720, 1280)
        """
        self.video_path= video_path
        self.limit_fps= limit_fps
        self.latency= latency
        self.target_HW= target_HW

    def check_src_format(self):
        if isinstance(self.video_path , int):
            return "webcam"
        elif self.video_path.startswith('rtsp'):
            return 'rtsp'
        elif os.path.isfile(self.video_path):
            return "file"        
        else:
            # force the error occurs when video format is not supported.
            # Have to be fixed in gstreamer. self.capture.isOpened() is None in IPCamCapture.
            print("Unsupported video format")
            return 'rtsp'


    def get_video_src(self):
        """
        取得Gstreamer 來源的pipeline部份
        """
        if self.check_src_format()=='webcam':
            gst_str_start = f'v4l2src device=/dev/video{str(self.video_path)}'
        elif self.check_src_format()=='rtsp':
            gst_str_start = f'rtspsrc location={self.video_path} latency={int(self.latency)} is-live=True'
        elif self.check_src_format()=='file':
            gst_str_start = f'filesrc location={self.video_path}'

        return [gst_str_start]
        

    def get_parser(self):
        """
        取得Gstreamer 針對Video parser的pipeline部份
        """
        if self.check_src_format()=='webcam':
            return [' ! image/jpeg,format=MJPG']
        elif self.check_src_format()=='rtsp':
            return [' ! rtph264depay ! h264parse', ' ! rtph265depay ! h265parse']
        elif self.check_src_format()=='file': 
            ext_name= os.path.splitext(self.video_path)[1].lower()
            h264_h265_parsers= [' ! h264parse', ' ! h265parse']
            ext_name_demux_dict={
                '.avi': [' ! avidemux'],
                '.mkv': [' ! matroskademux'],
                '.mp4': [' ! qtdemux'],
                '.mov': [' ! qtdemux'],                
            }

            demuxes= [' ! qtdemux', ' ! avidemux', ' ! matroskademux']
            if ext_name in ext_name_demux_dict:
                demuxes= ext_name_demux_dict[ext_name]
                        
            file_parsers= list(''.join(one_product) for one_product in product(demuxes, h264_h265_parsers))
            
            return file_parsers
    
    def get_limit_fps(self):
        """
        取得Gstreamer 限制FPS的pipeline部份
        """
        if self.limit_fps is None:
            return ['']
        else:
            return [f' ! videorate ! video/x-raw(memory:NVMM),framerate={self.limit_fps}/1']
    
    
    def get_decoder(self):
        """
        取得Gstreamer decoder的pipeline部份
        """
        jetson_decode_param= ''
        webcam_decode_param= ''

        is_webcam= (self.check_src_format()=='webcam')

        if is_webcam:
            webcam_decode_param= "mjpeg=1"
        if is_x86PC()==False:
            jetson_decode_param= "enable-max-performance=1"

        return [f' ! nvv4l2decoder {jetson_decode_param} {webcam_decode_param}']

        
    def get_converter(self):
        """
        取得Gstreamer converter的pipeline部份，通常用來轉換color space以及縮放影片大小
        """
        convert_list= None
        #nvvidconv video/x-raw(memory:NVMM),format=BGRx,width={width},height={height}
        if is_x86PC():
            convert_list= [' ! nvvideoconvert ! video/x-raw, format=(string)BGR{}', \
                    ' ! nvvideoconvert ! video/x-raw, format=(string)BGRx{} ! videoconvert'
            ]            
        else:            
            convert_list= [' ! nvvideoconvert compute-hw=1 ! video/x-raw, format=(string)BGR{}', \
                            ' ! nvvidconv ! video/x-raw, format=(string)BGRx{} ! videoconvert'
            ]
        
        if self.target_HW is None:
            convert_list= [ one_convert_str.format("") for one_convert_str in convert_list]
        else:
            convert_list= [ one_convert_str.format(f",width={self.target_HW[1]},height={self.target_HW[0]}") for one_convert_str in convert_list]

        return convert_list
            
    def get_output_sink(self):
        """
        取得Gstreamer 輸出的pipeline部份
        """
        src_format= self.check_src_format()
        if src_format in ['webcam', 'rtsp']:
            return [' ! appsink max-buffers=1 drop=true']
        else:
            return [" ! appsink"]

    def create_pipeline(self):
        """
        產生嘗試的Pipeline
        Return:
            pipelines(list): 產生要嘗試的pipeline組合
        """
        src_format= self.get_video_src()
        parser= self.get_parser()        
        decoder= self.get_decoder()
        limit_fps= self.get_limit_fps()
        converter= self.get_converter()
        output_sink= self.get_output_sink()

        product_gst_pipelines= product(src_format, parser, decoder, limit_fps, converter, output_sink)                
        gst_pipelines= [''.join(one_pipeline) for one_pipeline in product_gst_pipelines]
        
        return gst_pipelines


def create_gstreamerVideoCapture(video_path, limit_fps=None, latency=0, target_HW= None, platform= 'nvidia'):
    """
    產生Gstreamer的影片解碼VideoCapture    
    Args:
        video_path(str): 影片的來源(支援rtsp, webcam, video_path)。Ex: rtsp-->rtsp://192.168.1.1/mystream, webcam--> 0, video_path--> ./data/video.mp4
        limit_fps(int): 限制video輸出的fps
        latency(int): 用來控制rtsp的latency (msec)，當此數值為0每次都會近可能取得最新畫面(但有機會lag或封包丟失)，當此數值較大會與限制畫面差異latency msec，但封包較不易丟失。
        target_HW(None|tuple): 用來控制解碼完的video是否resize到指定大小target_HW。 Ex: target_HW=(720, 1280)
        platform(str): 指定平台，目前支援'nvidia', 'orangepi'
    Return:
        cap(cv2.VideoCapture): OpenCV呼叫GStreamer作為backend的VideoCapture
    """
    gst_pipeline_creator= GstreamerPipeLineCreator_Nvidia(video_path= video_path, limit_fps= limit_fps, latency= latency, target_HW= target_HW)
    gst_pipelines= gst_pipeline_creator.create_pipeline()
    for one_pipeline in gst_pipelines:
        cap= cv2.VideoCapture(one_pipeline)
        if cap.isOpened():
            print("Open video_path by : {}".format(one_pipeline))
            return cap
        
        #釋放開啟失敗的VideoCapture
        cap.release()
    
    print("Can't open video by hardware decode, try to use opencv software decoder")
    cap= cv2.VideoCapture(video_path)
    return cap



if __name__ == "__main__":
    rtsp_src= "rtsp://192.168.33.76:8554/live"
    cap = create_gstreamerVideoCapture(video_path= rtsp_src)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot read the video.")
            break
        cv2.namedWindow("result_frmae",cv2.WINDOW_NORMAL)
        cv2.imshow("result_frmae",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


import cv2
import numpy as np
import yaml
import onnxruntime as ort

class YOLOv8:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model,  yaml_file, confidence_thres=0.3, iou_thres=0.45, debug_mode=False):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.model = ort.InferenceSession(
            onnx_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']) #, 'TensorrtExecutionProvider'
        print("use providers:", self.model.get_providers())
        input_details = self.model.get_inputs()
        
        print("input_details shape : {}".format(input_details[0].shape))
        
        self.in_batch = input_details[0].shape[0]
        self.in_height, self.in_width = input_details[0].shape[2:]

        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        # Load the class names from the COCO dataset
        if yaml_file is None:
            self.classes = {i: i for i in range(1000)}
        else:
            with open(yaml_file) as f:
                self.classes = yaml.safe_load(f)["names"]

        # print("Loaded class names:", self.classes)

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(
            0, 255, size=(len(self.classes), 3))

        self.ratio = 0
        self.new_pad = [0, 0]
        self.padding = [0, 0, 0, 0]  # top, left, bottom, right
        self.scale_factor = [] # ratio, top, left, bottom, right
        self.inimage = 0
        self.debug = debug_mode

    def letterbox(self, img, new_shape=(640,640)):
        """Resizes and reshapes images while maintaining aspect ratio by adding padding, suitable for YOLO models."""
        shape = img.shape[:2]  # current shape [height, width]
        # Scale ratio (new / old)
        self.ratio = min(float(new_shape[0]) / float(shape[0]),
                         float(new_shape[1]) / float(shape[1]))

        # Compute padding
        self.new_pad = (int(round(shape[1] * self.ratio)),
                        int(round(shape[0] * self.ratio)))
        dw, dh = (new_shape[1] - self.new_pad[0]) / \
            2, (new_shape[0] - self.new_pad[1]) / 2  # wh padding

        if shape[::-1] != self.new_pad:  # resize
            img = cv2.resize(img, self.new_pad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        self.padding = [top, left, bottom, right]
        self.scale_factor.append([self.ratio,top,left,bottom,right])
        self.inimage = img
        if self.debug:
            print(f"Letterbox resize: original shape={shape}, new shape={new_shape}, resized shape={self.new_pad}, padding={self.padding}, ratio={self.ratio}")
            cv2.namedWindow("Letterbox Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Letterbox Image", img)
        return img

    def draw_detections(self, img, box, score, class_id,extra_text: str = ""):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]
        # ---- 動態決定文字顏色（避免撞色）----
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        luminance = 0.299*r + 0.587*g + 0.114*b
        text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)),
                      (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        # label = f"{self.classes[class_id]}: {score:.4f}"
            # ---- label（加入 extra_text）----
        if extra_text:
            label = f"{self.classes[class_id]} : {score:.4f} {extra_text}"
        else:
            label = f"{self.classes[class_id]} : {score:.4f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )
        # Draw outline for readability
        cv2.putText(img, label, (int(label_x), int(label_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0) if text_color==(255,255,255) else (255,255,255),
                    3, cv2.LINE_AA)

        # Draw main text
        cv2.putText(img, label, (int(label_x), int(label_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    def sacle_bbox(self, bboxes, img_shape):
        """將邊界框從調整大小的圖像比例縮放回原始圖像比例
        
        Args:
            bboxes: 多個邊界框 [[x, y, w, h], ...]
            img_shape: 原始圖像形狀 (height, width)
            Returns:
            縮放後的邊界框 [[x, y, w, h], ...]
        """
        top, left, bottom, right = self.padding
        bboxes_before = bboxes.copy()
        bboxes[0][:,0] -= left
        bboxes[0][:,1] -= top
        bboxes[0] /= self.ratio
        np.clip(bboxes[0][:,0], 0, None, out=bboxes[0][:,0])
        np.clip(bboxes[0][:,1], 0, None, out=bboxes[0][:,1])
        np.clip(bboxes[0][:,2], 0, img_shape[1], out=bboxes[0][:,2])
        np.clip(bboxes[0][:,3], 0, img_shape[0], out=bboxes[0][:,3])
        if self.debug:
            print(f"padding: top={top}, left={left}, bottom={bottom}, right={right}")
            print(f"ratio={self.ratio}")
            print(f"bbox before={bboxes_before}, after={bboxes}")
        return bboxes
        

    def nms_np(self, boxes, scores, iou_thres):
        """NumPy NMS
        Args:
            boxes: 邊界框
            scores: 分數
            iou_thres: IOU 閥值
        Returns:
            keep: 保留的索引
        """
        # x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        x1 = boxes[:, 0] - (boxes[:, 2] / 2)
        y1 = boxes[:, 1] - (boxes[:, 3] / 2)
        x2 = boxes[:, 0] + (boxes[:, 2] / 2)
        y2 = boxes[:, 1] + (boxes[:, 3] / 2)
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]

        return keep

    def check_output_range(self, outputs):
        max_value = np.max(outputs[:,:,:4])
        # print("max_value : {}".format(max_value))
        if max_value <= 1.5:
            return True
        else:
            return False

    def preprocess(self, img):
        """
        Preprocesses the input image before performing inference.
        Args:
            img: The input image.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # do letterbox resize
        img = self.letterbox(img, (self.in_height, self.in_width))
        # print(img.shape)
        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        # img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, img, outputs):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            img (numpy.ndarray): The input image.
            outputs (numpy.ndarray): The output of the model.
            pad (Tuple[float, float]): Padding used by letterbox.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        print("outputs :{}".format(outputs))
        print("outputs shape:{}".format(outputs.shape))


        if outputs.shape[0] == 0:
            return []
        outputs = outputs.transpose(0, 2, 1)
        outputs[..., 0:2] -= outputs[..., 2:4] / 2  # cxcy -> x1y1
        
        if self.check_output_range(outputs):
            print("0-1 -> 0-model input size")
            outputs[:,:,[0,2]] *= self.in_width
            outputs[:,:,[1,3]] *= self.in_height
        print("outputs :{}".format(outputs))
        print("outputs shape:{}".format(outputs.shape))


        batch_list = []
        bboxes = outputs[:,:,:4]
        scores = outputs[:,:,4:]
        max_score = np.max(scores, axis = -1)
        classes = np.argmax(scores, axis = -1)
        masks = max_score > self.confidence_thres
        final_bboxes = bboxes[masks]
        findal_scores = max_score[masks]
        final_classes = classes[masks]
        keeps = self.nms_np(final_bboxes, findal_scores, self.iou_thres)
        batch_list = [final_bboxes[keeps],findal_scores[keeps],final_classes[keeps]]
        box_data = self.sacle_bbox(batch_list,img.shape[:2])

        print("box_data:{}".format(box_data))
                           
        return box_data
            
    
    def detect(self, img):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.
        Args:
            img: input image
        Returns:
            output_img: The output image with drawn detections.
        """

        # Preprocess the image data
        img_data = self.preprocess(img)
        # print(img_data)

        model_inputs = self.model.get_inputs()[0].name

        # Run inference using the preprocessed image data
        outputs = self.model.run(None, {model_inputs: img_data})

        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(img, outputs[0])  # output image

    def batch_scale_bbox(self, bboxes, img_shape, batch_id):
        """Rescale bounding boxes from letterboxed size back to original image size.
        
        Args:
            bboxes: List containing [boxes_array, scores_array, classes_array]
                    boxes_array shape: (N, 4) in [x1, y1, w, h] format.
            img_shape: Tuple of (height, width) of the original image.
            batch_id: Index in the current batch.
        """
        ratio, top, left, _, _ = self.scale_factor[batch_id]
        boxes = bboxes[0].astype(np.float32, copy=True)
        
        # 1. Remove padding & scale back
        boxes[:, 0] -= left
        boxes[:, 1] -= top
        boxes /= ratio
        
        # 2. Correct Clipping for [x1, y1, w, h]
        h_orig, w_orig = img_shape
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w_orig) # x1
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h_orig) # y1
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w_orig - boxes[:, 0]) # width
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h_orig - boxes[:, 1]) # height
        
        bboxes[0] = boxes
        return bboxes

    def batch_preprocess(self, images):
        """
        Preprocess the input images before performing inference.
        Args:
            images: List of input images (BGR).
        Returns:
            image_data: Preprocessed images data ready for inference.
        """
        self.scale_factor = [] # CRITICAL: Clear previous batch state
        new_images = [] 
        for im in images:
            img = self.letterbox(im, (self.in_height, self.in_width))
            img = img[:, :, ::-1] # BGR -> RGB
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            new_images.append(img)
            
        image_data = np.stack(new_images, axis=0)
        
        # If engine batch size is fixed but fewer images provided, pad with zeros
        if image_data.shape[0] < self.in_batch:
            padding = np.zeros((self.in_batch - image_data.shape[0], *image_data.shape[1:]), dtype=np.float32)
            image_data = np.concatenate([image_data, padding], axis=0)
            
        return image_data
    
    def batch_postprocess(self, shape_list, outputs):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            shape_list: List of original image shapes (H, W).
            outputs (numpy.ndarray): The output of the model.

        Returns:
            box_data: List of detection results for each image.
        """
        if outputs.shape[0] == 0:
            return []
        outputs = outputs.transpose(0, 2, 1)
        outputs[..., 0:2] -= outputs[..., 2:4] / 2  # cxcy -> x1y1
        if self.check_output_range(outputs):
            outputs[:,:,[0,2]] *= self.in_width
            outputs[:,:,[1,3]] *= self.in_height
        
        box_data = []
        bboxes = outputs[:, :, :4]
        scores = outputs[:, :, 4:]
        max_scores = np.max(scores, axis=-1)
        classes = np.argmax(scores, axis=-1)
        maskes = max_scores > self.confidence_thres

        # Iterate only through images actually provided in the batch
        for i in range(len(shape_list)):
            bbox = bboxes[i]
            score = max_scores[i]
            classId = classes[i]
            mask = maskes[i]
            
            final_bboxes = bbox[mask]    
            final_scores = score[mask]
            final_classes = classId[mask]
            keeps = self.nms_np(final_bboxes, final_scores, self.iou_thres)
            batch_result = [final_bboxes[keeps], final_scores[keeps], final_classes[keeps]]
            
            batch_result = self.batch_scale_bbox(batch_result, shape_list[i], i)
            box_data.append(batch_result)
            
        return box_data
    def get_images_shape(self,images):
        shape_list = []
        for im in images:
            shape_list.append(im.shape[:2])
        return shape_list

    def detect_batch(self,imgs):
        """
        多批次 YOLOv8 推論
        imgs: list[np.ndarray]  (BGR)
        return: list[box_data] 與 detect() 相同格式，只是變成 list 對應每張圖片
        """

        batch_size = len(imgs)
        if batch_size == 0:
            return []
        
        

        shape_list = self.get_images_shape(imgs)

        images_data = self.batch_preprocess(imgs)

        print("images_data shape :{}".format(images_data.shape))

        batch_input = np.ascontiguousarray(images_data)  # (B,3,H,W)

        model_input = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {model_input: batch_input})
        
        return self.batch_postprocess(shape_list, outputs[0])
    
if __name__ == "__main__":
    # model_path = "/workspaces_data/repo/onnx/model/yolov8n6432.onnx"
    model_path = "/workspaces_data/repo/onnx/model/yolov8n6432bs02.onnx"
    yaml_path = "/workspaces_data/repo/onnx/model/metadata.yaml"
    image_path = "/workspaces_data/repo/onnx/img/000000000459.jpg"

    image_path1 = "/workspaces_data/repo/tensorRT/img/000000000459.jpg"
    image_path2 = "/workspaces_data/repo/tensorRT/img/000000003136.jpg"
    image_list = [image_path1,image_path2]
    images = [] 
    onnx_detect = YOLOv8(onnx_model=model_path,yaml_file=yaml_path,debug_mode=False)
    infer_mode = "multi"
    if infer_mode == "single":
        print("single inference")
        # single mode
        data = cv2.imread(image_path)
        data_copy = data.copy()
        results = onnx_detect.detect(data)
        for i in range(len(results[0])):
            onnx_detect.draw_detections(data_copy,results[0][i],results[1][i],results[2][i])
        cv2.namedWindow("test",cv2.WINDOW_NORMAL)
        cv2.imshow("test",data_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    elif infer_mode == "multi":
        print("multi inference")
        for i in range(len(image_list)):
            img = cv2.imread(image_list[i])
            images.append(img)
        results = onnx_detect.detect_batch(images)
        for i in range(len(results)):
            for j in range(len(results[i][0])):
                onnx_detect.draw_detections(images[i],results[i][0][j],results[i][1][j],results[i][2][j])
            cv2.namedWindow("img{}".format(i), cv2.WINDOW_NORMAL)
            cv2.imshow("img{}".format(i), images[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    

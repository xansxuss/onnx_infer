import cv2
import numpy as np
import yaml
import onnxruntime as ort

class YOLOv8:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model,  yaml_file, confidence_thres=0.3, iou_thres=0.45, debug=False):
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
        self.in_height, self.in_width = input_details[0].shape[2:]
        # print(input_details[0].shape[2:])
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
        self.new_pad = (0, 0)
        self.padding = (0, 0, 0, 0)  # top, left, bottom, right
        self.inimage = 0
        self.debug = debug

    def letterbox(self, img, new_shape):
        """Resizes and reshapes images while maintaining aspect ratio by adding padding, suitable for YOLO models."""
        shape = img.shape[:2]  # current shape [height, width]
        # Scale ratio (new / old)
        # r = min(float(new_shape[0]) / float(shape[0]),
        #         float(new_shape[1]) / float(shape[1]))
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
        self.padding = (top, left, bottom, right)
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

    def scale_bbox(self, bbox, img_shape):
        """將邊界框從調整大小的圖像比例縮放回原始圖像比例
        Args:
            bbox: 單一邊界框 [x, y, w, h]
            img_shape: 原始圖像形狀 (height, width)
        Returns:
            縮放後的邊界框 [x, y, w, h]
        """
        top, left, bottom, right = self.padding

        bbox_before = bbox.copy()
        bbox[0] = (bbox[0] - left) / self.ratio
        bbox[1] = (bbox[1] - top) / self.ratio
        bbox[2] = bbox[2] / self.ratio
        bbox[3] = bbox[3] / self.ratio
        # check bbox range
        bbox[0] = np.clip(bbox[0], 0, img_shape[1] - 10)
        bbox[1] = np.clip(bbox[1], 0, img_shape[0] - 10)
        if bbox[0] + bbox[2] > img_shape[1]:
            bbox[2] = img_shape[1] - bbox[0]
        if bbox[1] + bbox[3] > img_shape[0]:
            bbox[3] = img_shape[0] - bbox[1]

        # debug information
        if self.debug:
            print(f"padding: top={top}, left={left}, ")
            print(f"ratio={self.ratio}")
            print(f"bbox before={bbox_before}, after={bbox}")
        return bbox

    def batch_sacle_bbox(self, bboxes, img_shape):
        """將邊界框從調整大小的圖像比例縮放回原始圖像比例
        
        Args:
            bboxes: 多個邊界框 [[x, y, w, h], ...]
            img_shape: 原始圖像形狀 (height, width)
            Returns:
            縮放後的邊界框 [[x, y, w, h], ...]
        """
        top, left, bottom, right = self.padding

        bboxes[:, 0] = (bboxes[:, 0] - left) / self.ratio
        bboxes[:, 1] = (bboxes[:, 1] - top) / self.ratio
        bboxes[:, 2] = bboxes[:, 2] / self.ratio
        bboxes[:, 3] = bboxes[:, 3] / self.ratio
        # check bbox range
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, None)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, None)
        # clip width
        bboxes[:, 2] = np.minimum(
            bboxes[:, 2],
            img_shape[1] - bboxes[:, 0]
        )

        # clip height
        bboxes[:, 3] = np.minimum(
            bboxes[:, 3],
            img_shape[0] - bboxes[:, 1]
        )
        
        if self.debug:
            print(f"box batch shape before scaling: {bboxes.shape}")
            for i in range(bboxes.shape[0]):
                print(f"box [{i}] shape : {bboxes[i].shape}")
                print(f"[{i}] padding: top={top}, left={left}, bottom={bottom}, right={right}")
                print(f"[{i}] ratio={self.ratio}")
                print(f"[{i}] bbox after={bboxes[i]}")
        return bboxes
        

    def nms_np(self, boxes, scores, iou_thres):
        """純 NumPy NMS"""
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

    def batch_nms(self, outputs, confidence_thres=0.25, iou_thres=0.45, class_agnostic=True):
        """
        outputs: [batch, num_boxes, 4+C] (boxes + class scores)
        class_agnostic: True -> 不分類別做 NMS, False -> 逐類別做 NMS
        """
        batch_results = []
        Batch = outputs.shape[0]

        for b in range(Batch):
            output = outputs[b]  # [num_boxes, 4+C]
            if output.shape[0] == 0:
                batch_results.append(
                    (np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)))
                continue

            boxes = output[:, :4].copy()
            class_scores = output[:, 4:]

            # 判斷座標是否為 0~1
            if boxes.max() <= 1.5:
                boxes[:, [0, 2]] *= self.in_width
                boxes[:, [1, 3]] *= self.in_height

            # 修正負寬高
            # boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0])
            # boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1])

            # 每個框取最高類別分數
            scores = class_scores.max(1)
            classes = class_scores.argmax(1)

            # 濾掉低信心框
            mask = scores > confidence_thres
            boxes, scores, classes = boxes[mask], scores[mask], classes[mask]

            if boxes.shape[0] == 0:
                batch_results.append(
                    (np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)))
                continue

            final_boxes, final_scores, final_classes = [], [], []

            if class_agnostic:
                keep = self.nms_np(boxes, scores, iou_thres)
                final_boxes.append(boxes[keep])
                final_scores.append(scores[keep])
                final_classes.append(classes[keep])
                if self.debug:
                    print(
                        f"[Batch {b}] Class-agnostic NMS kept {len(keep)} boxes")
            else:
                for cls in np.unique(classes):
                    cls_mask = classes == cls
                    cls_boxes = boxes[cls_mask]
                    cls_scores = scores[cls_mask]

                    keep = self.nms_np(cls_boxes, cls_scores, iou_thres)

                    final_boxes.append(cls_boxes[keep])
                    final_scores.append(cls_scores[keep])
                    final_classes.append(np.full(len(keep), cls))
                    if self.debug:
                        print(
                            f"[Batch {b}] Class {cls} NMS kept {len(keep)} boxes")

            final_boxes = np.concatenate(final_boxes, axis=0)
            final_scores = np.concatenate(final_scores, axis=0)
            final_classes = np.concatenate(final_classes, axis=0)

            batch_results.append((final_boxes, final_scores, final_classes))

        return batch_results

    def preprocess(self, img):
        """
        Preprocesses the input image before performing inference.

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
        outputs = outputs.transpose(0, 2, 1)
        outputs[..., 0:2] -= outputs[..., 2:4] / 2  # cxcy -> x1y1
        
        batch_results = self.batch_nms(
            outputs, self.confidence_thres, self.iou_thres)

        box_data = []
        for boxes, scores, class_ids in batch_results:
            for i in range(len(boxes)):
                box = boxes[i]
                score = scores[i]
                class_id = class_ids[i]
                if self.debug:
                    # check NMS draw detections
                    self.draw_detections(self.inimage, box, score, class_id)
                    cv2.namedWindow("Debug NMS Image", cv2.WINDOW_NORMAL)
                    cv2.imshow("Debug NMS Image", self.inimage)

                result = self.scale_bbox(box, img.shape)
                box_data.append([result, score, class_id])

        if self.debug:
            # final box data
            for box_info in box_data:
                cv2.circle(self.inimage, (int(box_info[0][0]), int(box_info[0][1])), 10, (0, 0, 255), -1)
                print("Final box data: {}".format(box_info))
                cv2.namedWindow("Debug scale Image", cv2.WINDOW_NORMAL)
                cv2.imshow("Debug scale Image", self.inimage)
        return box_data
        # return self.inimage, img, box_data

    def detect(self, img):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

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
    def detect_batch(self, imgs):
        """
        多批次 YOLOv8 推論
        imgs: list[np.ndarray]  (BGR)
        return: list[box_data] 與 detect() 相同格式，只是變成 list 對應每張圖片
        """

        batch_size = len(imgs)
        if batch_size == 0:
            return []

        preprocessed = []
        origin_imgs = []
        results = []

        # ---- 前處理：每張圖片 letterbox + normalize ----
        for img in imgs:
            origin_imgs.append(img.copy())  # 保留原圖
            img_data = self.preprocess(img)  # shape = (1,3,H,W)
            preprocessed.append(img_data)

        # ---- 堆疊成 batch ----
        # preprocessed list: [(1,3,640,640), (1,3,640,640)...]
        batch_input = np.concatenate(preprocessed, axis=0)  # (B,3,H,W)

        model_input = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {model_input: batch_input})
        # ---- 後處理 ----
        batch_output = outputs[0]  # 原始 batch output: (B, C, N)

        results = []
        for index, image in enumerate(origin_imgs):
            # ---- 取出單張 slice ----
            single_output = batch_output[index]  # shape = (C, N)

            # ---- 改成 (1, C, N) ----
            single_output = np.expand_dims(single_output, axis=0)  # (1, C, N)
            if self.debug:
                print(f"index:{index}, output shape: {single_output.shape}")

            # ---- 後處理 ----
            result = self.postprocess(image, single_output)
            results.append(result)

        return results
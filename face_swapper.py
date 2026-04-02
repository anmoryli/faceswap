"""
核心换脸模块 v5
使用 inswapper_128.onnx + gfpgan_1.4.onnx
关键修复：从 ONNX 模型中提取 emap 矩阵并做 embedding 投影（官方实现必须步骤）
"""
import os
import cv2
import numpy as np
import onnxruntime as ort

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clip(0, max_shape[1])
        y1 = y1.clip(0, max_shape[0])
        x2 = x2.clip(0, max_shape[1])
        y2 = y2.clip(0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clip(0, max_shape[1])
            py = py.clip(0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

def nms(dets, thresh):
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]
    return keep


# ─────────────────────────────────────────────
# 人脸检测器（RetinaFace det_10g.onnx）
# ─────────────────────────────────────────────

class FaceDetector:
    def __init__(self, model_path, input_size=(640, 640), det_thresh=0.5, nms_thresh=0.4):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_size = input_size
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh
        self._strides = [8, 16, 32]
        self._num_anchors = 2
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, img):
        h, w = img.shape[:2]
        im_ratio = min(self.input_size[0] / h, self.input_size[1] / w)
        new_h, new_w = int(h * im_ratio), int(w * im_ratio)
        resized = cv2.resize(img, (new_w, new_h))
        blob = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
        blob[:new_h, :new_w] = resized
        blob = blob.astype(np.float32)
        blob -= np.array([127.5, 127.5, 127.5])
        blob /= 128.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        return blob, im_ratio

    def detect(self, img):
        blob, im_ratio = self._preprocess(img)
        outputs = self.session.run(None, {self.input_name: blob})

        scores_list, bboxes_list, kps_list = [], [], []
        fmc = 3
        for idx, stride in enumerate(self._strides):
            h = self.input_size[0] // stride
            w = self.input_size[1] // stride
            scores = outputs[idx]
            bbox_preds = outputs[idx + fmc] * stride
            kps_preds  = outputs[idx + fmc * 2] * stride

            anchors = []
            for i in range(h):
                for j in range(w):
                    for _ in range(self._num_anchors):
                        anchors.append([j * stride, i * stride])
            anchors = np.array(anchors, dtype=np.float32)

            pos_inds = np.where(scores[:, 0] >= self.det_thresh)[0]
            if len(pos_inds) == 0:
                continue
            bboxes = distance2bbox(anchors, bbox_preds)[pos_inds]
            kps = distance2kps(anchors, kps_preds)[pos_inds]
            kps = kps.reshape(-1, 5, 2)
            sc = scores[pos_inds, 0]

            scores_list.append(sc)
            bboxes_list.append(bboxes)
            kps_list.append(kps)

        if not scores_list:
            return [], []

        scores_all = np.concatenate(scores_list)
        bboxes_all = np.concatenate(bboxes_list)
        kps_all = np.concatenate(kps_list)

        dets = np.hstack([bboxes_all, scores_all[:, np.newaxis]])
        keep = nms(dets, self.nms_thresh)
        dets = dets[keep]
        kps_all = kps_all[keep]

        dets[:, :4] /= im_ratio
        kps_all /= im_ratio
        return dets, kps_all


# ─────────────────────────────────────────────
# 人脸识别器（ArcFace w600k_r50.onnx）
# ─────────────────────────────────────────────

ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

def estimate_norm(lmk, image_size=112):
    from skimage.transform import SimilarityTransform
    dst = ARCFACE_DST * image_size / 112.0
    try:
        tform = SimilarityTransform.from_estimate(lmk, dst)
    except AttributeError:
        tform = SimilarityTransform()
        tform.estimate(lmk, dst)
    return tform.params[0:2, :]

def norm_crop(img, landmark, image_size=112):
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

class FaceEmbedder:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def get_embedding(self, face_img):
        """face_img: 112x112 BGR，返回 L2 归一化的 512 维向量"""
        img = face_img.astype(np.float32)
        img = (img - 127.5) / 127.5
        img = img[:, :, ::-1]  # BGR -> RGB
        img = img.transpose(2, 0, 1)[np.newaxis]
        emb = self.session.run(None, {self.input_name: img})[0][0]
        emb = emb / np.linalg.norm(emb)
        return emb


# ─────────────────────────────────────────────
# InSwapper 换脸模型（inswapper_128.onnx）
# ─────────────────────────────────────────────

class FaceSwapModel:
    def __init__(self, model_path):
        # 从 ONNX 中提取 emap 矩阵（官方实现关键步骤）
        import onnx
        from onnx import numpy_helper
        m = onnx.load(model_path)
        self.emap = numpy_helper.to_array(m.graph.initializer[-1]).astype(np.float32)
        print(f"emap shape: {self.emap.shape}")

        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        inputs = self.session.get_inputs()
        self.input_names = [i.name for i in inputs]
        self.input_size = 128
        print(f"InSwapper 输入: {[f'{i.name}{i.shape}' for i in inputs]}")

    def get_latent(self, normed_embedding):
        """
        官方投影: embedding(512,) @ emap(512,512) -> 再 L2 归一化
        """
        latent = normed_embedding.reshape(1, -1)
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        return latent.astype(np.float32)

    def swap(self, target_face_img, source_embedding):
        """
        target_face_img: 对齐后的人脸图 128x128 BGR uint8
        source_embedding: ArcFace L2 归一化特征向量 (512,)
        returns: (换脸后的 128x128 BGR uint8, 对齐后的原脸 128x128 BGR)
        """
        # 预处理 target：BGR -> RGB, /255, NCHW
        aimg = target_face_img  # 保存原对齐图（用于贴回 fake_diff mask）
        img = aimg.astype(np.float32) / 255.0
        img = img[:, :, ::-1].copy()  # BGR -> RGB
        img = img.transpose(2, 0, 1)[np.newaxis]  # (1,3,128,128)

        # embedding 投影
        latent = self.get_latent(source_embedding)  # (1,512)

        pred = self.session.run(None, {
            self.input_names[0]: img,    # target
            self.input_names[1]: latent, # source
        })[0]  # (1,3,128,128) float32 [0,1] RGB

        # 后处理：NCHW -> HWC, RGB -> BGR, *255
        fake = pred[0].transpose(1, 2, 0)
        fake_bgr = np.clip(255 * fake, 0, 255).astype(np.uint8)[:, :, ::-1]
        return fake_bgr, aimg


# ─────────────────────────────────────────────
# GFPGAN 人脸增强（gfpgan_1.4.onnx）
# ─────────────────────────────────────────────

class GFPGANEnhancer:
    """GFPGAN 1.4: input(1,3,512,512) -> output(1,3,512,512), 值域 [-1,1]"""
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = 512
        print(f"GFPGAN输入: {inp.name}{inp.shape}")

    def enhance(self, face_img):
        """face_img: BGR uint8，任意尺寸；返回 BGR uint8 512x512"""
        img = cv2.resize(face_img, (self.input_size, self.input_size))
        img = img.astype(np.float32)
        img = img[:, :, ::-1].copy()  # BGR -> RGB
        img = (img / 127.5) - 1.0     # [0,255] -> [-1,1]
        img = img.transpose(2, 0, 1)[np.newaxis]

        result = self.session.run(None, {self.input_name: img})[0]
        result = result[0].transpose(1, 2, 0)
        result = (result + 1.0) * 127.5   # [-1,1] -> [0,255]
        result = result[:, :, ::-1].copy() # RGB -> BGR
        return result.clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# 主流程封装
# ─────────────────────────────────────────────

class FaceSwapPipeline:
    def __init__(self, use_enhancer=True):
        det_path  = os.path.join(MODELS_DIR, 'det_10g.onnx')
        emb_path  = os.path.join(MODELS_DIR, 'w600k_r50.onnx')
        swap_path = os.path.join(MODELS_DIR, 'inswapper_128.onnx')
        enh_path  = os.path.join(MODELS_DIR, 'gfpgan_1.4.onnx')

        for p in [det_path, emb_path, swap_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f'模型文件不存在: {p}')

        self.detector  = FaceDetector(det_path)
        self.embedder  = FaceEmbedder(emb_path)
        self.swapper   = FaceSwapModel(swap_path)
        self.use_enhancer = use_enhancer and os.path.exists(enh_path)
        if self.use_enhancer:
            self.enhancer = GFPGANEnhancer(enh_path)
            print('GFPGAN增强器已加载')
        else:
            self.enhancer = None
        print('所有模型加载完成')

    def _get_biggest_face(self, img):
        dets, kps = self.detector.detect(img)
        if len(dets) == 0:
            return None, None
        areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        idx = areas.argmax()
        return dets[idx], kps[idx]

    def get_source_embedding(self, source_img):
        det, kps = self._get_biggest_face(source_img)
        if det is None:
            return None
        face = norm_crop(source_img, kps, image_size=112)
        return self.embedder.get_embedding(face)

    def swap_image(self, source_img, target_img, progress_cb=None):
        src_emb = self.get_source_embedding(source_img)
        if src_emb is None:
            raise ValueError('来源图片中未检测到人脸')

        dets, kps_all = self.detector.detect(target_img)
        if len(dets) == 0:
            raise ValueError('目标图片中未检测到人脸')

        result = target_img.copy()
        for i, (det, kps) in enumerate(zip(dets, kps_all)):
            result = self._swap_single_face(result, src_emb, kps)
            if progress_cb:
                progress_cb((i + 1) / len(dets))
        return result

    def _swap_single_face(self, img, src_emb, kps):
        """对图中单张人脸执行换脸并贴回原图（官方 inswapper 贴回逻辑）"""
        h, w = img.shape[:2]

        # 1. 对齐到 128x128
        M = estimate_norm(kps, image_size=128)
        aimg = cv2.warpAffine(img, M, (128, 128), borderValue=0.0)

        # 2. InSwapper 换脸
        fake_bgr, aimg = self.swapper.swap(aimg, src_emb)

        # 3. 可选 GFPGAN 增强（512px 后 resize 回 128）
        if self.use_enhancer and self.enhancer is not None:
            enhanced = self.enhancer.enhance(fake_bgr)  # 512x512
            fake_bgr = cv2.resize(enhanced, (128, 128))

        # 4. 官方贴回逻辑：用 fake_diff 差值 mask + img_white 椭圆 mask 融合
        result = self._paste_back(img, fake_bgr, aimg, M)
        return result

    def _paste_back(self, target_img, fake_bgr, aimg, M):
        """官方 inswapper 贴回逻辑（结合差值 mask 和形态学处理）"""
        H, W = target_img.shape[:2]
        IM = cv2.invertAffineTransform(M)

        # 将换脸结果和白色 mask warp 回原图坐标
        img_fake_back  = cv2.warpAffine(fake_bgr, IM, (W, H), borderValue=0.0)
        img_white      = np.full((128, 128), 255, dtype=np.float32)
        img_white_back = cv2.warpAffine(img_white, IM, (W, H), borderValue=0.0)
        img_white_back[img_white_back > 20] = 255

        # 计算换脸前后差值 mask（高差异区域=换脸区，低差异=背景噪声）
        fake_diff = np.abs(fake_bgr.astype(np.float32) - aimg.astype(np.float32)).mean(axis=2)
        fake_diff[:2, :] = 0; fake_diff[-2:, :] = 0
        fake_diff[:, :2] = 0; fake_diff[:, -2:] = 0
        fake_diff_back = cv2.warpAffine(fake_diff, IM, (W, H), borderValue=0.0)
        fake_diff_back[fake_diff_back < 10]  = 0
        fake_diff_back[fake_diff_back >= 10] = 255

        # 形态学腐蚀 + 高斯模糊，得到平滑 mask
        mask_h, mask_w = np.where(img_white_back > 20)[0], np.where(img_white_back > 20)[1]
        if len(mask_h) == 0:
            return target_img
        mask_size = int(np.sqrt(len(mask_h)))
        k = max(mask_size // 10, 10)
        kernel = np.ones((k, k), np.uint8)
        img_mask = cv2.erode(img_white_back, kernel, iterations=1)
        blur_k = max(2 * (mask_size // 20) + 1, 3)
        img_mask = cv2.GaussianBlur(img_mask, (blur_k, blur_k), 0)
        fake_diff_back = cv2.dilate(fake_diff_back, np.ones((2, 2), np.uint8), iterations=1)
        fake_diff_back = cv2.GaussianBlur(fake_diff_back, (11, 11), 0)

        # 综合两个 mask
        img_mask_final = (img_mask / 255.0)
        # diff mask 用于增强换脸区内的覆盖（乘法综合）
        # 可单独只用 img_mask，更稳定
        img_mask_final = img_mask_final.reshape(H, W, 1)

        result = (img_mask_final * img_fake_back.astype(np.float32) +
                  (1 - img_mask_final) * target_img.astype(np.float32))
        return result.clip(0, 255).astype(np.uint8)

    def swap_video(self, source_img, video_path, output_path, progress_cb=None):
        src_emb = self.get_source_embedding(source_img)
        if src_emb is None:
            raise ValueError('来源图片中未检测到人脸')

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dets, kps_all = self.detector.detect(frame)
            result = frame.copy()
            for det, kps in zip(dets, kps_all):
                result = self._swap_single_face(result, src_emb, kps)
            out.write(result)
            frame_idx += 1
            if progress_cb and total > 0:
                progress_cb(frame_idx / total)

        cap.release()
        out.release()
        return output_path

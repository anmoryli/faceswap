"""
核心换脸模块 v5
使用 inswapper_128.onnx + gfpgan_1.4.onnx
关键修复：从 ONNX 模型中提取 emap 矩阵并做 embedding 投影（官方实现必须步骤）
"""
import os
import cv2
import numpy as np
import onnxruntime as ort
import threading
from queue import Queue

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
    def __init__(self, model_path, input_size=(640, 640), det_thresh=0.45, nms_thresh=0.4):
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


def reinhard_color_transfer(source_bgr, target_bgr):
    """将 source 的整体肤色和明暗统计迁移到 target，提升融合自然度。"""
    src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    for ch in range(3):
        src_mean, src_std = cv2.meanStdDev(src_lab[:, :, ch])
        tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab[:, :, ch])
        src_mean = float(src_mean[0][0])
        src_std = float(src_std[0][0]) + 1e-6
        tgt_mean = float(tgt_mean[0][0])
        tgt_std = float(tgt_std[0][0]) + 1e-6
        tgt_lab[:, :, ch] = (tgt_lab[:, :, ch] - tgt_mean) * (src_std / tgt_std) + src_mean

    tgt_lab = np.clip(tgt_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(tgt_lab, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
# 主流程封装
# ─────────────────────────────────────────────

class FaceSwapPipeline:
    def __init__(self, use_enhancer=True, det_thresh=0.45):
        det_path  = os.path.join(MODELS_DIR, 'det_10g.onnx')
        emb_path  = os.path.join(MODELS_DIR, 'w600k_r50.onnx')
        swap_path = os.path.join(MODELS_DIR, 'inswapper_128.onnx')
        enh_path  = os.path.join(MODELS_DIR, 'gfpgan_1.4.onnx')

        for p in [det_path, emb_path, swap_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f'模型文件不存在: {p}')

        self.detector  = FaceDetector(det_path, det_thresh=det_thresh)
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

    def swap_image(self, source_img, target_img, progress_cb=None, enable_enhancement=None, max_faces=None):
        src_emb = self.get_source_embedding(source_img)
        if src_emb is None:
            raise ValueError('来源图片中未检测到人脸')

        dets, kps_all = self.detector.detect(target_img)
        if len(dets) == 0:
            raise ValueError('目标图片中未检测到人脸')

        if enable_enhancement is None:
            enable_enhancement = self.use_enhancer

        if max_faces is not None and len(dets) > max_faces:
            areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
            keep = np.argsort(-areas)[:max_faces]
            dets = dets[keep]
            kps_all = kps_all[keep]

        result = target_img.copy()
        total_faces = max(len(dets), 1)
        for i, (det, kps) in enumerate(zip(dets, kps_all)):
            result = self._swap_single_face(result, src_emb, kps, enable_enhancement=enable_enhancement)
            if progress_cb:
                progress_cb((i + 1) / total_faces)
        return result

    def _swap_single_face(self, img, src_emb, kps, enable_enhancement=True):
        """对图中单张人脸执行换脸并贴回原图（官方 inswapper 贴回逻辑）"""
        h, w = img.shape[:2]

        # 1. 对齐到 128x128
        M = estimate_norm(kps, image_size=128)
        aimg = cv2.warpAffine(img, M, (128, 128), borderValue=0.0)

        # 2. InSwapper 换脸
        fake_bgr, aimg = self.swapper.swap(aimg, src_emb)

        # 3. 色彩迁移，减少贴回后的肤色割裂
        fake_bgr = reinhard_color_transfer(aimg, fake_bgr)

        # 4. 可选 GFPGAN 增强（512px 后 resize 回 128）
        if enable_enhancement and self.enhancer is not None:
            enhanced = self.enhancer.enhance(fake_bgr)  # 512x512
            fake_bgr = cv2.resize(enhanced, (128, 128), interpolation=cv2.INTER_CUBIC)

        # 5. 贴回原图
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

        # 综合两个 mask：img_mask 负责平滑边缘，fake_diff_back 负责增强真实换脸区域覆盖
        img_mask_final = img_mask / 255.0
        diff_mask = (fake_diff_back / 255.0) * 0.35
        img_mask_final = np.clip(np.maximum(img_mask_final, diff_mask), 0.0, 1.0)
        img_mask_final = img_mask_final.reshape(H, W, 1)

        result = (img_mask_final * img_fake_back.astype(np.float32) +
                  (1 - img_mask_final) * target_img.astype(np.float32))
        return result.clip(0, 255).astype(np.uint8)

    def _detect_with_scale(self, frame, detect_scale=1.0):
        if detect_scale is None or detect_scale >= 0.999:
            return self.detector.detect(frame)
        h, w = frame.shape[:2]
        sw = max(64, int(w * detect_scale))
        sh = max(64, int(h * detect_scale))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
        dets, kps_all = self.detector.detect(small)
        if len(dets) == 0:
            return dets, kps_all
        dets = dets.copy()
        kps_all = kps_all.copy()
        dets[:, :4] /= detect_scale
        kps_all /= detect_scale
        return dets, kps_all

    def _kps_to_dets(self, kps_all, frame_shape):
        if kps_all is None or len(kps_all) == 0:
            return [], []
        h, w = frame_shape[:2]
        dets = []
        for kps in kps_all:
            x1, y1 = kps.min(axis=0)
            x2, y2 = kps.max(axis=0)
            bw = x2 - x1
            bh = y2 - y1
            pad_x = bw * 0.5
            pad_y = bh * 0.7
            dets.append([
                max(0.0, x1 - pad_x),
                max(0.0, y1 - pad_y),
                min(float(w - 1), x2 + pad_x),
                min(float(h - 1), y2 + pad_y),
                1.0,
            ])
        return np.asarray(dets, dtype=np.float32), np.asarray(kps_all, dtype=np.float32)

    def _track_keypoints(self, prev_gray, gray, prev_kps_all):
        if prev_gray is None or gray is None or prev_kps_all is None or len(prev_kps_all) == 0:
            return None
        pts = prev_kps_all.reshape(-1, 1, 2).astype(np.float32)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_pts is None or status is None:
            return None
        status = status.reshape(-1)
        next_pts = next_pts.reshape(prev_kps_all.shape)
        face_status = status.reshape(-1, 5)
        good_faces = face_status.mean(axis=1) >= 0.8
        if not np.any(good_faces):
            return None
        tracked = next_pts[good_faces]
        return tracked.astype(np.float32)

    def swap_video(self, source_img, video_path, output_path, progress_cb=None, enable_enhancement=None, max_faces=1):
        src_emb = self.get_source_embedding(source_img)
        if src_emb is None:
            raise ValueError('来源图片中未检测到人脸')

        if enable_enhancement is None:
            enable_enhancement = False if self.enhancer is not None else self.use_enhancer

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened():
            raise ValueError('无法打开视频文件')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))

        read_queue = Queue(maxsize=12)
        write_queue = Queue(maxsize=12)
        errors = {'reader': None, 'writer': None}

        def reader():
            try:
                idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    read_queue.put((idx, frame))
                    idx += 1
            except Exception as e:
                errors['reader'] = e
            finally:
                read_queue.put(None)

        def writer():
            try:
                while True:
                    item = write_queue.get()
                    if item is None:
                        break
                    _, frame = item
                    out.write(frame)
            except Exception as e:
                errors['writer'] = e

        reader_thread = threading.Thread(target=reader, daemon=True)
        writer_thread = threading.Thread(target=writer, daemon=True)
        reader_thread.start()
        writer_thread.start()

        detect_scale = 0.75 if max(fw, fh) >= 960 else 1.0
        frame_skip = 2 if not enable_enhancement else 1
        prev_gray = None
        prev_kps_all = None
        frame_idx = 0
        last_detect_frame = -999999

        try:
            while True:
                item = read_queue.get()
                if item is None:
                    break
                _, frame = item
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                use_tracking = prev_gray is not None and prev_kps_all is not None and (frame_idx - last_detect_frame) < frame_skip
                if use_tracking:
                    tracked_kps = self._track_keypoints(prev_gray, gray, prev_kps_all)
                    if tracked_kps is not None and len(tracked_kps) > 0:
                        dets, kps_all = self._kps_to_dets(tracked_kps, frame.shape)
                    else:
                        dets, kps_all = self._detect_with_scale(frame, detect_scale=detect_scale)
                        last_detect_frame = frame_idx
                else:
                    dets, kps_all = self._detect_with_scale(frame, detect_scale=detect_scale)
                    last_detect_frame = frame_idx

                if max_faces is not None and len(dets) > max_faces:
                    areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
                    keep = np.argsort(-areas)[:max_faces]
                    dets = dets[keep]
                    kps_all = kps_all[keep]

                result = frame.copy()
                for _, kps in zip(dets, kps_all):
                    result = self._swap_single_face(result, src_emb, kps, enable_enhancement=enable_enhancement)
                write_queue.put((frame_idx, result))

                prev_gray = gray
                prev_kps_all = np.asarray(kps_all, dtype=np.float32) if len(kps_all) > 0 else None
                frame_idx += 1
                if progress_cb and total > 0:
                    progress_cb(frame_idx / total)

            write_queue.put(None)
            reader_thread.join()
            writer_thread.join()
            if errors['reader']:
                raise errors['reader']
            if errors['writer']:
                raise errors['writer']
        finally:
            cap.release()
            out.release()
        return output_path

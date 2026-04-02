"""
FaceSwap WebUI - 基于 Gradio 的换脸界面
支持：图片换脸 / 视频换脸
"""
import os
import sys
import time
import traceback
import warnings
import logging
warnings.filterwarnings('ignore')
os.environ['ORT_DISABLE_ALL_LOGS'] = '1'
# 修复 Windows 控制台编码问题
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
import cv2
import numpy as np
import gradio as gr
from face_swapper import FaceSwapPipeline

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 全局单例，避免重复加载模型 ──────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = FaceSwapPipeline()
    return _pipeline


def load_image(path_or_arr):
    """加载图片，统一转为 BGR (OpenCV格式)"""
    if isinstance(path_or_arr, np.ndarray):
        # Gradio type="numpy" 传入的是 RGB，转为 BGR
        if path_or_arr.ndim == 3 and path_or_arr.shape[2] == 3:
            return cv2.cvtColor(path_or_arr, cv2.COLOR_RGB2BGR)
        return path_or_arr
    if isinstance(path_or_arr, str):
        return cv2.imread(path_or_arr)
    return None


# ── 图片换脸 ─────────────────────────────────────────────
def swap_image_fn(source_img, target_img, progress=gr.Progress()):
    if source_img is None:
        return None, "请上传来源人脸图片"
    if target_img is None:
        return None, "请上传目标图片"

    progress(0, desc="加载模型...")
    try:
        pipe = get_pipeline()
    except Exception as e:
        return None, f"模型加载失败: {e}"

    progress(0.2, desc="检测人脸...")
    try:
        src = load_image(source_img)
        tgt = load_image(target_img)
        if src is None:
            return None, "无法读取来源图片（请确认图片格式为 JPG/PNG）"
        if tgt is None:
            return None, "无法读取目标图片（请确认图片格式为 JPG/PNG）"

        print(f"来源图片: {src.shape}, 目标图片: {tgt.shape}")
        progress(0.5, desc="换脸中...")
        result = pipe.swap_image(src, tgt)
        progress(0.9, desc="保存结果...")

        out_path = os.path.join(OUTPUT_DIR, f"result_{int(time.time())}.jpg")
        cv2.imwrite(out_path, result)
        progress(1.0, desc="完成")
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result_rgb, f"换脸成功！已保存到: {out_path}"
    except Exception as e:
        err = traceback.format_exc()
        print("换脸错误:\n", err)
        return None, f"换脸失败: {e}\n\n{err}"


# ── 视频换脸 ─────────────────────────────────────────────
def swap_video_fn(source_img, target_video, progress=gr.Progress()):
    if source_img is None:
        return None, "请上传来源人脸图片"
    if target_video is None:
        return None, "请上传目标视频"

    progress(0, desc="加载模型...")
    try:
        pipe = get_pipeline()
    except Exception as e:
        return None, f"模型加载失败: {e}"

    try:
        src = load_image(source_img)
        if src is None:
            return None, "无法读取来源图片（请确认图片格式为 JPG/PNG）"

        out_path = os.path.join(OUTPUT_DIR, f"result_{int(time.time())}.mp4")

        def prog_cb(p):
            progress(0.1 + p * 0.85, desc=f"处理帧 {p*100:.0f}%...")

        progress(0.05, desc="开始处理视频...")
        pipe.swap_video(src, target_video, out_path, progress_cb=prog_cb)
        progress(1.0, desc="完成")
        return out_path, f"换脸完成！已保存到: {out_path}"
    except Exception as e:
        err = traceback.format_exc()
        print("视频换脸错误:\n", err)
        return None, f"视频换脸失败: {e}\n\n{err}"


# ── Gradio 界面 ──────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="本地换脸工具") as demo:
        gr.Markdown("""
# 本地换脸工具
**使用说明：**  
1. 上传**来源人脸图片**（提供要换入的人脸，建议正脸、清晰）  
2. 上传**目标图片或视频**（需要被换脸的内容）  
3. 点击**开始换脸**  

> ⚠️ CPU运行较慢，图片约10-30秒，视频按帧数估算
        """)

        with gr.Tabs():
            # ── 图片换脸 Tab ──────────────
            with gr.Tab("图片换脸"):
                with gr.Row():
                    with gr.Column():
                        src_img = gr.Image(label="来源人脸图片", type="numpy", height=300)
                        tgt_img = gr.Image(label="目标图片", type="numpy", height=300)
                        img_btn = gr.Button("开始换脸", variant="primary", size="lg")
                    with gr.Column():
                        img_output = gr.Image(label="换脸结果", height=600)
                        img_status = gr.Textbox(label="状态", interactive=False)

                img_btn.click(
                    fn=swap_image_fn,
                    inputs=[src_img, tgt_img],
                    outputs=[img_output, img_status],
                )

            # ── 视频换脸 Tab ──────────────
            with gr.Tab("视频换脸"):
                with gr.Row():
                    with gr.Column():
                        vsrc_img = gr.Image(label="来源人脸图片", type="numpy", height=300)
                        vtgt_video = gr.Video(label="目标视频", height=300)
                        vid_btn = gr.Button("开始换脸", variant="primary", size="lg")
                    with gr.Column():
                        vid_output = gr.Video(label="换脸结果视频")
                        vid_status = gr.Textbox(label="状态", interactive=False)

                vid_btn.click(
                    fn=swap_video_fn,
                    inputs=[vsrc_img, vtgt_video],
                    outputs=[vid_output, vid_status],
                )

        gr.Markdown("""
---
**模型说明：**
- 人脸检测: RetinaFace (det_10g.onnx)  
- 人脸识别: ArcFace (w600k_r50.onnx)  
- 换脸模型: InSwapper 128 (inswapper_128.onnx)
        """)

    return demo


if __name__ == '__main__':
    print("正在启动换脸 WebUI...")
    print("模型目录:", os.path.join(os.path.dirname(__file__), 'models'))
    print("输出目录:", OUTPUT_DIR)

    # 预加载模型
    print("预加载模型中（首次加载需要一点时间）...")
    try:
        get_pipeline()
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("启动后在界面中也会尝试加载")

    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )

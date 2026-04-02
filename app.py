"""
FaceSwap WebUI - 基于 Gradio 的换脸界面
支持：图片换脸 / 视频换脸 / 视频后台任务
"""
import os
import sys
import time
import uuid
import traceback
import warnings
import threading
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
_TASKS = {}
_TASKS_LOCK = threading.Lock()


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
    if isinstance(path_or_arr, dict):
        for key in ('path', 'image', 'name'):
            value = path_or_arr.get(key)
            if isinstance(value, str) and os.path.exists(value):
                return cv2.imread(value)
        return None
    if isinstance(path_or_arr, str):
        return cv2.imread(path_or_arr)
    return None


def load_video_path(video_value):
    """兼容 Gradio Video 可能返回的多种结构。"""
    if isinstance(video_value, str):
        return video_value
    if isinstance(video_value, dict):
        for key in ('path', 'video', 'name'):
            value = video_value.get(key)
            if isinstance(value, str):
                return value
    if isinstance(video_value, (list, tuple)) and video_value:
        first = video_value[0]
        if isinstance(first, str):
            return first
    return None


def create_task(enable_enhancement, max_faces):
    task_id = uuid.uuid4().hex[:12]
    task = {
        'task_id': task_id,
        'status': 'queued',
        'progress': 0.0,
        'message': '任务已创建，等待后台开始处理...',
        'output_path': None,
        'error': None,
        'enable_enhancement': bool(enable_enhancement),
        'max_faces': max_faces,
        'created_at': time.time(),
    }
    with _TASKS_LOCK:
        _TASKS[task_id] = task
    return task_id


def update_task(task_id, **kwargs):
    with _TASKS_LOCK:
        task = _TASKS.get(task_id)
        if not task:
            return
        task.update(kwargs)


def get_task(task_id):
    with _TASKS_LOCK:
        task = _TASKS.get(task_id)
        if not task:
            return None
        return dict(task)


def run_video_task(task_id, source_img, target_video_path, enable_enhancement, max_faces):
    try:
        update_task(
            task_id,
            status='running',
            progress=0.01,
            message='后台任务已启动，正在加载模型...',
        )
        pipe = get_pipeline()

        src = load_image(source_img)
        if src is None:
            raise ValueError('无法读取来源图片（请确认图片格式为 JPG/PNG）')
        if not target_video_path or not os.path.exists(target_video_path):
            raise ValueError('无法读取目标视频文件')

        out_path = os.path.join(OUTPUT_DIR, f"video_task_{task_id}.mp4")

        def prog_cb(p):
            p = max(0.0, min(1.0, float(p)))
            update_task(
                task_id,
                status='running',
                progress=p,
                message=f"后台处理中：{p * 100:.1f}%",
            )

        mode_name = '高质量模式' if enable_enhancement else '快速模式'
        scope_name = '主要人脸' if max_faces == 1 else '全部人脸'
        update_task(
            task_id,
            status='running',
            progress=0.03,
            message=f'正在执行视频换脸（{mode_name} / {scope_name}）...',
        )

        pipe.swap_video(
            src,
            target_video_path,
            out_path,
            progress_cb=prog_cb,
            enable_enhancement=enable_enhancement,
            max_faces=max_faces,
        )

        update_task(
            task_id,
            status='completed',
            progress=1.0,
            message=f'任务完成，结果已保存到: {out_path}',
            output_path=out_path,
        )
    except Exception as e:
        err = traceback.format_exc()
        print(f'视频后台任务失败 {task_id}:\n{err}')
        update_task(
            task_id,
            status='failed',
            progress=1.0,
            message=f'任务失败: {e}',
            error=err,
        )


# ── 图片换脸 ─────────────────────────────────────────────
def swap_image_fn(source_img, target_img, enable_enhancement=True, progress=gr.Progress()):
    if source_img is None:
        return None, '请上传来源人脸图片'
    if target_img is None:
        return None, '请上传目标图片'

    progress(0, desc='加载模型...')
    try:
        pipe = get_pipeline()
    except Exception as e:
        return None, f'模型加载失败: {e}'

    progress(0.2, desc='检测人脸...')
    try:
        src = load_image(source_img)
        tgt = load_image(target_img)
        if src is None:
            return None, '无法读取来源图片（请确认图片格式为 JPG/PNG）'
        if tgt is None:
            return None, '无法读取目标图片（请确认图片格式为 JPG/PNG）'

        progress(0.5, desc='换脸中...')
        result = pipe.swap_image(src, tgt, enable_enhancement=enable_enhancement)
        progress(0.9, desc='保存结果...')

        out_path = os.path.join(OUTPUT_DIR, f'result_{int(time.time())}.jpg')
        cv2.imwrite(out_path, result)
        progress(1.0, desc='完成')
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        speed_hint = '高质量模式' if enable_enhancement else '快速模式'
        return result_rgb, f'换脸成功（{speed_hint}）！已保存到: {out_path}'
    except Exception as e:
        err = traceback.format_exc()
        print('换脸错误:\n', err)
        return None, f'换脸失败: {e}\n\n{err}'


# ── 视频后台任务 ─────────────────────────────────────────
def start_video_task_fn(source_img, target_video, quality_mode='快速模式', face_scope='主要人脸'):
    if source_img is None:
        return '', '请上传来源人脸图片', 0.0, None, gr.update(active=False)
    if target_video is None:
        return '', '请上传目标视频', 0.0, None, gr.update(active=False)

    video_path = load_video_path(target_video)
    if not video_path or not os.path.exists(video_path):
        return '', '无法读取目标视频文件', 0.0, None, gr.update(active=False)

    src = load_image(source_img)
    if src is None:
        return '', '无法读取来源图片（请确认图片格式为 JPG/PNG）', 0.0, None, gr.update(active=False)

    enable_enhancement = quality_mode == '高质量模式'
    max_faces = 1 if face_scope == '主要人脸' else None
    task_id = create_task(enable_enhancement=enable_enhancement, max_faces=max_faces)
    src_copy = src.copy()

    worker = threading.Thread(
        target=run_video_task,
        args=(task_id, src_copy, video_path, enable_enhancement, max_faces),
        daemon=True,
    )
    worker.start()

    mode_desc = '高质量模式（更慢）' if enable_enhancement else '快速模式（推荐视频）'
    scope_desc = '只处理主要人脸（更快）' if max_faces == 1 else '处理全部人脸（更慢）'
    status = f'任务已提交到后台，任务ID: {task_id}，当前模式: {mode_desc}，范围: {scope_desc}'
    return task_id, status, 0.0, None, gr.update(active=True)


def query_video_task_fn(task_id):
    if not task_id:
        return '请输入任务ID', 0.0, None, gr.update(active=False)

    task = get_task(task_id.strip())
    if not task:
        return '任务不存在，请确认任务ID', 0.0, None, gr.update(active=False)

    status = task['message']
    if task.get('status') == 'failed' and task.get('error'):
        status = f"{status}\n\n{task['error']}"

    output_video = task['output_path'] if task.get('status') == 'completed' else None
    should_poll = task.get('status') in ('queued', 'running')
    return status, round(float(task.get('progress', 0.0)) * 100, 1), output_video, gr.update(active=should_poll)


# ── Gradio 界面 ──────────────────────────────────────────
def build_ui():
    with gr.Blocks(title='本地换脸工具') as demo:
        gr.Markdown("""
# 本地换脸工具
**使用说明：**  
1. 上传**来源人脸图片**（提供要换入的人脸，建议正脸、清晰）  
2. 上传**目标图片或视频**（需要被换脸的内容）  
3. 图片支持快速/高质量两种模式；视频默认建议使用快速模式  
4. 视频任务现在在**后台执行**，切换页面后可通过任务ID继续查看进度
        """)

        with gr.Tabs():
            with gr.Tab('图片换脸'):
                with gr.Row():
                    with gr.Column():
                        src_img = gr.Image(label='来源人脸图片', type='numpy', height=300)
                        tgt_img = gr.Image(label='目标图片', type='numpy', height=300)
                        img_enhance = gr.Checkbox(label='增强修复（更慢，画质更好）', value=True)
                        img_btn = gr.Button('开始换脸', variant='primary', size='lg')
                    with gr.Column():
                        img_output = gr.Image(label='换脸结果', height=600)
                        img_status = gr.Textbox(label='状态', interactive=False)

                img_btn.click(
                    fn=swap_image_fn,
                    inputs=[src_img, tgt_img, img_enhance],
                    outputs=[img_output, img_status],
                )

            with gr.Tab('视频换脸'):
                poll_timer = gr.Timer(value=2.0, active=False)
                with gr.Row():
                    with gr.Column():
                        vsrc_img = gr.Image(label='来源人脸图片', type='numpy', height=300)
                        vtgt_video = gr.Video(label='目标视频', height=300)
                        quality_mode = gr.Radio(
                            label='处理模式',
                            choices=['快速模式', '高质量模式'],
                            value='快速模式',
                        )
                        face_scope = gr.Radio(
                            label='处理范围',
                            choices=['主要人脸', '全部人脸'],
                            value='主要人脸',
                        )
                        vid_btn = gr.Button('提交后台任务', variant='primary', size='lg')
                    with gr.Column():
                        task_id_box = gr.Textbox(label='任务ID')
                        vid_progress = gr.Slider(label='处理进度(%)', minimum=0, maximum=100, step=0.1, value=0, interactive=False)
                        vid_status = gr.Textbox(label='任务状态', interactive=False, lines=8)
                        refresh_btn = gr.Button('刷新任务状态')
                        vid_output = gr.Video(label='换脸结果视频')

                vid_btn.click(
                    fn=start_video_task_fn,
                    inputs=[vsrc_img, vtgt_video, quality_mode, face_scope],
                    outputs=[task_id_box, vid_status, vid_progress, vid_output, poll_timer],
                    queue=False,
                )

                refresh_btn.click(
                    fn=query_video_task_fn,
                    inputs=[task_id_box],
                    outputs=[vid_status, vid_progress, vid_output, poll_timer],
                    queue=False,
                )

                poll_timer.tick(
                    fn=query_video_task_fn,
                    inputs=[task_id_box],
                    outputs=[vid_status, vid_progress, vid_output, poll_timer],
                    queue=False,
                    show_progress='hidden',
                )

        gr.Markdown("""
---
**模型说明：**
- 人脸检测: RetinaFace (det_10g.onnx)  
- 人脸识别: ArcFace (w600k_r50.onnx)  
- 换脸模型: InSwapper 128 (inswapper_128.onnx)  
- 增强模型: GFPGAN 1.4（可选，视频建议关闭以提升速度）
        """)

    return demo


if __name__ == '__main__':
    print('正在启动换脸 WebUI...')
    print('模型目录:', os.path.join(os.path.dirname(__file__), 'models'))
    print('输出目录:', OUTPUT_DIR)

    # 预加载模型
    print('预加载模型中（首次加载需要一点时间）...')
    try:
        get_pipeline()
        print('模型加载成功！')
    except Exception as e:
        print(f'模型加载失败: {e}')
        print('启动后在界面中也会尝试加载')

    demo = build_ui()
    demo.queue(default_concurrency_limit=8)
    demo.launch(
        server_name='127.0.0.1',
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )

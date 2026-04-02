"""
FaceSwap WebUI - 基于 Gradio 的换脸界面
支持：图片换脸 / 视频换脸 / 视频后台任务 / 批量图片换脸 / 邮件通知
"""
import os
import sys
import time
import uuid
import json
import traceback
import warnings
import threading
import zipfile
import smtplib
from email.mime.text import MIMEText
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

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

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
EMAIL_CONFIG_PATH = os.path.join(BASE_DIR, 'email_config.json')
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


def safe_filename(name):
    base = os.path.basename(name)
    keep = []
    for ch in base:
        if ch.isalnum() or ch in ('-', '_', '.', ' '):
            keep.append(ch)
        else:
            keep.append('_')
    return ''.join(keep).strip() or 'file'


def load_email_config():
    if not os.path.exists(EMAIL_CONFIG_PATH):
        return None
    try:
        with open(EMAIL_CONFIG_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        required = ['smtp_server', 'smtp_port', 'username', 'password', 'from_email', 'to_email']
        if not all(data.get(k) for k in required):
            return None
        return data
    except Exception:
        return None


def send_email_notification(task_id, task_name, result_path):
    cfg = load_email_config()
    if not cfg:
        return False, 'email_config.json 不存在或配置不完整'

    subject = f'FaceSwap 任务完成: {task_name}'
    body = (
        f'任务已完成。\n\n'
        f'任务类型: {task_name}\n'
        f'任务ID: {task_id}\n'
        f'结果路径: {result_path}\n'
        f'完成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}\n'
    )

    msg = MIMEText(body, 'plain', 'utf-8')
    msg['Subject'] = subject
    msg['From'] = cfg['from_email']
    msg['To'] = cfg['to_email']

    server = smtplib.SMTP(cfg['smtp_server'], int(cfg['smtp_port']), timeout=30)
    try:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(cfg['username'], cfg['password'])
        server.send_message(msg)
    finally:
        try:
            server.quit()
        except Exception:
            pass
    return True, '邮件已发送'


def send_email_notification_async(task_id, task_name, result_path):
    def worker():
        try:
            ok, message = send_email_notification(task_id, task_name, result_path)
            if ok:
                update_task(task_id, email_status=message)
            else:
                update_task(task_id, email_status=f'邮件未发送: {message}')
        except Exception as e:
            update_task(task_id, email_status=f'邮件发送失败: {e}')

    threading.Thread(target=worker, daemon=True).start()


def extract_file_path(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ('path', 'image', 'video', 'name'):
            value = item.get(key)
            if isinstance(value, str) and os.path.exists(value):
                return value
    if hasattr(item, 'name') and isinstance(item.name, str) and os.path.exists(item.name):
        return item.name
    return None


def normalize_file_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    paths = []
    for item in items:
        path = extract_file_path(item)
        if path:
            paths.append(path)
    return paths


def load_image(path_or_arr):
    """加载图片，统一转为 BGR (OpenCV格式)"""
    if isinstance(path_or_arr, np.ndarray):
        if path_or_arr.ndim == 3 and path_or_arr.shape[2] == 3:
            return cv2.cvtColor(path_or_arr, cv2.COLOR_RGB2BGR)
        return path_or_arr
    path = extract_file_path(path_or_arr)
    if path:
        return cv2.imread(path)
    return None


def load_video_path(video_value):
    return extract_file_path(video_value)


def create_task(task_type, **extra):
    task_id = uuid.uuid4().hex[:12]
    task = {
        'task_id': task_id,
        'task_type': task_type,
        'status': 'queued',
        'progress': 0.0,
        'message': '任务已创建，等待后台开始处理...',
        'output_path': None,
        'output_paths': [],
        'zip_path': None,
        'error': None,
        'email_status': None,
        'created_at': time.time(),
    }
    task.update(extra)
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


def build_zip_from_dir(task_dir, zip_path):
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(task_dir):
            for name in files:
                full_path = os.path.join(root, name)
                if os.path.abspath(full_path) == os.path.abspath(zip_path):
                    continue
                arcname = os.path.relpath(full_path, task_dir)
                zf.write(full_path, arcname)


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

        out_path = os.path.join(OUTPUT_DIR, f'video_task_{task_id}.mp4')

        def prog_cb(p):
            p = max(0.0, min(1.0, float(p)))
            update_task(
                task_id,
                status='running',
                progress=p,
                message=f'后台处理中：{p * 100:.1f}%',
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
            output_paths=[out_path],
        )
        send_email_notification_async(task_id, '视频换脸', out_path)
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


def run_batch_image_task(task_id, source_paths, target_paths, enable_enhancement, max_workers):
    try:
        pipe = get_pipeline()
        total = len(source_paths) * len(target_paths)
        if total <= 0:
            raise ValueError('来源图片和目标图片都不能为空')

        task_dir = os.path.join(OUTPUT_DIR, f'batch_{task_id}')
        os.makedirs(task_dir, exist_ok=True)
        zip_path = os.path.join(OUTPUT_DIR, f'batch_{task_id}.zip')

        update_task(
            task_id,
            status='running',
            progress=0.01,
            message=f'批量换脸任务已启动，共 {total} 张结果待生成...',
        )

        results = []
        counter = {'done': 0}
        counter_lock = threading.Lock()

        pairs = []
        for src_idx, src_path in enumerate(source_paths, start=1):
            for tgt_idx, tgt_path in enumerate(target_paths, start=1):
                pairs.append((src_idx, src_path, tgt_idx, tgt_path))

        def worker(job):
            src_idx, src_path, tgt_idx, tgt_path = job
            src_img = cv2.imread(src_path)
            tgt_img = cv2.imread(tgt_path)
            if src_img is None:
                raise ValueError(f'无法读取来源图片: {src_path}')
            if tgt_img is None:
                raise ValueError(f'无法读取目标图片: {tgt_path}')

            result = pipe.swap_image(
                src_img,
                tgt_img,
                enable_enhancement=enable_enhancement,
            )
            src_name = os.path.splitext(safe_filename(src_path))[0]
            tgt_name = os.path.splitext(safe_filename(tgt_path))[0]
            out_name = f'{src_idx:03d}_{src_name}__to__{tgt_idx:03d}_{tgt_name}.jpg'
            out_path = os.path.join(task_dir, out_name)
            cv2.imwrite(out_path, result)

            with counter_lock:
                counter['done'] += 1
                progress = counter['done'] / total
                results.append(out_path)
                update_task(
                    task_id,
                    progress=progress,
                    output_paths=list(results),
                    message=f'批量处理中：{counter["done"]}/{total}',
                )
            return out_path

        workers = max(1, min(int(max_workers), max(1, len(pairs)), max(1, os.cpu_count() or 4)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker, job) for job in pairs]
            for future in as_completed(futures):
                future.result()

        build_zip_from_dir(task_dir, zip_path)
        update_task(
            task_id,
            status='completed',
            progress=1.0,
            zip_path=zip_path,
            message=f'批量换脸完成，共生成 {total} 张结果，压缩包: {zip_path}',
        )
    except Exception as e:
        err = traceback.format_exc()
        print(f'批量图片任务失败 {task_id}:\n{err}')
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
        return '', '请上传来源人脸图片', 0.0, None, gr.update(active=False), ''
    if target_video is None:
        return '', '请上传目标视频', 0.0, None, gr.update(active=False), ''

    video_path = load_video_path(target_video)
    if not video_path or not os.path.exists(video_path):
        return '', '无法读取目标视频文件', 0.0, None, gr.update(active=False), ''

    src = load_image(source_img)
    if src is None:
        return '', '无法读取来源图片（请确认图片格式为 JPG/PNG）', 0.0, None, gr.update(active=False), ''

    enable_enhancement = quality_mode == '高质量模式'
    max_faces = 1 if face_scope == '主要人脸' else None
    task_id = create_task(
        'video',
        enable_enhancement=enable_enhancement,
        max_faces=max_faces,
    )
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
    return task_id, status, 0.0, None, gr.update(active=True), ''


def query_video_task_fn(task_id):
    if not task_id:
        return '请输入任务ID', 0.0, None, gr.update(active=False), ''

    task = get_task(task_id.strip())
    if not task:
        return '任务不存在，请确认任务ID', 0.0, None, gr.update(active=False), ''

    status = task['message']
    if task.get('email_status'):
        status += f'\n\n邮件状态: {task["email_status"]}'
    if task.get('status') == 'failed' and task.get('error'):
        status = f"{status}\n\n{task['error']}"

    output_video = task['output_path'] if task.get('status') == 'completed' else None
    should_poll = task.get('status') in ('queued', 'running')
    zip_path = task.get('zip_path') or ''
    return status, round(float(task.get('progress', 0.0)) * 100, 1), output_video, gr.update(active=should_poll), zip_path


# ── 批量图片后台任务 ─────────────────────────────────────
def start_batch_image_task_fn(source_files, target_files, enable_enhancement=True, worker_count=4):
    source_paths = normalize_file_list(source_files)
    target_paths = normalize_file_list(target_files)
    if not source_paths:
        return '', '请至少上传 1 张来源图片', 0.0, None, gr.update(active=False)
    if not target_paths:
        return '', '请至少上传 1 张目标图片', 0.0, None, gr.update(active=False)

    total = len(source_paths) * len(target_paths)
    task_id = create_task(
        'batch_image',
        enable_enhancement=bool(enable_enhancement),
        worker_count=int(worker_count),
        source_count=len(source_paths),
        target_count=len(target_paths),
        total_count=total,
    )

    worker = threading.Thread(
        target=run_batch_image_task,
        args=(task_id, source_paths, target_paths, bool(enable_enhancement), int(worker_count)),
        daemon=True,
    )
    worker.start()

    status = f'批量任务已提交，任务ID: {task_id}，source={len(source_paths)}，target={len(target_paths)}，总输出={total}'
    return task_id, status, 0.0, None, gr.update(active=True)


def query_batch_task_fn(task_id):
    if not task_id:
        return '请输入任务ID', 0.0, None, gr.update(active=False)

    task = get_task(task_id.strip())
    if not task:
        return '任务不存在，请确认任务ID', 0.0, None, gr.update(active=False)

    status = task['message']
    if task.get('status') == 'failed' and task.get('error'):
        status = f"{status}\n\n{task['error']}"

    zip_output = task.get('zip_path') if task.get('status') == 'completed' else None
    should_poll = task.get('status') in ('queued', 'running')
    return status, round(float(task.get('progress', 0.0)) * 100, 1), zip_output, gr.update(active=should_poll)


# ── Gradio 界面 ──────────────────────────────────────────
def build_ui():
    default_email_hint = '已配置' if load_email_config() else '未配置'

    with gr.Blocks(title='本地换脸工具') as demo:
        gr.Markdown(f"""
# 本地换脸工具
**使用说明：**  
1. 图片支持单张换脸、批量排列组合换脸  
2. 视频任务在**后台执行**，切换页面或关闭浏览器后不会中断  
3. 视频完成后支持**邮件通知**（当前邮箱配置：{default_email_hint}）  
4. 视频默认建议使用快速模式 + 主要人脸
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

            with gr.Tab('批量图片换脸'):
                batch_timer = gr.Timer(value=2.0, active=False)
                with gr.Row():
                    with gr.Column():
                        batch_src = gr.File(label='来源图片列表（可多选）', file_count='multiple', file_types=['image'])
                        batch_tgt = gr.File(label='目标图片列表（可多选）', file_count='multiple', file_types=['image'])
                        batch_enhance = gr.Checkbox(label='增强修复（更慢，画质更好）', value=False)
                        batch_workers = gr.Slider(label='并发任务数', minimum=1, maximum=max(2, min(8, os.cpu_count() or 4)), step=1, value=min(4, max(2, os.cpu_count() or 4)))
                        batch_btn = gr.Button('提交批量后台任务', variant='primary', size='lg')
                    with gr.Column():
                        batch_task_id = gr.Textbox(label='任务ID')
                        batch_progress = gr.Slider(label='处理进度(%)', minimum=0, maximum=100, step=0.1, value=0, interactive=False)
                        batch_status = gr.Textbox(label='任务状态', interactive=False, lines=8)
                        batch_refresh = gr.Button('刷新批量任务状态')
                        batch_zip = gr.File(label='批量结果压缩包')

                batch_btn.click(
                    fn=start_batch_image_task_fn,
                    inputs=[batch_src, batch_tgt, batch_enhance, batch_workers],
                    outputs=[batch_task_id, batch_status, batch_progress, batch_zip, batch_timer],
                    queue=False,
                )

                batch_refresh.click(
                    fn=query_batch_task_fn,
                    inputs=[batch_task_id],
                    outputs=[batch_status, batch_progress, batch_zip, batch_timer],
                    queue=False,
                )

                batch_timer.tick(
                    fn=query_batch_task_fn,
                    inputs=[batch_task_id],
                    outputs=[batch_status, batch_progress, batch_zip, batch_timer],
                    queue=False,
                    show_progress='hidden',
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
                        vid_zip_hidden = gr.Textbox(visible=False)

                vid_btn.click(
                    fn=start_video_task_fn,
                    inputs=[vsrc_img, vtgt_video, quality_mode, face_scope],
                    outputs=[task_id_box, vid_status, vid_progress, vid_output, poll_timer, vid_zip_hidden],
                    queue=False,
                )

                refresh_btn.click(
                    fn=query_video_task_fn,
                    inputs=[task_id_box],
                    outputs=[vid_status, vid_progress, vid_output, poll_timer, vid_zip_hidden],
                    queue=False,
                )

                poll_timer.tick(
                    fn=query_video_task_fn,
                    inputs=[task_id_box],
                    outputs=[vid_status, vid_progress, vid_output, poll_timer, vid_zip_hidden],
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
    print('模型目录:', os.path.join(BASE_DIR, 'models'))
    print('输出目录:', OUTPUT_DIR)

    print('预加载模型中（首次加载需要一点时间）...')
    try:
        get_pipeline()
        print('模型加载成功！')
    except Exception as e:
        print(f'模型加载失败: {e}')
        print('启动后在界面中也会尝试加载')

    demo = build_ui()
    demo.queue(default_concurrency_limit=16)
    demo.launch(
        server_name='127.0.0.1',
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )

"""
FaceSwap 守护进程
自动启动并监控 WebUI 和 frpc，崩溃自动重启
"""
import subprocess, time, os, sys, logging

BASE = r'd:\Downloads\faceswap'
PYTHON = r'D:\miniconda3\python.exe'
LOGS = os.path.join(BASE, 'logs')
os.makedirs(LOGS, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS, 'daemon.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.info

webui_proc = None
frpc_proc  = None

def start_webui():
    global webui_proc
    log('启动 WebUI...')
    with open(os.path.join(LOGS, 'webui.log'), 'a', encoding='utf-8') as out:
        webui_proc = subprocess.Popen(
            [PYTHON, os.path.join(BASE, 'app.py')],
            cwd=BASE,
            stdout=out,
            stderr=out,
        )
    log(f'WebUI PID={webui_proc.pid}')

def start_frpc():
    global frpc_proc
    log('启动 frpc...')
    with open(os.path.join(LOGS, 'frpc.log'), 'a', encoding='utf-8') as out:
        frpc_proc = subprocess.Popen(
            [os.path.join(BASE, 'frp', 'frpc.exe'),
             '-c', os.path.join(BASE, 'frp', 'frpc.toml')],
            cwd=os.path.join(BASE, 'frp'),
            stdout=out,
            stderr=out,
        )
    log(f'frpc PID={frpc_proc.pid}')

log('=== FaceSwap 守护进程启动 ===')
start_webui()
time.sleep(15)   # 等WebUI加载模型
start_frpc()

while True:
    time.sleep(30)

    # 检查 WebUI
    if webui_proc and webui_proc.poll() is not None:
        log(f'WebUI 已退出 (code={webui_proc.returncode})，3秒后重启...')
        time.sleep(3)
        start_webui()
        time.sleep(15)

    # 检查 frpc
    if frpc_proc and frpc_proc.poll() is not None:
        log(f'frpc 已退出 (code={frpc_proc.returncode})，5秒后重启...')
        time.sleep(5)
        start_frpc()

import argparse
import json
import os
import sys
import torch
from utils.helper import set_device
from src.PromptManager import PromptManager

def run_cogvideo(params):
    from src.CogVideoManager import CogVideoManager
    device = set_device()
    mgr = CogVideoManager(device, torch.bfloat16)
    if params.get('prompt'):
        mgr.set_prompt(params['prompt'])
    if params.get('model_path') or params.get('generate_type'):
        mp = params.get('model_path', 'THUDM/CogVideoX-2b')
        gt = params.get('generate_type', 't2v')
        mgr.set_model(mp, gt)
    mgr.set_output_layout(
        output_path=params.get('output_path', 'output_cogvideox'),
        width=int(params.get('width', 768)),
        height=int(params.get('height', 432)),
        fps=int(params.get('fps', 8)),
        num_frames=int(params.get('num_frames', 24)),
        num_inference_steps=int(params.get('num_inference_steps', 50))
    )
    ip = params.get('input_path')
    if ip:
        mgr.set_input_image_or_video(ip)
    mgr.generate()
    mgr.cleanup()

def run_hidream(params, prepare=False):
    from src.HiDreamManager import HiDreamManager
    device = set_device()
    mgr = HiDreamManager(device, torch.bfloat16)
    if params.get('prompt'):
        mgr.set_prompt(params['prompt'])
    if params.get('width') and params.get('height'):
        mgr.set_output_layout(int(params['width']), int(params['height']))
    if prepare:
        mgr.setup()
        mgr.cleanup()
        return
    mgr.setup()
    mgr.generate()
    mgr.cleanup()

def run_wan_video(params, prepare=False):
    from src.WanVideoManager import WanVideoManager
    device = set_device()
    mgr = WanVideoManager(device, torch.bfloat16)
    enable_i2v = bool(params.get('enable_i2v', False))
    if params.get('prompt'):
        mgr.set_prompt(params['prompt'])
    if enable_i2v and params.get('image_path'):
        mgr.set_image(params['image_path'])
    mgr.set_output_layout(
        output_path=params.get('output_path', 'output_wanvideo'),
        width=int(params.get('width', 832)),
        height=int(params.get('height', 480)),
        fps=int(params.get('fps', 8)),
        num_frames=int(params.get('num_frames', 25)),
        num_inference_steps=int(params.get('num_inference_steps', 30))
    )
    if prepare:
        mgr.setup(enable_i2v=enable_i2v)
        mgr.cleanup()
        return
    mgr.setup(enable_i2v=enable_i2v)
    mgr.generate(enable_i2v=enable_i2v)
    mgr.cleanup()

def run_ltxvideo(params, prepare=False):
    from src.LTXvideoManager import LTXVideoManager
    device = set_device()
    mgr = LTXVideoManager(device, torch.bfloat16)
    if params.get('prompt'):
        mgr.prompt = params['prompt']
    if params.get('negative_prompt'):
        mgr.negative_prompt = params['negative_prompt']
    mgr.width = int(params.get('width', 768))
    mgr.height = int(params.get('height', 512))
    mgr.frame_rate = int(params.get('frame_rate', 8))
    mgr.num_frames = int(params.get('num_frames', 17))
    mgr.num_inference_steps = int(params.get('num_inference_steps', 50))
    mgr.stg_mode = 'stg_a' if params.get('stg_mode', 'stg-a') == 'stg-a' else 'stg_r'
    mgr.stg_scale = float(params.get('stg_scale', 1.25))
    mgr.stg_rescale = float(params.get('stg_rescale', 0.7))
    mgr.stg_skip_layers = str(params.get('stg_skip_layers', '19'))
    ip = params.get('input_image')
    if ip:
        mgr.input_image_path = ip
    mgr.output_path = params.get('output_path', 'outputs')
    if prepare:
        mgr.check_models()
        return
    mgr.setup()
    mgr.generate()
    mgr.cleanup()

def run_mochi(params, prepare=False):
    from src.MochiManager import MochiManager
    device = set_device()
    mgr = MochiManager(device, torch.bfloat16)
    if params.get('prompt'):
        mgr.set_prompt(params['prompt'], params.get('negative_prompt', mgr.negative_prompt))
    mgr.set_output_layout(
        output_path=params.get('output_path', './mochi_output.mp4'),
        width=int(params.get('width', 432)),
        height=int(params.get('height', 256)),
        num_frames=int(params.get('num_frames', 7)),
        fps=int(params.get('fps', 6)),
        num_inference_steps=int(params.get('num_inference_steps', 64))
    )
    if prepare:
        mgr.download_weights(mgr.model_cache)
        return
    mgr.setup()
    mgr.generate()
    mgr.cleanup()

def run_hyvideo(params, prepare=False):
    from src.HunyuanVideoManager import HunyuanVideoManager
    device = set_device()
    mgr = HunyuanVideoManager(device, torch.bfloat16)
    if params.get('prompt'):
        mgr.set_prompt(params['prompt'])
    mgr.set_output_layout(
        output_path=params.get('output_path', 'output_hyvideo'),
        width=int(params.get('width', 480)),
        height=int(params.get('height', 352)),
        fps=int(params.get('fps', 15)),
        num_frames=int(params.get('num_frames', 61)),
        num_inference_steps=int(params.get('num_inference_steps', 30))
    )
    if prepare:
        return
    mgr.setup_low_mem()
    mgr.generate_low_mem()
    mgr.cleanup()

def run_cogvideofun(params, prepare=False):
    from src.CogVideoFunManager import CogVideoFunManager
    device = set_device()
    mgr = CogVideoFunManager(device, torch.bfloat16)
    if params.get('prompt'):
        mgr.prompt = params['prompt']
    mgr.video_size = [int(params.get('height', 272)), int(params.get('width', 480))]
    mgr.fps = int(params.get('fps', 8))
    mgr.num_frames = int(params.get('num_frames', 41))
    mgr.num_inference_steps = int(params.get('num_inference_steps', 50))
    mgr.output_path = params.get('output_path', 'output_cogvideox_fun')
    if prepare:
        mgr.download()
        return
    mgr.setup()
    mgr.run_t2v()
    mgr.cleanup()

def run_omnigen(params, prepare=False):
    from src.OmnigenManager import OmnigenManager
    device = set_device()
    mgr = OmnigenManager(device, torch.bfloat16)
    if params.get('prompt'):
        mgr.prompt = params['prompt']
    mgr.output_path = params.get('output_path', 'output_omnigen')
    if prepare:
        return
    mgr.setup()
    mgr.generate()
    mgr.cleanup()

def run_story_diffusion(params, prepare=False):
    script = os.path.join(os.getcwd(), 'run_story_diffusion.py')
    if prepare:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id='TencentARC/PhotoMaker', filename='photomaker-v1.bin')
        return
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    rc = subprocess_run_python(script)
    return rc

def run_flow_edit(params, prepare=False):
    script = os.path.join(os.getcwd(), 'run_flow_edit.py')
    args = ['--device_number', str(int(params.get('device_number', 0))), '--exp_yaml', params.get('exp_yaml', 'src/FlowEdit/FLUX_exp.yaml')]
    rc = subprocess_run_python(script, args)
    return rc

def run_rmbg(params, prepare=False):
    from src.RMBGManager import RMBGManager
    device = set_device()
    mgr = RMBGManager(device, torch.bfloat16)
    if prepare:
        return
    mgr.setup()
    mgr.generate()
    mgr.cleanup()

def run_realesrgan(params, prepare=False):
    script = os.path.join(os.getcwd(), 'run_realesrgan.py')
    rc = subprocess_run_python(script)
    return rc

def run_rife(params, prepare=False):
    script = os.path.join(os.getcwd(), 'run_rife.py')
    rc = subprocess_run_python(script)
    return rc

def run_kokoro_82m(params, prepare=False):
    script = os.path.join(os.getcwd(), 'run_kokoro_82m.py')
    rc = subprocess_run_python(script)
    return rc

def run_suno_bark(params, prepare=False):
    script = os.path.join(os.getcwd(), 'run_suno_bark.py')
    rc = subprocess_run_python(script)
    return rc

def run_mmaudio(params, prepare=False):
    script = os.path.join(os.getcwd(), 'run_mmaudio.py')
    rc = subprocess_run_python(script)
    return rc

def run_qwen_omni(params, prepare=False):
    script = os.path.join(os.getcwd(), 'run_qwen_omni.py')
    rc = subprocess_run_python(script)
    return rc

def run_glm(params, prepare=False):
    script = os.path.join(os.getcwd(), 'run_glm.py')
    rc = subprocess_run_python(script)
    return rc

def run_phi(params, prepare=False):
    script = os.path.join(os.getcwd(), 'run_phi.py')
    rc = subprocess_run_python(script)
    return rc

def run_deepseek_r1(params, prepare=False):
    script = os.path.join(os.getcwd(), 'run_deepseek_r1.py')
    rc = subprocess_run_python(script)
    return rc

def run_flux(params, prepare=False):
    script = os.path.join(os.getcwd(), 'run_flux.py')
    rc = subprocess_run_python(script)
    return rc

def subprocess_run_python(script, args=None):
    import subprocess
    cmd = [sys.executable, script]
    if args:
        cmd += args
    p = subprocess.Popen(cmd)
    rc = p.wait()
    return rc

RUNNERS = {
    'cogvideo': run_cogvideo,
    'hidream': run_hidream,
    'wan_video': run_wan_video,
    'ltxvideo': run_ltxvideo,
    'mochi': run_mochi,
    'hyvideo': run_hyvideo,
    'cogvideofun': run_cogvideofun,
    'omnigen': run_omnigen,
    'story_diffusion': run_story_diffusion,
    'flow_edit': run_flow_edit,
    'rmbg': run_rmbg,
    'realesrgan': run_realesrgan,
    'rife': run_rife,
    'kokoro_82m': run_kokoro_82m,
    'suno_bark': run_suno_bark,
    'mmaudio': run_mmaudio,
    'qwen_omni': run_qwen_omni,
    'glm': run_glm,
    'phi': run_phi,
    'deepseek_r1': run_deepseek_r1,
    'flux': run_flux,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    model = cfg['model']
    prepare = bool(cfg.get('prepare', False))
    params = cfg.get('params', {})
    if model not in RUNNERS:
        print(f'Unknown model {model}')
        sys.exit(1)
    fn = RUNNERS[model]
    rc = fn(params, prepare) if fn.__code__.co_argcount == 2 else fn(params)
    if isinstance(rc, int):
        sys.exit(rc)

if __name__ == '__main__':
    main()
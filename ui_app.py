import json
import os
import sys
import tempfile
import threading
import subprocess
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.properties import DictProperty, StringProperty, ObjectProperty, ListProperty
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.slider import Slider
from kivy.uix.spinner import Spinner
from kivy.uix.switch import Switch
from kivy.uix.filechooser import FileChooserIconView

KV = """
ScreenManager:
    id: sm
    HomeScreen:
        id: home
        name: 'home'
    ConfigScreen:
        id: config
        name: 'config'
    ExecuteScreen:
        id: exec
        name: 'exec'

<HomeScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: 10
        spacing: 10
        Label:
            text: 'Models'
            size_hint_y: None
            height: '40dp'
        ScrollView:
            GridLayout:
                id: model_list
                cols: 1
                size_hint_y: None
                height: self.minimum_height

<ConfigScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: 10
        spacing: 10
        Label:
            id: model_title
            text: root.model_name
            size_hint_y: None
            height: '50dp'
        ScrollView:
            GridLayout:
                id: form
                cols: 2
                size_hint_y: None
                height: self.minimum_height
                row_default_height: '48dp'
                row_force_default: True
        BoxLayout:
            size_hint_y: None
            height: '48dp'
            spacing: 10
            Button:
                text: 'Prepare'
                on_release: root.on_prepare()
            Button:
                text: 'Next'
                on_release: root.on_next()
            Button:
                text: 'Back'
                on_release: app.go_home()

<ExecuteScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: 10
        spacing: 10
        Label:
            id: exec_title
            text: root.model_name
            size_hint_y: None
            height: '40dp'
        BoxLayout:
            size_hint_y: None
            height: '48dp'
            spacing: 10
            Button:
                text: 'Run'
                on_release: root.on_run()
            Button:
                text: 'Cancel'
                on_release: root.on_cancel()
            Button:
                text: 'Home'
                on_release: app.go_home()
            Button:
                text: 'Exit'
                on_release: app.stop()
        TextInput:
            id: logs
            readonly: True
            multiline: True
"""

MODEL_REGISTRY = {
    'cogvideo': {
        'title': 'CogVideoX',
        'description': 'CogVideoX is an open-source video generation model originating from QingYing.',
        'prepare': False,
        'env': {},
        'params': [
            {'key': 'prompt', 'type': 'str', 'default': '', 'widget': 'text'},
            {'key': 'model_path', 'type': 'str', 'default': 'THUDM/CogVideoX-2b', 'widget': 'text'},
            {'key': 'generate_type', 'type': 'choice', 'choices': ['t2v','i2v','v2v'], 'default': 't2v'},
            {'key': 'width', 'type': 'int', 'min': 256, 'max': 1280, 'default': 768, 'widget': 'slider'},
            {'key': 'height', 'type': 'int', 'min': 256, 'max': 720, 'default': 432, 'widget': 'slider'},
            {'key': 'fps', 'type': 'int', 'min': 1, 'max': 30, 'default': 8, 'widget': 'slider'},
            {'key': 'num_frames', 'type': 'int', 'min': 1, 'max': 161, 'default': 24, 'widget': 'slider'},
            {'key': 'num_inference_steps', 'type': 'int', 'min': 1, 'max': 200, 'default': 50, 'widget': 'slider'},
            {'key': 'guidance_scale', 'type': 'float', 'min': 0.0, 'max': 20.0, 'default': 6.5, 'widget': 'slider'},
            {'key': 'input_path', 'type': 'path', 'default': '', 'widget': 'file'},
            {'key': 'output_path', 'type': 'str', 'default': 'output_cogvideox', 'widget': 'text'}
        ]
    },
    'hidream': {
        'title': 'HiDream',
        'description': 'HiDream-I1 is a diffusion-based image generation model by HiDream-ai.',
        'prepare': True,
        'env': {'HF_ENDPOINT': 'https://hf-mirror.com'},
        'params': [
            {'key': 'prompt', 'type': 'str', 'default': '', 'widget': 'text'},
            {'key': 'model_id', 'type': 'choice', 'choices': ['HiDream-ai/HiDream-I1-Dev','HiDream-ai/HiDream-I1-Full','HiDream-ai/HiDream-I1-Fast'], 'default': 'HiDream-ai/HiDream-I1-Dev'},
            {'key': 'width', 'type': 'int', 'min': 256, 'max': 1024, 'default': 512, 'widget': 'slider'},
            {'key': 'height', 'type': 'int', 'min': 256, 'max': 1024, 'default': 512, 'widget': 'slider'},
            {'key': 'guidance_scale', 'type': 'float', 'min': 0.0, 'max': 20.0, 'default': 5.0, 'widget': 'slider'},
            {'key': 'num_inference_steps', 'type': 'int', 'min': 1, 'max': 200, 'default': 28, 'widget': 'slider'},
            {'key': 'output_path', 'type': 'str', 'default': 'output_hidream', 'widget': 'text'}
        ]
    },
    'wan_video': {
        'title': 'Wan2.1',
        'description': 'Wan2.1 is a text-to-video and image-to-video diffusion model.',
        'prepare': True,
        'env': {'HF_ENDPOINT': 'https://hf-mirror.com', 'PYTORCH_ENABLE_MPS_FALLBACK': '1'},
        'params': [
            {'key': 'enable_i2v', 'type': 'bool', 'default': False, 'widget': 'toggle'},
            {'key': 'prompt', 'type': 'str', 'default': '', 'widget': 'text'},
            {'key': 'negative_prompt', 'type': 'str', 'default': '', 'widget': 'text'},
            {'key': 'width', 'type': 'int', 'min': 256, 'max': 1280, 'default': 832, 'widget': 'slider'},
            {'key': 'height', 'type': 'int', 'min': 256, 'max': 720, 'default': 480, 'widget': 'slider'},
            {'key': 'num_frames', 'type': 'int', 'min': 1, 'max': 65, 'default': 25, 'widget': 'slider'},
            {'key': 'fps', 'type': 'int', 'min': 1, 'max': 30, 'default': 8, 'widget': 'slider'},
            {'key': 'num_inference_steps', 'type': 'int', 'min': 1, 'max': 100, 'default': 30, 'widget': 'slider'},
            {'key': 'guidance_scale', 'type': 'float', 'min': 0.0, 'max': 20.0, 'default': 5.0, 'widget': 'slider'},
            {'key': 'image_path', 'type': 'path', 'default': '', 'widget': 'file'},
            {'key': 'output_path', 'type': 'str', 'default': 'output_wanvideo', 'widget': 'text'}
        ]
    },
    'ltxvideo': {
        'title': 'LTX-Video + STG',
        'description': 'LTX-Video with spatiotemporal guidance for video synthesis.',
        'prepare': True,
        'env': {},
        'params': [
            {'key': 'prompt', 'type': 'str', 'default': '', 'widget': 'text'},
            {'key': 'negative_prompt', 'type': 'str', 'default': '', 'widget': 'text'},
            {'key': 'width', 'type': 'int', 'min': 256, 'max': 1280, 'default': 768, 'widget': 'slider'},
            {'key': 'height', 'type': 'int', 'min': 256, 'max': 720, 'default': 512, 'widget': 'slider'},
            {'key': 'frame_rate', 'type': 'int', 'min': 1, 'max': 30, 'default': 8, 'widget': 'slider'},
            {'key': 'num_frames', 'type': 'int', 'min': 1, 'max': 257, 'default': 17, 'widget': 'slider'},
            {'key': 'num_inference_steps', 'type': 'int', 'min': 1, 'max': 200, 'default': 50, 'widget': 'slider'},
            {'key': 'stg_mode', 'type': 'choice', 'choices': ['stg-a','stg-r'], 'default': 'stg-a'},
            {'key': 'stg_scale', 'type': 'float', 'min': 0.0, 'max': 2.0, 'default': 1.25, 'widget': 'slider'},
            {'key': 'stg_rescale', 'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.7, 'widget': 'slider'},
            {'key': 'stg_skip_layers', 'type': 'str', 'default': '19', 'widget': 'text'},
            {'key': 'input_image', 'type': 'path', 'default': '', 'widget': 'file'},
            {'key': 'output_path', 'type': 'str', 'default': 'outputs', 'widget': 'text'}
        ]
    },
    'mochi': {
        'title': 'Mochi',
        'description': 'Mochi-1 is a video generative model by Genmo.',
        'prepare': True,
        'env': {},
        'params': [
            {'key': 'prompt', 'type': 'str', 'default': '', 'widget': 'text'},
            {'key': 'negative_prompt', 'type': 'str', 'default': 'worst quality, inconsistent motion, blurry, jittery, distorted', 'widget': 'text'},
            {'key': 'width', 'type': 'int', 'min': 416, 'max': 848, 'default': 432, 'widget': 'slider'},
            {'key': 'height', 'type': 'int', 'min': 240, 'max': 480, 'default': 256, 'widget': 'slider'},
            {'key': 'fps', 'type': 'int', 'min': 1, 'max': 30, 'default': 6, 'widget': 'slider'},
            {'key': 'num_frames', 'type': 'int', 'min': 1, 'max': 64, 'default': 7, 'widget': 'slider'},
            {'key': 'num_inference_steps', 'type': 'int', 'min': 1, 'max': 128, 'default': 64, 'widget': 'slider'},
            {'key': 'output_path', 'type': 'path', 'default': './mochi_output.mp4', 'widget': 'file'}
        ]
    },
    'hyvideo': {
        'title': 'HunyuanVideo',
        'description': 'HunyuanVideo is a text-to-video diffusion model by Tencent.',
        'prepare': True,
        'env': {'HF_ENDPOINT': 'https://hf-mirror.com'},
        'params': [
            {'key': 'prompt', 'type': 'str', 'default': '', 'widget': 'text'},
            {'key': 'width', 'type': 'int', 'min': 320, 'max': 1280, 'default': 480, 'widget': 'slider'},
            {'key': 'height', 'type': 'int', 'min': 240, 'max': 720, 'default': 352, 'widget': 'slider'},
            {'key': 'num_frames', 'type': 'int', 'min': 1, 'max': 61, 'default': 61, 'widget': 'slider'},
            {'key': 'num_inference_steps', 'type': 'int', 'min': 1, 'max': 200, 'default': 30, 'widget': 'slider'},
            {'key': 'fps', 'type': 'int', 'min': 1, 'max': 30, 'default': 15, 'widget': 'slider'},
            {'key': 'output_path', 'type': 'str', 'default': 'output_hyvideo', 'widget': 'text'}
        ]
    },
    'cogvideofun': {
        'title': 'CogVideoX-Fun',
        'description': 'CogVideoX-Fun is a lightweight runner for CogVideoX with web utilities.',
        'prepare': True,
        'env': {'PYTORCH_ENABLE_MPS_FALLBACK': '1'},
        'params': [
            {'key': 'prompt', 'type': 'str', 'default': '', 'widget': 'text'},
            {'key': 'width', 'type': 'int', 'min': 256, 'max': 1280, 'default': 768, 'widget': 'slider'},
            {'key': 'height', 'type': 'int', 'min': 256, 'max': 720, 'default': 432, 'widget': 'slider'},
            {'key': 'fps', 'type': 'int', 'min': 1, 'max': 30, 'default': 8, 'widget': 'slider'},
            {'key': 'num_frames', 'type': 'int', 'min': 1, 'max': 81, 'default': 24, 'widget': 'slider'},
            {'key': 'num_inference_steps', 'type': 'int', 'min': 1, 'max': 200, 'default': 50, 'widget': 'slider'},
            {'key': 'output_path', 'type': 'str', 'default': 'output_cogvideox_fun', 'widget': 'text'}
        ]
    },
    'omnigen': {
        'title': 'OmniGen',
        'description': 'OmniGen is a flexible image generation model by VectorSpaceLab.',
        'prepare': False,
        'env': {},
        'params': [
            {'key': 'prompt', 'type': 'str', 'default': '', 'widget': 'text'},
            {'key': 'output_path', 'type': 'str', 'default': 'output_omnigen', 'widget': 'text'}
        ]
    },
    'story_diffusion': {
        'title': 'StoryDiffusion',
        'description': 'StoryDiffusion generates coherent multi-image stories and offers a Gradio UI.',
        'prepare': True,
        'env': {},
        'params': [
            {'key': 'noop', 'type': 'str', 'default': '', 'widget': 'text'}
        ]
    },
    'flow_edit': {
        'title': 'FlowEdit',
        'description': 'FlowEdit performs image editing for FLUX or SD3.',
        'prepare': False,
        'env': {},
        'params': [
            {'key': 'device_number', 'type': 'int', 'min': 0, 'max': 8, 'default': 0, 'widget': 'slider'},
            {'key': 'exp_yaml', 'type': 'path', 'default': 'src/FlowEdit/FLUX_exp.yaml', 'widget': 'file'}
        ]
    },
    'rmbg': {
        'title': 'RMBG-2.0',
        'description': 'RMBG-2.0 removes image backgrounds.',
        'prepare': True,
        'env': {'HF_ENDPOINT': 'https://hf-mirror.com', 'PYTORCH_ENABLE_MPS_FALLBACK': '1'},
        'params': [
            {'key': 'output_path', 'type': 'str', 'default': 'output_rmbg', 'widget': 'text'}
        ]
    },
    'realesrgan': {
        'title': 'Real-ESRGAN',
        'description': 'Real-ESRGAN performs image/video super-resolution.',
        'prepare': False,
        'env': {},
        'params': [
            {'key': 'input_path', 'type': 'path', 'default': '', 'widget': 'file'},
            {'key': 'output_path', 'type': 'path', 'default': 'output_realesrgan.mp4', 'widget': 'file'}
        ]
    },
    'rife': {
        'title': 'RIFE',
        'description': 'RIFE performs video frame interpolation.',
        'prepare': False,
        'env': {},
        'params': [
            {'key': 'input_path', 'type': 'path', 'default': '', 'widget': 'file'},
            {'key': 'output_path', 'type': 'path', 'default': 'output_rife.mp4', 'widget': 'file'}
        ]
    },
    'kokoro_82m': {
        'title': 'Kokoro-82M',
        'description': 'Kokoro ONNX model for TTS audio generation.',
        'prepare': True,
        'env': {},
        'params': [
            {'key': 'text', 'type': 'str', 'default': 'Hello from Kokoro', 'widget': 'text'},
            {'key': 'output_path', 'type': 'path', 'default': 'output_kokoro.wav', 'widget': 'file'}
        ]
    },
    'suno_bark': {
        'title': 'Bark',
        'description': 'Bark is a text-to-audio model from Suno.',
        'prepare': False,
        'env': {'SUNO_ENABLE_MPS': '1', 'SUNO_OFFLOAD_CPU': '1', 'SUNO_USE_SMALL_MODELS': '1'},
        'params': [
            {'key': 'text', 'type': 'str', 'default': 'Hello from Bark', 'widget': 'text'},
            {'key': 'output_path', 'type': 'path', 'default': 'output_bark.wav', 'widget': 'file'}
        ]
    },
    'mmaudio': {
        'title': 'MMAudio',
        'description': 'MMAudio generates audio effects and music.',
        'prepare': False,
        'env': {},
        'params': [
            {'key': 'text', 'type': 'str', 'default': 'Metal clank', 'widget': 'text'},
            {'key': 'output_path', 'type': 'path', 'default': 'output_mmaudio.wav', 'widget': 'file'}
        ]
    },
    'qwen_omni': {
        'title': 'Qwen2.5-Omni',
        'description': 'Qwen2.5-Omni is a multimodal LLM generating audio and video.',
        'prepare': True,
        'env': {'HF_ENDPOINT': 'https://hf-mirror.com'},
        'params': [
            {'key': 'text', 'type': 'str', 'default': 'Describe the scene', 'widget': 'text'}
        ]
    },
    'glm': {
        'title': 'GLM-4-9B',
        'description': 'GLM-4-9B Chat model by THUDM.',
        'prepare': False,
        'env': {},
        'params': [
            {'key': 'text', 'type': 'str', 'default': 'Hello', 'widget': 'text'}
        ]
    },
    'phi': {
        'title': 'Phi-3.5-mini',
        'description': 'Phi-3.5-mini-instruct by Microsoft.',
        'prepare': False,
        'env': {},
        'params': [
            {'key': 'text', 'type': 'str', 'default': 'Hello', 'widget': 'text'}
        ]
    },
    'deepseek_r1': {
        'title': 'DeepSeek-R1-Distill',
        'description': 'DeepSeek-R1 distilled reasoning models.',
        'prepare': False,
        'env': {},
        'params': [
            {'key': 'text', 'type': 'str', 'default': 'Hello', 'widget': 'text'}
        ]
    },
    'flux': {
        'title': 'Flux.1',
        'description': 'Flux.1 image generation pipeline.',
        'prepare': False,
        'env': {},
        'params': [
            {'key': 'prompt', 'type': 'str', 'default': '', 'widget': 'text'},
            {'key': 'width', 'type': 'int', 'min': 256, 'max': 1024, 'default': 512, 'widget': 'slider'},
            {'key': 'height', 'type': 'int', 'min': 256, 'max': 1024, 'default': 512, 'widget': 'slider'},
            {'key': 'output_path', 'type': 'str', 'default': 'output_flux', 'widget': 'text'}
        ]
    }
}

SAFE_LIMITS = {
    'cogvideo': {
        'width': 720,
        'height': 480,
        'fps': 16,
        'num_frames': 49,
        'num_inference_steps': 100,
    },
    'cogvideofun': {
        'width': 512,
        'height': 320,
        'fps': 16,
        'num_frames': 41,
        'num_inference_steps': 100,
    },
    'hyvideo': {
        'width': 640,
        'height': 480,
        'fps': 15,
        'num_frames': 61,
        'num_inference_steps': 50,
    },
    'wan_video': {
        'width': 832,
        'height': 480,
        'fps': 12,
        'num_frames': 17,
        'num_inference_steps': 25,
    },
}

class HomeScreen(Screen):
    def on_enter(self):
        Clock.schedule_once(self._populate, 0)
    def _populate(self, dt):
        gl = self.ids.get('model_list') if hasattr(self, 'ids') else None
        if gl is None:
            return
        gl.clear_widgets()
        for k, spec in MODEL_REGISTRY.items():
            b = self._make_model_button(k, spec)
            gl.add_widget(b)
    def _make_model_button(self, key, spec):
        from kivy.uix.button import Button
        text = f"{spec['title']}: {spec['description']}"
        btn = Button(text=text, size_hint_y=None, height='48dp')
        def on_release(_):
            self.manager.transition = SlideTransition(direction='left')
            cs = self.manager.get_screen('config')
            cs.model_key = key
            cs.model_name = spec['title']
            cs.model_spec = spec
            cs.build_form()
            self.manager.current = 'config'
        btn.bind(on_release=on_release)
        return btn

class ConfigScreen(Screen):
    model_key = StringProperty('')
    model_name = StringProperty('')
    model_spec = DictProperty({})
    values = DictProperty({})
    def build_form(self):
        form = self.ids.form
        form.clear_widgets()
        self.values = {}
        for p in self.model_spec['params']:
            self._add_field(form, p)
    def _add_field(self, form, param):
        form.add_widget(Label(text=param['key']))
        w = self._make_widget(param)
        form.add_widget(w)
    def _make_widget(self, param):
        t = param.get('widget', 'text')
        if 'output_path' in param.get('key', ''):
            t = 'file'
        # initialize default in values
        if 'default' in param:
            self.values[param['key']] = param['default']
        if t == 'text':
            ti = TextInput(text=str(param.get('default', '')), multiline=False)
            def on_text(instance, value):
                self.values[param['key']] = value
            ti.bind(text=on_text)
            return ti
        if t == 'slider':
            mi = param.get('min', 0)
            limit = SAFE_LIMITS.get(self.model_key, {}).get(param['key'])
            ma = min(param.get('max', 100), limit) if limit else param.get('max', 100)
            de = param.get('default', mi)
            if de > ma:
                de = ma
            step = 16 if (self.model_key == 'wan_video' and param['key'] in ('width','height')) else (1 if param.get('type') == 'int' else 0)
            sl = Slider(min=mi, max=ma, value=de, step=step)
            val = Label(text=str(de), size_hint_x=None, width=60)
            box = BoxLayout(orientation='horizontal', spacing=10)
            box.add_widget(sl)
            box.add_widget(val)
            def on_value(instance, value):
                if param['type'] == 'int':
                    iv = int(round(value))
                    self.values[param['key']] = iv
                    val.text = str(iv)
                else:
                    fv = float(value)
                    self.values[param['key']] = fv
                    val.text = f"{fv:.2f}"
            sl.bind(value=on_value)
            return box
        if t == 'toggle':
            sw = Switch(active=bool(param.get('default', False)))
            def on_active(instance, value):
                self.values[param['key']] = bool(value)
            sw.bind(active=on_active)
            return sw
        if t == 'file':
            row = BoxLayout(orientation='horizontal', spacing=10)
            ti = TextInput(text=str(param.get('default','')), multiline=False)
            btn = Button(text='📁', size_hint_x=None, width=50)
            def open_picker(_):
                chooser = FileChooserIconView(path=os.getcwd())
                if 'output_path' in param['key']:
                    chooser.dirselect = True
                picker_box = BoxLayout(orientation='vertical', spacing=10)
                action_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=50, spacing=10)
                ok_btn = Button(text='Select')
                cancel_btn = Button(text='Cancel')
                popup = Popup(title='Select path', content=picker_box, size_hint=(0.9,0.9))
                def do_select(_btn):
                    sel = chooser.selection
                    if sel:
                        ti.text = sel[0]
                        self.values[param['key']] = sel[0]
                    popup.dismiss()
                def do_cancel(_btn):
                    popup.dismiss()
                ok_btn.bind(on_release=do_select)
                cancel_btn.bind(on_release=do_cancel)
                picker_box.add_widget(chooser)
                action_box.add_widget(ok_btn)
                action_box.add_widget(cancel_btn)
                picker_box.add_widget(action_box)
                popup.open()
            btn.bind(on_release=open_picker)
            def on_text(instance, value):
                self.values[param['key']] = value
            ti.bind(text=on_text)
            row.add_widget(ti)
            row.add_widget(btn)
            return row
        if t == 'choice':
            sp = Spinner(text=str(param.get('default')), values=param.get('choices', []), size_hint_x=1)
            def on_text(instance, value):
                self.values[param['key']] = value
            sp.bind(text=on_text)
            return sp
        return TextInput(text=str(param.get('default', '')), multiline=False)
    def on_prepare(self):
        app = App.get_running_app()
        app.prepare_model(self.model_key, self.values)
    def on_next(self):
        es = self.manager.get_screen('exec')
        es.model_key = self.model_key
        es.model_name = self.model_spec['title']
        es.model_env = self.model_spec.get('env', {})
        params = dict(self.values)
        for k, v in (SAFE_LIMITS.get(self.model_key, {}) or {}).items():
            if k in params:
                try:
                    params[k] = min(int(params[k]) if isinstance(params[k], int) else float(params[k]), v)
                except Exception:
                    pass
        es.params = params
        self.manager.transition = SlideTransition(direction='left')
        self.manager.current = 'exec'

class ExecuteScreen(Screen):
    model_key = StringProperty('')
    model_name = StringProperty('')
    model_env = DictProperty({})
    params = DictProperty({})
    process = ObjectProperty(None)
    log_lines = ListProperty([])
    def on_pre_enter(self):
        self.ids.logs.text = ''
        self.log_lines = []
    def on_run(self):
        app = App.get_running_app()
        app.run_model(self.model_key, self.params, self.model_env, self._on_output, self._on_done)
    def on_cancel(self):
        app = App.get_running_app()
        app.cancel_run()
    def _on_output(self, text):
        def apply(_dt):
            self.log_lines.append(text)
            self.ids.logs.text = "".join(self.log_lines[-200:])
        Clock.schedule_once(apply)
    def _on_done(self, rc):
        self._on_output(f"\nProcess finished with code {rc}\n")

class BoboApp(App):
    def build(self):
        return Builder.load_string(KV)
    def go_home(self):
        self.root.transition = SlideTransition(direction='right')
        self.root.current = 'home'
    def prepare_model(self, key, params):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        with open(tmp.name, 'w') as f:
            json.dump({'model': key, 'prepare': True, 'params': params}, f)
        env = dict(os.environ)
        env.update(MODEL_REGISTRY.get(key, {}).get('env', {}))
        p = subprocess.Popen([sys.executable, 'backend_runner.py', '--config', tmp.name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        def reader():
            for line in iter(p.stdout.readline, b''):
                pass
        threading.Thread(target=reader, daemon=True).start()
    def run_model(self, key, params, env_vars, on_output, on_done):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        with open(tmp.name, 'w') as f:
            json.dump({'model': key, 'prepare': False, 'params': params}, f)
        env = dict(os.environ)
        env.update(env_vars or {})
        env['PYTHONUNBUFFERED'] = '1'
        on_output('Starting model run\n')
        self._proc = subprocess.Popen(
            [sys.executable, '-u', 'backend_runner.py', '--config', tmp.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=1,
            text=True
        )
        def reader():
            for line in self._proc.stdout:
                try:
                    on_output(line)
                except Exception:
                    pass
            rc = self._proc.wait()
            Clock.schedule_once(lambda dt: on_done(rc))
        threading.Thread(target=reader, daemon=True).start()
    def cancel_run(self):
        if hasattr(self, '_proc') and self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

if __name__ == '__main__':
    BoboApp().run()
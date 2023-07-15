# Ultralytics YOLO 🚀, AGPL-3.0 license

import ast
import contextlib
import json
import platform
import zipfile
from collections import OrderedDict, namedtuple
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ultralytics.yolo.utils import LINUX, LOGGER, ROOT, yaml_load
from ultralytics.yolo.utils.checks import check_requirements, check_suffix, check_version, check_yaml
from ultralytics.yolo.utils.downloads import attempt_download_asset, is_url
from ultralytics.yolo.utils.ops import xywh2xyxy


def check_class_names(names):
    """Check class names. Map imagenet class codes to human-readable names if required. Convert lists to dicts."""
    if isinstance(names, list):  # names is a list
        names = dict(enumerate(names))  # convert to dict
    if isinstance(names, dict):
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(f'{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices '
                           f'{min(names.keys())}-{max(names.keys())} defined in your dataset YAML.')
        # imagenet class codes, i.e. 'n01440764'
        if isinstance(names[0], str) and names[0].startswith('n0'):
            # human-readable names
            map = yaml_load(ROOT / 'datasets/ImageNet.yaml')['map']
            names = {k: map[v] for k, v in names.items()}
    return names


class AutoBackend(nn.Module):

    def __init__(self,
                 weights='yolov8n.pt',
                 device=torch.device('cpu'),
                 dnn=False,
                 data=None,
                 fp16=False,
                 fuse=True,
                 verbose=True):
        """
        MultiBackend class for python inference on various platforms using Ultralytics YOLO.

        Args:
            weights (str): The path to the weights file. Default: 'yolov8n.pt'
            device (torch.device): The device to run the model on.
            dnn (bool): Use OpenCV DNN module for inference if True, defaults to False.
            data (str | Path | optional): Additional data.yaml file for class names.
            fp16 (bool): If True, use half precision. Default: False
            fuse (bool): Whether to fuse the model or not. Default: True
            verbose (bool): Whether to run in verbose mode or not. Default: True

        Supported formats and their naming conventions:
            | Format                | Suffix           |
            |-----------------------|------------------|
            | PyTorch               | *.pt             |
            | TorchScript           | *.torchscript    |
            | ONNX Runtime          | *.onnx           |
            | ONNX OpenCV DNN       | *.onnx dnn=True  |
            | OpenVINO              | *.xml            |
            | CoreML                | *.mlmodel        |
            | TensorRT              | *.engine         |
            | TensorFlow SavedModel | *_saved_model    |
            | TensorFlow GraphDef   | *.pb             |
            | TensorFlow Lite       | *.tflite         |
            | TensorFlow Edge TPU   | *_edgetpu.tflite |
            | PaddlePaddle          | *_paddle_model   |
        """
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, torch.nn.Module)
        pt = True
        fp16 &= pt
        # BHWC formats (vs torch BCWH)
        nhwc = False
        stride = 32  # default stride
        model, metadata = None, None
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        if not (pt or nn_module):
            w = attempt_download_asset(w)  # download if not local

        # NOTE: special case: in-memory pytorch model
        if nn_module:
            model = weights.to(device)
            model = model.fuse(verbose=verbose) if fuse else model
            if hasattr(model, 'kpt_shape'):
                kpt_shape = model.kpt_shape  # pose-only
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(
                model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
            pt = True
        elif pt:  # PyTorch
            from ultralytics.nn.tasks import attempt_load_weights
            model = attempt_load_weights(weights if isinstance(weights, list) else w,
                                         device=device,
                                         inplace=True,
                                         fuse=fuse)
            if hasattr(model, 'kpt_shape'):
                kpt_shape = model.kpt_shape  # pose-only
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(
                model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        else:
            from ultralytics.yolo.engine.exporter import export_formats
            raise TypeError(f"model='{w}' is not a supported model format. "
                            'See https://docs.ultralytics.com/modes/predict for help.'
                            f'\n\n{export_formats()}')

        # Load external metadata YAML
        if isinstance(metadata, (str, Path)) and Path(metadata).exists():
            metadata = yaml_load(metadata)
        if metadata:
            for k, v in metadata.items():
                if k in ('stride', 'batch'):
                    metadata[k] = int(v)
                elif k in ('imgsz', 'names', 'kpt_shape') and isinstance(v, str):
                    metadata[k] = eval(v)
            stride = metadata['stride']
            task = metadata['task']
            batch = metadata['batch']
            imgsz = metadata['imgsz']
            names = metadata['names']
            kpt_shape = metadata.get('kpt_shape')
        elif not (pt or triton or nn_module):
            LOGGER.warning(
                f"WARNING ⚠️ Metadata not found for 'model={weights}'")

        # Check names
        if 'names' not in locals():  # names missing
            names = self._apply_default_class_names(data)
        names = check_class_names(names)

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        """
        Runs inference on the YOLOv8 MultiBackend model.

        Args:
            im (torch.Tensor): The image tensor to perform inference on.
            augment (bool): whether to perform data augmentation during inference, defaults to False
            visualize (bool): whether to visualize the output predictions, defaults to False

        Returns:
            (tuple): Tuple containing the raw output tensor, and processed output for visualization (if visualize=True)
        """
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            # torch BCHW to numpy BHWC shape(1,320,192,3)
            im = im.permute(0, 2, 3, 1)

        if self.pt or self.nn_module:  # PyTorch
            y = self.model(
                im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.jit:  # TorchScript
            y = self.model(im)
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {
                                 self.session.get_inputs()[0].name: im})
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = list(self.executable_network([im]).values())
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings['images'].shape:
                i = self.model.get_binding_index('images')
                self.context.set_binding_shape(
                    i, im.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(
                    shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(
                        tuple(self.context.get_binding_shape(i)))
            s = self.bindings['images'].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:  # CoreML
            im = im[0].cpu().numpy()
            im_pil = Image.fromarray((im * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.BILINEAR)
            # coordinates are xywh normalized
            y = self.model.predict({'image': im_pil})
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] *
                                [[w, h, w, h]])  # xyxy pixels
                conf, cls = y['confidence'].max(
                    1), y['confidence'].argmax(1).astype(np.float)
                y = np.concatenate(
                    (box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            elif len(y) == 1:  # classification model
                y = list(y.values())
            elif len(y) == 2:  # segmentation model
                # reversed for segmentation models (pred, proto)
                y = list(reversed(y.values()))
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(
                x).copy_to_cpu() for x in self.output_names]
        elif self.triton:  # NVIDIA Triton Inference Server
            y = self.model(im)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(
                    im, training=False) if self.keras else self.model(im)
                if not isinstance(y, list):
                    y = [y]
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
                if len(y) == 2 and len(self.names) == 999:  # segments and names not defined
                    ip, ib = (0, 1) if len(y[0].shape) == 4 else (
                        1, 0)  # index of protos, boxes
                    # y = (1, 160, 160, 32), (1, 116, 8400)
                    nc = y[ib].shape[1] - y[ip].shape[3] - 4
                    self.names = {i: f'class{i}' for i in range(nc)}
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                # is TFLite quantized int8 model
                int8 = input['dtype'] == np.int8
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.int8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output['index'])
                    if int8:
                        scale, zero_point = output['quantization']
                        x = (x.astype(np.float32) - zero_point) * \
                            scale  # re-scale
                    y.append(x)
            # TF segment fixes: export is reversed vs ONNX export and protos are transposed
            if len(y) == 2:  # segment with (det, proto) output order reversed
                if len(y[1].shape) != 4:
                    # should be y = (1, 116, 8400), (1, 160, 160, 32)
                    y = list(reversed(y))
                # should be y = (1, 116, 8400), (1, 32, 160, 160)
                y[1] = np.transpose(y[1], (0, 3, 1, 2))
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            # y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        # for x in y:
        #     print(type(x), len(x)) if isinstance(x, (list, tuple)) else print(type(x), x.shape)  # debug shapes
        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """
         Convert a numpy array to a tensor.

         Args:
             x (np.ndarray): The array to be converted.

         Returns:
             (torch.Tensor): The converted tensor
         """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Warm up the model by running one forward pass with a dummy input.

        Args:
            imgsz (tuple): The shape of the dummy input tensor in the format (batch_size, channels, height, width)

        Returns:
            (None): This method runs the forward pass and don't return any value
        """
        warmup_types = self.pt, self.nn_module
        if any(warmup_types) and (self.device.type != 'cpu'):
            im = torch.empty(
                *imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _apply_default_class_names(data):
        """Applies default class names to an input YAML file or returns numerical class names."""
        with contextlib.suppress(Exception):
            return yaml_load(check_yaml(data))['names']
        # return default if above errors
        return {i: f'class{i}' for i in range(999)}

    @staticmethod
    def _model_type(p='path/to/model.pt'):
        """
        This function takes a path to a model file and returns the model type

        Args:
            p: path to the model file. Defaults to path/to/model.pt
        """
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from ultralytics.yolo.engine.exporter import export_formats
        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False) and not isinstance(p, str):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all(
            [any(s in url.scheme for s in ['http', 'grpc']), url.netloc])
        print("returning types", types + [triton])
        return types + [triton]

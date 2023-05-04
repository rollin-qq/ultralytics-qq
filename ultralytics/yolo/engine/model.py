# Ultralytics YOLO 🚀, GPL-3.0 license

from pathlib import Path

import sys
from ultralytics import yolo  # noqa
from ultralytics.nn.tasks import (ClassificationModel, DetectionModel, SegmentationModel, attempt_load_one_weight,
                                  guess_model_task)
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.engine.exporter import Exporter
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, callbacks, yaml_load
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.torch_utils import smart_inference_mode

# Map head to model, trainer, validator, and predictor classes
MODEL_MAP = {
    "classify": [
        ClassificationModel, 'yolo.TYPE.classify.ClassificationTrainer', 'yolo.TYPE.classify.ClassificationValidator',
        'yolo.TYPE.classify.ClassificationPredictor'],
    "detect": [
        DetectionModel, 'yolo.TYPE.detect.DetectionTrainer', 'yolo.TYPE.detect.DetectionValidator',
        'yolo.TYPE.detect.DetectionPredictor'],
    "segment": [
        SegmentationModel, 'yolo.TYPE.segment.SegmentationTrainer', 'yolo.TYPE.segment.SegmentationValidator',
        'yolo.TYPE.segment.SegmentationPredictor']}


class YOLO:
    """
    YOLO

    A python interface which emulates a model-like behaviour by wrapping trainers.
    """

    def __init__(self, model='yolov8n.yaml', type="v8") -> None:
        """
        Initializes the YOLO object.

        Args:
            model (str, Path): model to load or create
            type (str): Type/version of models to use. Defaults to "v8".
        """
        self.type = type
        self.ModelClass = None  # model class
        self.TrainerClass = None  # trainer class
        self.ValidatorClass = None  # validator class
        self.PredictorClass = None  # predictor class
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.task = None  # task type
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object

        # Load or create new YOLO model
        load_methods = {'.pt': self._load, '.yaml': self._new}
        suffix = Path(model).suffix
        if suffix in load_methods:
            {'.pt': self._load, '.yaml': self._new}[suffix](model)
        else:
            raise NotImplementedError(f"'{suffix}' model loading not implemented")

     #__call__ 方法提供了一种方便的方式来调用 predict 方法，
     # 因为它允许直接使用对象实例进行函数调用，而不必显式地指定方法名。
    def __call__(self, source=None, stream=False, **kwargs):
        return self.predict(source, stream, **kwargs)

    def _new(self, cfg: str, verbose=True):
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            verbose (bool): display model info on load
        """
        cfg = check_yaml(cfg)  # check YAML
        cfg_dict = yaml_load(cfg, append_filename=True)  # model dict
        self.task = guess_model_task(cfg_dict)
        self.ModelClass, self.TrainerClass, self.ValidatorClass, self.PredictorClass = \
            self._assign_ops_from_task(self.task)
        self.model = self.ModelClass(cfg_dict, verbose=verbose)  # initialize
        self.cfg = cfg

    def _load(self, weights: str):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
        """
        #调用 attempt_load_one_weight 函数加载模型，并将返回值分别赋给 self.model 和 self.ckpt
        self.model, self.ckpt = attempt_load_one_weight(weights)
        self.ckpt_path = weights
        self.task = self.model.args["task"]
        self.overrides = self.model.args  #将整个参数字典赋值给实例变量 self.overrides。
        self._reset_ckpt_args(self.overrides)
        self.ModelClass, self.TrainerClass, self.ValidatorClass, self.PredictorClass = \
            self._assign_ops_from_task(self.task)

    def reset(self):
        """
        Resets the model modules.
        """
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True

    def info(self, verbose=False):
        """
        Logs model info.

        Args:
            verbose (bool): Controls verbosity.
        """
        self.model.info(verbose=verbose)

    def fuse(self):
        self.model.fuse()

    @smart_inference_mode()
    def predict(self, source=None, stream=False, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (dict): The prediction results.
        """
        overrides = self.overrides.copy()
        overrides["conf"] = 0.25
        overrides.update(kwargs)
        overrides["mode"] = "predict"
        overrides["save"] = kwargs.get("save", False)  # get() 方法表示在 kwargs 中查找 "save" 的键值，如果没有找到则默认值为 False。
        #判断是否已经创建了 predictor 对象。若没有，则创建一个指定类型的 PredictorClass 对象，并通过调用 setup_model() 方法来设置模型；
        #否则，只需更新 predictor 对象的参数即可，此处使用 get_cfg() 方法来获取新的配置信息并更新到原有参数中。
        if not self.predictor:
            self.predictor = self.PredictorClass(overrides=overrides)
            self.predictor.setup_model(model=self.model)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)
        #根据程序运行时的文件名判断是否是命令行模式，如果是则调用 predict_cli() 方法进行预测;
        #否则直接调用 predict() 方法并传入 source 和 stream 参数进行预测，最终返回预测结果。
        is_cli = sys.argv[0].endswith('yolo') or sys.argv[0].endswith('ultralytics')
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    @smart_inference_mode()
    def val(self, data=None, **kwargs):
        """
        Validate a model on a given dataset .

        Args:
            data (str): The dataset to validate on. Accepts all formats accepted by yolo
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        overrides["mode"] = "val"
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.data = data or args.data
        args.task = self.task

        validator = self.ValidatorClass(args=args)
        validator(model=self.model)

    @smart_inference_mode()
    def export(self, **kwargs):
        """
        Export model.

        Args:
            **kwargs : Any other args accepted by the predictors. To see all args check 'configuration' section in docs
        """

        overrides = self.overrides.copy()
        overrides.update(kwargs)
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.task = self.task

        exporter = Exporter(overrides=args)
        exporter(model=self.model)

    def train(self, **kwargs):
        """
        Trains the model on a given dataset.

        Args:
            **kwargs (Any): Any number of arguments representing the training configuration.
        """
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        if kwargs.get("cfg"):
            LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
            overrides = yaml_load(check_yaml(kwargs["cfg"]), append_filename=True)
        overrides["task"] = self.task
        overrides["mode"] = "train"
        #获取"data"参数的值，如果该值不存在或为None，则意味着数据集文件路径未指定。
        if not overrides.get("data"):     #if not true/if none
            raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
        if overrides.get("resume"):
            overrides["resume"] = self.ckpt_path

        self.trainer = self.TrainerClass(overrides=overrides)
        if not overrides.get("resume"):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.train()
        # update model and cfg after training
        if RANK in {0, -1}:
            self.model, _ = attempt_load_one_weight(str(self.trainer.best))
            self.overrides = self.model.args

    def to(self, device):
        """
        Sends the model to the given device.

        Args:
            device (str): device
        """
        self.model.to(device)

    def _assign_ops_from_task(self, task):
        model_class, train_lit, val_lit, pred_lit = MODEL_MAP[task]
        # warning: eval is unsafe. Use with caution
        # 动态选择要实例化的类，eval可见字符串转成相应对象
        trainer_class = eval(train_lit.replace("TYPE", f"{self.type}"))     #将字符串train_lit中的"TYPE"替换为self.type的值
        validator_class = eval(val_lit.replace("TYPE", f"{self.type}"))
        predictor_class = eval(pred_lit.replace("TYPE", f"{self.type}"))

        return model_class, trainer_class, validator_class, predictor_class

    @property
    def names(self):
        """
         Returns class names of the loaded model.
        """
        return self.model.names

    @property
    def transforms(self):
        """
         Returns transform of the loaded model.
        """
        return self.model.transforms if hasattr(self.model, 'transforms') else None

    @staticmethod
    def add_callback(event: str, func):
        """
        Add callback
        """
        callbacks.default_callbacks[event].append(func)

    @staticmethod
    def _reset_ckpt_args(args):
        for arg in 'verbose', 'project', 'name', 'exist_ok', 'resume', 'batch', 'epochs', 'cache', 'save_json', \
                'half', 'v5loader':
            args.pop(arg, None)         #在列表args中删除索引位置为arg的元素，并返回该元素的值。如果列表args中不存在索引位置为arg的元素，则返回默认值None。

        args["device"] = ''  # set device to '' to prevent auto-DDP usage

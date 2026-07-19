import yaml
import argparse
from typing import Dict, Any, Optional
import os
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y", "on"):
        return True
    if v.lower() in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")

class ConfigNode:
    """包装字典，使其支持 attribute 访问"""
    def __init__(self, data: Dict[str, Any]):
        for k, v in data.items():
            if isinstance(v, dict):
                v = ConfigNode(v)
            setattr(self, k, v)

    def to_dict(self):
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigNode):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out

    def __repr__(self):
        return str(self.to_dict())


class Config:
    """超参数配置管理类，支持 YAML 加载 + CLI 覆盖，并提供属性访问"""

    def __init__(self, config_path: Optional[str] = None):
        self.config: Dict[str, Any] = {}
        if config_path:
            self.load_from_yaml(config_path)

        self.node = ConfigNode(self.config)

    def load_from_yaml(self, config_path: str) -> None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                self.config = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"YAML 文件解析错误: {e}")

    def update_from_cli(self) -> None:
        parser = argparse.ArgumentParser(description="超参数配置")
        parser.add_argument('--config', type=str, help='配置文件路径')

        # 第一次解析（先解析是否提供了 --config）
        known_args, remaining_args = parser.parse_known_args()

        if known_args.config:
            self.load_from_yaml(known_args.config)

        # 动态添加参数
        for arg in remaining_args:
            if arg.startswith('--'):
                key = arg[2:].replace('-', '_')
                # parser.add_argument(arg, type=self._get_value_type(key))
                value_type = self._get_value_type(key)
                if value_type is bool:
                    parser.add_argument(arg, type=str2bool)
                else:
                    parser.add_argument(arg, type=value_type)

        args = parser.parse_args()
        args_dict = vars(args)
        args_dict.pop('config', None)

        # 合并参数
        for key, value in args_dict.items():
            if value is not None:
                self._set_nested_key(key, value)

        # 更新 node（保持属性访问）
        self.node = ConfigNode(self.config)

    def _get_value_type(self, key: str) -> type:
        keys = key.split('.')
        current = self.config

        for k in keys[:-1]:
            if k not in current:
                return str
            current = current[k]

        last_key = keys[-1]
        if last_key in current:
            return type(current[last_key])

        return str

    def _set_nested_key(self, key: str, value: Any) -> None:
        keys = key.split('.')
        current = self.config

        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def get(self, key: str, default=None):
        keys = key.split('.')
        current = self.config

        for k in keys:
            if k not in current:
                return default
            current = current[k]

        return current

    def __getitem__(self, key):
        return self.get(key)

    def __getattr__(self, item):
        """允许直接 config.xxx"""
        if item == "node":
            return super().__getattribute__(item)
        if item in self.node.__dict__:
            return getattr(self.node, item)
        raise AttributeError(item)

    def __str__(self):
        return yaml.dump(self.config, sort_keys=False, allow_unicode=True)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, sort_keys=False, allow_unicode=True)

if __name__ == "__main__":
    config = Config("base_train_template.yaml")
    config.update_from_cli()
    print("当前配置:")
    print(config)
    print(config.wandb.enable)
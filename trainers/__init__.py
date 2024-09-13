import pkgutil
import importlib
from dassl.engine import TrainerX


# 遍历当前包内的所有模块
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    # 跳过基类模块本身，避免重复导入或循环导入
    # 动态导入模块
    module = importlib.import_module(f".{module_name}", __name__)
    # 检查模块中的所有属性，找到继承了TrainX的类
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if isinstance(attribute, type) and issubclass(attribute, TrainerX) and attribute is not TrainerX:
            # 将找到的类导入到当前模块的命名空间中
            globals()[attribute_name] = attribute

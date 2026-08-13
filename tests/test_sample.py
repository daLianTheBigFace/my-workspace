"""示例测试：确认包可正常导入、配置路径正确。"""

import intent_recognition
from intent_recognition.config import PROJECT_ROOT


def test_version():
    assert intent_recognition.__version__


def test_project_root():
    # src/intent_recognition 的父目录的父目录 = 项目根目录
    assert (PROJECT_ROOT / "pyproject.toml").exists()

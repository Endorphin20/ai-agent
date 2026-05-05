from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# @lru_cache(maxsize=1)实现单例模式，全局只加载一次配置
# Path 跨平台兼容，Windows 用 \，Mac/Linux 用 /，Path 会自动适配系统，不用手动写分隔符
# BaseSettings：Pydantic 官方配置类，自动读环境变量，继承它，定义的字段会自动映射环境变量

# 定位到项目根目录（当前文件往上 2 级），所有相对路径都会基于这个目录计算，保证路径计算不出错
BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"), # 自动从项目根目录 .env 加载环境变量
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 应用基础配置（全是Settings类的Field字段）
    app_name: str = Field(default="ai-agent", alias="APP_NAME")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8123, alias="APP_PORT")
    app_context_path: str = Field(default="/api", alias="APP_CONTEXT_PATH")
    # LLM的配置
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="glm-4.6v", alias="LLM_MODEL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="https://open.bigmodel.cn/api/paas/v4/", alias="LLM_BASE_URL")
    llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    # 向量嵌入模型的配置
    embedding_model: str = Field(default="", alias="EMBEDDING_MODEL")
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_base_url: str = Field(default="", alias="EMBEDDING_BASE_URL")
    # 外部服务配置
    search_api_key: str = Field(default="", alias="SEARCH_API_KEY")
    # 文件存储路径的配置
    docs_dir: Path = Field(default=Path("./data"), alias="DOCS_DIR")
    workspace_dir: Path = Field(default=Path("./workspace"), alias="WORKSPACE_DIR")
    # Agent运行限制的配置
    manus_max_steps: int = Field(default=20, alias="MANUS_MAX_STEPS")


@lru_cache(maxsize=1)  # 创建一个全局唯一的配置单例，整个程序运行期间，只调用 1 次 get_settings()，之后无论多少模块调用，都返回同一个对象。
def get_settings() -> Settings:
    # 创建 Settings 对象
    settings = Settings()
    # 相对路径 → 绝对路径
    if not settings.docs_dir.is_absolute():
        settings.docs_dir = BASE_DIR / settings.docs_dir
    if not settings.workspace_dir.is_absolute():
        settings.workspace_dir = BASE_DIR / settings.workspace_dir
    # 如果目录不存在，自动创建文件夹
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    return settings

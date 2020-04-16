from starlette.config import Config
from starlette.datastructures import Secret

config = Config("./envs/.env")

API_KEY: Secret = config("API_KEY", cast=Secret)
IS_DEBUG: bool = config("IS_DEBUG", cast=bool, default=False)

MODEL_PATH: str = config("MODEL_PATH")s
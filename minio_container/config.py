from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_s3_endpoint: str
    bucket_name: str

    class Config:
        env_file = ".env"

settings = Settings()

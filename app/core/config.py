from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_name: str = "genai-mega-no-ui"
    env: str = "dev"
    log_level: str = "INFO"
    
    #LLM settings(we will wire later)
    llm_provider : str = "stub" # or "local"
    llm_model: str = "dummy"

    #Rate limiting(we will implement Day 18)
    rate_limit_per_minute: int = 60

settings=Settings()
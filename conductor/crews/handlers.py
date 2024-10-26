from conductor.crews.cache import RedisCrewCacheHandler
from crewai.telemetry import Telemetry
from crewai.utilities import FileHandler, Logger, RPMController
from crewai import Crew
from crewai import LLM
from pydantic import PrivateAttr, model_validator


class RedisCacheHandlerCrew(Crew):
    _cache_handler: RedisCrewCacheHandler = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def set_private_attrs(self) -> "Crew":
        """Set private attributes."""
        self._cache_handler = RedisCrewCacheHandler()
        self._logger = Logger(verbose=self.verbose)
        if self.output_log_file:
            self._file_handler = FileHandler(self.output_log_file)
        self._rpm_controller = RPMController(max_rpm=self.max_rpm, logger=self._logger)
        if self.function_calling_llm:
            if isinstance(self.function_calling_llm, str):
                self.function_calling_llm = LLM(model=self.function_calling_llm)
            elif not isinstance(self.function_calling_llm, LLM):
                self.function_calling_llm = LLM(
                    model=getattr(self.function_calling_llm, "model_name", None)
                    or getattr(self.function_calling_llm, "deployment_name", None)
                    or str(self.function_calling_llm)
                )
        self._telemetry = Telemetry()
        self._telemetry.set_tracer()
        return self

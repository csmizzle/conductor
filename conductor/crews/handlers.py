from conductor.crews.cache import RedisCrewCacheHandler
from crewai.telemetry import Telemetry
from crewai.utilities import FileHandler, Logger, RPMController
from crewai import Crew
from pydantic import PrivateAttr, model_validator


class RedisCacheHandlerCrew(Crew):
    _cache_handler: RedisCrewCacheHandler = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def set_private_attrs(self) -> "RedisCacheHandlerCrew":
        """Set private attributes."""
        self._cache_handler = RedisCrewCacheHandler()
        self._logger = Logger(self.verbose)
        if self.output_log_file:
            self._file_handler = FileHandler(self.output_log_file)
        self._rpm_controller = RPMController(max_rpm=self.max_rpm, logger=self._logger)
        self._telemetry = Telemetry()
        self._telemetry.set_tracer()
        self._telemetry.crew_creation(self)
        return self

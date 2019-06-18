import os

from hypothesis import HealthCheck
from hypothesis import unlimited
from hypothesis import settings
from hypothesis import Verbosity


settings.register_profile(name="single", deadline=400.0, max_examples=1)

settings.register_profile(name="dev", deadline=400.0, max_examples=100)

settings.register_profile(
    name="ci",
    deadline=1000.0,
    max_examples=300,
    suppress_health_check=[HealthCheck.too_slow],
    timeout=unlimited,
    verbosity=Verbosity.debug,
)

settings.load_profile(name=os.getenv("HYPOTHESIS_PROFILE", "dev"))

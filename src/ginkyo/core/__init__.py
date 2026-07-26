# Re-export model types for ``from ginkyo.core import Recording``.
from ginkyo.core.model import Channel, Recording, SeriesMeta
from ginkyo.core.project import Project, Series

__all__ = ["Channel", "Recording", "SeriesMeta", "Project", "Series"]

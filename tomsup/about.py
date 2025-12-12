__title__ = "tomsup"
__download_url__ = "https://github.com/KennethEnevoldsen/tomsup"

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

__version__ = version("tomsup")

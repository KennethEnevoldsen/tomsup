Here are some simple instructions for getting the psychopy experiment to run.

1) Install pandas and tomsup with "pip install pandas" and "pip install tomsup", respectively
2) Install psychopy. This can also be done with pip install. Note that a lot of dependencies are installed this way, so in case one wants to avoid that, follow 2.a
    2.a) psychopy can be installed with "pip install psychopy --no-deps" (as by the website https://www.psychopy.org/download.html#pip-install). If this is done, some dependencies might need to be installed manually.
    These include
    future / pyyaml / wxpython / json_tricks / pyglet / python-bidi / freetype-py / requests
    Note also that the freetype library needs to be installed. If it isn't this can be done with homebrew on macs, and a google search should show how to do it on a windows.
3) Make sure that the 'images' folder is in the same directory as the python script, and that it contains the six images that it contains on tomsup's github repo.
the repo can be found here: https://github.com/KennethEnevoldsen/tomsup/tree/master/tutorials/psychopy_experiment

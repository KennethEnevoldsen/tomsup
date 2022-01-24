Here are some simple instructions for getting the psychopy experiment to run.

1) Install tomsup with "pip install tomsup"

2) Install psychopy. This can also be done with pip install. Note that a lot of dependencies are installed this way, so in case one wants to avoid that, follow 2.a
    2.a) psychopy can be installed with "pip install psychopy --no-deps" (as by the website https://www.psychopy.org/download.html#pip-install). If this is done, some dependencies might need to be installed manually.
    These include
    future / pyyaml / wxpython / json_tricks / pyglet / python-bidi / freetype-py / requests
    Note also that the freetype library needs to be installed. If it isn't this can be done with homebrew on macs, and a google search should show how to do it on a windows.

3) Make sure to download the folder containing the python script as well as the 'images' folder.
Make sure that the 'images' folder contains the six images as on the github repo, and that the 'data' folder is empty or nonexistent.
The repo can be found here: https://github.com/KennethEnevoldsen/tomsup/tree/master/tutorials/psychopy_experiment

The python script should now run.
Note that:
- saving internal states is only possible for TOM and QL agents.
- the syntax for specifying opponent parameters needs to follow the structure used by tomsup. For example:
        {'bias': 0.7} #for an RB agent
        {'volatility': -2, 'b_temp': -1, 'bias': 0, 'dilution': None} #for a TOM agent'
    Unspecified parameter values are set to tomsup's defaults.
- on some computers, there might be an issue with closing Psychopy after the experiment is finished. Here, cmd+Q or ctrl+Q can solve it 
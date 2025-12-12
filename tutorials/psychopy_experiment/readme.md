Here are some simple instructions for getting the psychopy experiment to run.

1) Install tomsup and psychopy using with `pip install tomsup psychopy`. 

1.1) Troubles installing psychopy? Psychopy can be installed with "pip install psychopy --no-deps" (as by the website https://www.psychopy.org/download.html#pip-install). If this is done, some dependencies might need to be installed manually.
    These include
    future / pyyaml / wxpython / json_tricks / pyglet / python-bidi / freetype-py / requests. These can be installed by running `pip install future pyyaml wxpython json_tricks "pyglet>=1.5.11,<1.6.0" python-bidi freetype-py requests`.
    Note also that the freetype library needs to be installed as well. If it isn't this can e.g. be done with homebrew on Macs using `brew install freetype`.

2) Make sure to download the folder containing the python script as well as the 'images' folder. This can be done using the command

```
# download folder
git clone https://github.com/KennethEnevoldsen/tomsup/

# move to folder
cd tomsup/tutorials/psychopy_experiment
```

Make sure that the 'images' folder contains the six images as on the github repo, and that the 'data' folder is empty or nonexistent.
The repo can be found [here](https://github.com/KennethEnevoldsen/tomsup/tree/main/tutorials/psychopy_experiment).

The python script should now run.

Note that:
- saving internal states is only possible for TOM and QL agents.
- the syntax for specifying opponent parameters needs to follow the structure used by tomsup. For example:
        {'bias': 0.7} #for an RB agent
        {'volatility': -2, 'b_temp': -1, 'bias': 0, 'dilution': None} #for a TOM agent'
    Unspecified parameter values are set to tomsup's defaults.
- on some computers, there might be an issue with closing Psychopy after the experiment is finished. Here, cmd+Q or ctrl+Q can solve it 

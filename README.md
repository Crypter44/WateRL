# BT Sammet

You can currently find two Modelica water networks in this repository. They were created with `Dymola 2024x` and compiled as FMUs for Windows and Linux. If you face problems while using the FMUs, test the corresponding unit via [this website](https://fmu-check.herokuapp.com/).

Furthermore the code in the folder `code` shows how to interact with the FMU using [sofirpy](https://sofirpy.readthedocs.io/en/stable/). It may be helpful to have a look at [fmpy](https://github.com/CATIA-Systems/FMPy)

## Modelica Models
- `simple_network_valve` consists of a pump, which is operated at a constant speed, and two sinks. There is a valve in front of one sink, the opening angle of which can be changed to vary the flow rate in the corresponding pipe.
# BT Sammet

You can currently find two Modelica water networks in this repository. They were created with `Dymola 2024x` and compiled as FMUs for Windows and Linux. If you face problems while using the FMUs, test the corresponding unit via [this website](https://fmu-check.herokuapp.com/).

Furthermore the code in the folder `code` shows how to interact with the FMU using [sofirpy](https://sofirpy.readthedocs.io/en/stable/). It may be helpful to have a look at [fmpy](https://github.com/CATIA-Systems/FMPy)

## Modelica Models
- `simple_network_valve` consists of a pump, which is operated at a constant speed, and two sinks. There is a valve in front of one sink, the opening angle of which can be changed to vary the flow rate in the corresponding pipe.
- `circular_water_network` consists of four valves and two pumps arranged around a circle of pipes. The pumps can be controlled by the rotational speed in the range from 0 to 1. The valves are connected to a PI controller which converts the input (volume flow in m^3/h) into a opening
of the valve (ranging from 0 to 1).
- `circular_water_network_wo_PI` is similar to `circular_water_network` except that the PI controller at the valves have beem removed. This means both the pumps and the valves can get inputs from 0 to 1. When opening the valves fully and
speeding up the pumps to their maximal speed valve_2 gets 1.36 m^3/h, valve_3 gets 1.29 m^3/h, valve_5 gets 1.32 m^3/h and valve_6 gets 1.33 m^3/h.
The pumps pressure is calculated with
$$
\Delta p_\mathrm{pump}= \alpha_1 Q^2 + \alpha_2 Q n + \alpha_3 n^2
$$
(volume flow $Q$ in m^3/s, $n \in [0, 1]$ und $p$ in m) for the pumps used in this model $\alpha_1 = -0.065158$, $\alpha_2 = 0.34196$ and $\alpha_3 = 8.1602$.

The pump's power consumtion si calculated as
$$
P_\mathrm{pump} = \beta_1 Q^3 + \beta_2 Q^2 n + \beta_3 Q n^2 + \beta_4 n^3+\beta_5
$$
$n \in [0, 1]$, volume flow $Q$ in m^3/h and $P$ in W. In this model $\beta_1 = -0.14637$, $\beta_2 = 1.1881$, $\beta_3=23.0824$, $\beta_4 = 53.0304$ and $\beta_5 = 6.0431$.

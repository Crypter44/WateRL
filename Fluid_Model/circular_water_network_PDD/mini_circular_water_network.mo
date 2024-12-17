within ;
model mini_circular_water_network
  "System consisting of tree pumps and multiple sinks (consumer)."

  //replaceable package Medium = Modelica.Media.Water.StandardWaterOnePhase
    //constrainedby Modelica.Media.Interfaces.PartialMedium;
  replaceable package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater
    constrainedby Modelica.Media.Interfaces.PartialMedium;

  inner Modelica.Fluid.System system(
    p_ambient=99000,
    T_ambient=293.15,
    g=9.81,
    energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
    m_flow_start=0)
    annotation (Placement(transformation(extent={{208,30},{228,50}})));

  Custom_Pump_V2.BaseClasses_Custom.Pump_vs pump_1(
    redeclare package Medium = Medium,
    allowFlowReversal=true,
    m_flow_start=0,
    redeclare function flowCharacteristic =
        Custom_Pump_V2.BaseClasses_Custom.PumpCharacteristics.quadraticFlow (c={
            -0.065158,0.34196,8.1602}),
    redeclare function powerCharacteristic =
        Custom_Pump_V2.BaseClasses_Custom.PumpCharacteristics.cubicPower (c={-0.14637,
            1.1881,23.0824,53.0304,6.0431}),
    checkValve=false,
    rpm_rel=0.93969,
    use_N_in=true,
    V=0.1,
    use_HeatTransfer=false,
    redeclare model HeatTransfer =
        Modelica.Fluid.Vessels.BaseClasses.HeatTransfer.IdealHeatTransfer)
    "Magna3_25_80"
    annotation (Placement(transformation(extent={{-36,-124},{-16,-104}})));

    //energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial, //only needed for Medium = Modelica.Media.Water.StandardWaterOnePhase
    //massDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial, //only needed for Medium = Modelica.Media.Water.StandardWaterOnePhase

    Custom_Pump_V2.BaseClasses_Custom.Pump_vs pump_4(
    redeclare package Medium = Medium,
    allowFlowReversal=true,
    m_flow_start=0,
    redeclare function flowCharacteristic =
        Custom_Pump_V2.BaseClasses_Custom.PumpCharacteristics.quadraticFlow (c=
            {-0.065158,0.34196,8.1602}),
    redeclare function powerCharacteristic =
        Custom_Pump_V2.BaseClasses_Custom.PumpCharacteristics.cubicPower (c={-0.14637,
            1.1881,23.0824,53.0304,6.0431}),
    checkValve=false,
    rpm_rel=0.93969,
    use_N_in=true,
    V=0.1,
    use_HeatTransfer=false,
    redeclare model HeatTransfer =
        Modelica.Fluid.Vessels.BaseClasses.HeatTransfer.IdealHeatTransfer)
    "Magna3_25_80"
    annotation (Placement(transformation(extent={{10,10},{-10,-10}},
        rotation=90,
        origin={88,240})));

  Modelica.Fluid.Sources.FixedBoundary source_1(
    nPorts=1,
    redeclare package Medium = Medium,
    use_T=true,
    T=system.T_ambient,
    p=system.p_ambient)
    annotation (Placement(transformation(extent={{-174,-124},{-154,-104}})));

  Modelica.Fluid.Sources.FixedBoundary source_4(
    nPorts=1,
    redeclare package Medium = Medium,
    use_T=true,
    T=system.T_ambient,
    p=system.p_ambient)
    annotation (Placement(transformation(extent={{46,294},{66,314}})));

  PressureDrivenDemand_smooth sink_2(
    redeclare package Medium = Medium,
    P0=10000,
    Pf=50000)
    annotation (Placement(transformation(extent={{-146,40},{-166,60}})));

  PressureDrivenDemand_smooth sink_3(
    redeclare package Medium = Medium,
    P0=10000,
    Pf=50000)
    annotation (Placement(transformation(extent={{-154,198},{-174,218}})));

  PressureDrivenDemand_smooth sink_5(
    redeclare package Medium = Medium,
    P0=10000,
    Pf=50000)
    annotation (Placement(transformation(extent={{218,128},{238,148}})));

  PressureDrivenDemand_smooth sink_6(
    redeclare package Medium = Medium,
    P0=10000,
    Pf=50000)
    annotation (Placement(transformation(extent={{220,-48},{240,-28}})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_1(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={-6,0})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_2(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=270,
        origin={-8,38})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_3(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=90,
        origin={-8,138})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_4(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=270,
        origin={-8,166})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_5(redeclare package
      Medium = Medium)
    annotation (Placement(transformation(extent={{90,148},{110,128}})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_6(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=180,
        origin={100,0})));

  Modelica.Fluid.Pipes.StaticPipe pipe_1(
    allowFlowReversal=true,
    length=30,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium,
    height_ab=0)
    annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-8,-34})));

  Modelica.Fluid.Pipes.StaticPipe pipe_2(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-40,40},{-60,60}})));

  Modelica.Fluid.Pipes.StaticPipe pipe_3(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-8,76})));

  Modelica.Fluid.Pipes.StaticPipe pipe_4(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-38,198},{-58,218}})));

  Modelica.Fluid.Pipes.StaticPipe pipe_5(
    allowFlowReversal=true,
    length=30,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium,
    height_ab=0)
    annotation (Placement(transformation(extent={{60,176},{40,196}})));
  Modelica.Fluid.Pipes.StaticPipe pipe_6(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{40,128},{60,148}})));

  Modelica.Fluid.Pipes.StaticPipe pipe_7(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{9,-11},{-9,11}},
        rotation=90,
        origin={101,77})));

  Modelica.Fluid.Pipes.StaticPipe pipe_8(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{40,-10},{60,10}})));

  Modelica.Fluid.Sensors.Pressure pressure_2(redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-118,66},{-138,86}})));
  Modelica.Fluid.Sensors.Pressure pressure_3(redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-124,226},{-144,246}})));
  Modelica.Fluid.Sensors.Pressure pressure_5(redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{186,152},{206,172}})));
  Modelica.Fluid.Sensors.Pressure pressure_6(redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{190,-76},{210,-96}})));

  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_1(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-92,-124},{-72,-104}})));

  Modelica.Fluid.Sensors.RelativePressure pressure_1(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-30,-72},{-50,-92}})));

  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_2(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-66,40},{-86,60}})));

  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_3(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-66,198},{-86,218}})));

  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_4(
    redeclare package Medium = Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=-90,
        origin={88,278})));

  Modelica.Fluid.Sensors.RelativePressure pressure_4(
    redeclare package Medium = Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=90,
        origin={42,244})));

  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_5(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{120,148},{140,128}})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_6(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{122,-28},{142,-48}})));

 Modelica.Blocks.Interfaces.RealOutput V_flow_1
    "Connector of Real output signal containing input signal u in another unit"
    annotation (Placement(transformation(extent={{-180,-74},{-200,-94}})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_1 "Relative pressure signal"
    annotation (Placement(transformation(extent={{-180,-74},{-200,-54}})));
  Modelica.Blocks.Interfaces.RealInput w_p_1
    "Prescribed rotational speed"
    annotation (Placement(transformation(extent={{-208,-62},{-168,-22}})));

  Modelica.Blocks.Interfaces.RealOutput V_flow_2
    "Connector of Real output signal containing input signal u in another unit"
    annotation (Placement(transformation(extent={{-178,106},{-198,126}})));
  Modelica.Blocks.Interfaces.RealOutput p_abs_2 "Pressure at port"
    annotation (Placement(transformation(extent={{-176,66},{-196,86}})));
  Modelica.Blocks.Interfaces.RealOutput V_flow_3
    "Connector of Real output signal containing input signal u in another unit"
    annotation (Placement(transformation(extent={{-176,270},{-196,290}})));
 Modelica.Blocks.Interfaces.RealOutput p_abs_3 "Pressure at port"
    annotation (Placement(transformation(extent={{-176,226},{-196,246}})));
  Modelica.Blocks.Interfaces.RealOutput V_flow_4
    "Volume flow rate from port_a to port_b" annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={2,322})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_4 "Relative pressure signal"
    annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=-90,
        origin={-28,324})));
  Modelica.Blocks.Interfaces.RealInput w_p_4
    "=1: completely open, =0: completely closed" annotation (Placement(
        transformation(
        extent={{20,-20},{-20,20}},
        rotation=90,
        origin={140,326})));

  Modelica.Blocks.Interfaces.RealOutput V_flow_5
    "Volume flow rate from port_a to port_b"
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=0,
        origin={326,88})));
  Modelica.Blocks.Interfaces.RealOutput p_abs_5 "Pressure at port"
    annotation (Placement(transformation(extent={{314,152},{334,172}})));
  Modelica.Blocks.Interfaces.RealOutput V_flow_6
    "Volume flow rate from port_a to port_b"
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={132,-144})));
  Modelica.Blocks.Interfaces.RealOutput p_abs_6 "Pressure at port"
    annotation (Placement(transformation(extent={{314,-96},{334,-76}})));

  Modelica.Blocks.Math.UnitConversions.To_bar to_bar annotation (Placement(transformation(extent={{-64,-74},
            {-84,-54}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar4 annotation (Placement(
        transformation(
        extent={{10,-10},{-10,10}},
        rotation=-90,
        origin={-28,266})));

  To_m3hr to_m3hr  annotation (Placement(transformation(extent={{-90,-94},{-110,
            -74}})));
  To_m3hr to_m3hr2 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-76,100})));
  To_m3hr to_m3hr3 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-76,252})));
  To_m3hr to_m3hr4
    annotation (Placement(transformation(extent={{34,268},{14,288}})));
  To_m3hr to_m3hr5 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={198,88})));

  Modelica.Blocks.Interfaces.RealOutput P_pum_1
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={-26,-144})));
  Modelica.Blocks.Interfaces.RealOutput P_pum_4 annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={96,320})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay(delayTime=1)
    annotation (Placement(transformation(extent={{-150,-52},{-130,-32}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay5(delayTime=0.1) annotation (
      Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={138,284})));
  Modelica.Blocks.Logical.GreaterThreshold greaterThreshold(threshold=0.01)
    annotation (Placement(transformation(extent={{-108,-40},{-88,-20}})));
  Modelica.Blocks.Logical.Switch switch1
    annotation (Placement(transformation(extent={{-62,-12},{-42,8}})));
  Modelica.Blocks.Sources.RealExpression realExpression(y=0)
    annotation (Placement(transformation(extent={{-62,-42},{-42,-22}})));
  Modelica.Fluid.Valves.ValveDiscrete valveDiscreteRamp(
    redeclare package Medium = Medium,
    allowFlowReversal=true,
    dp_nominal=1,
    m_flow_nominal=100)
    annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={16,-92})));
  Modelica.Blocks.Logical.GreaterThreshold greaterThreshold1(threshold=0.01)
    annotation (Placement(transformation(extent={{160,248},{180,268}})));
  Modelica.Blocks.Logical.Switch switch2
    annotation (Placement(transformation(extent={{140,220},{120,240}})));
  Modelica.Fluid.Valves.ValveDiscrete valveDiscreteRamp1(
    redeclare package Medium = Medium,
    allowFlowReversal=true,
    dp_nominal=1,
    m_flow_nominal=100)
    annotation (Placement(transformation(
        extent={{10,10},{-10,-10}},
        rotation=90,
        origin={88,208})));
  Modelica.Blocks.Sources.RealExpression realExpression1(y=0)
    annotation (Placement(transformation(extent={{174,206},{154,226}})));
  Modelica.Fluid.Sensors.MassFlowRate massFlowRate(redeclare package Medium =
        Medium)
    annotation (Placement(transformation(extent={{-92,40},{-112,60}})));
  Modelica.Blocks.Interfaces.RealOutput m_flow_2
    "Mass flow rate from port_a to port_b"
    annotation (Placement(transformation(extent={{-178,86},{-198,106}})));
  Modelica.Fluid.Sensors.MassFlowRate massFlowRate1(redeclare package Medium =
        Medium)
    annotation (Placement(transformation(extent={{-94,198},{-114,218}})));
  Modelica.Blocks.Interfaces.RealOutput m_flow_3
    "Mass flow rate from port_a to port_b"
    annotation (Placement(transformation(extent={{-176,248},{-196,268}})));
  Modelica.Fluid.Sensors.MassFlowRate massFlowRate2(redeclare package Medium =
        Medium)
    annotation (Placement(transformation(extent={{154,128},{174,148}})));
  Modelica.Blocks.Interfaces.RealOutput m_flow5
    "Mass flow rate from port_a to port_b"
    annotation (Placement(transformation(extent={{314,168},{334,188}})));
  Modelica.Fluid.Sensors.MassFlowRate massFlowRate3(redeclare package Medium =
        Medium)
    annotation (Placement(transformation(extent={{158,-48},{178,-28}})));
  Modelica.Blocks.Interfaces.RealOutput m_flow6
    "Mass flow rate from port_a to port_b"
    annotation (Placement(transformation(extent={{314,-28},{334,-8}})));
  To_m3s to_m3s_2 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-156,20})));
  To_m3s to_m3s_6
    annotation (Placement(transformation(extent={{274,-68},{254,-48}})));
  To_m3s to_m3s
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-164,178})));
  To_m3s to_m3s_5
    annotation (Placement(transformation(extent={{276,112},{256,132}})));

  Modelica.Blocks.Interfaces.RealInput w_v_2
    "Connector of Real input signal to be converted"
    annotation (Placement(transformation(extent={{-210,-24},{-170,16}})));
  Modelica.Blocks.Interfaces.RealInput w_v_3
    "Connector of Real input signal to be converted"
    annotation (Placement(transformation(extent={{-212,132},{-172,172}})));
  Modelica.Blocks.Interfaces.RealInput w_v_4
    "Connector of Real input signal to be converted" annotation (Placement(
        transformation(
        extent={{-20,-20},{20,20}},
        rotation=180,
        origin={326,122})));
  Modelica.Blocks.Interfaces.RealInput w_v_6
    "Connector of Real input signal to be converted"
    annotation (Placement(transformation(extent={{340,-78},{300,-38}})));
  To_m3hr to_m3hr1 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={132,-92})));
equation
  P_pum_1 = pump_1.W_total;
  P_pum_4 = pump_4.W_total;
  connect(volumeFlow_1.port_a, source_1.ports[1])
    annotation (Line(points={{-92,-114},{-154,-114}},
                                                   color={0,127,255}));
  connect(pipe_1.port_b,idealJunction_1. port_2)
    annotation (Line(points={{-8,-24},{-8,-10},{-6,-10}},
                                                 color={0,127,255}));
  connect(p_rel_4, p_rel_4)
    annotation (Line(points={{-28,324},{-28,324}},   color={0,0,127}));
  connect(pressure_1.port_a, pipe_1.port_a) annotation (Line(points={{-30,-82},
          {-8,-82},{-8,-44}},                    color={0,127,255}));
  connect(to_bar.y, p_rel_1)
    annotation (Line(points={{-85,-64},{-190,-64}}, color={0,0,127}));
  connect(to_bar.u, pressure_1.p_rel)
    annotation (Line(points={{-62,-64},{-40,-64},{-40,-73}}, color={0,0,127}));
  connect(pressure_4.p_rel, to_bar4.u)
    annotation (Line(points={{33,244},{-28,244},{-28,254}},
                                                        color={0,0,127}));
  connect(to_bar4.y, p_rel_4) annotation (Line(points={{-28,277},{-28,324}},
                                color={0,0,127}));
  connect(to_m3hr.u, volumeFlow_1.V_flow)
    annotation (Line(points={{-88,-84},{-82,-84},{-82,-103}},color={0,0,127}));
  connect(volumeFlow_4.V_flow, to_m3hr4.u)
    annotation (Line(points={{77,278},{36,278}}, color={0,0,127}));
  connect(pipe_4.port_b, volumeFlow_3.port_a)
    annotation (Line(points={{-58,208},{-66,208}},  color={0,127,255}));
  connect(volumeFlow_3.V_flow, to_m3hr3.u)
    annotation (Line(points={{-76,219},{-76,240}},
                                                 color={0,0,127}));
  connect(to_m3hr3.y, V_flow_3)
    annotation (Line(points={{-76,263},{-76,280},{-186,280}},color={0,0,127}));

  connect(to_m3hr.y, V_flow_1)
    annotation (Line(points={{-111,-84},{-190,-84}}, color={0,0,127}));
  connect(to_m3hr4.y, V_flow_4) annotation (Line(points={{13,278},{2,278},{2,
          322}},            color={0,0,127}));
  connect(V_flow_5, to_m3hr5.y) annotation (Line(points={{326,88},{209,88}},
                           color={0,0,127}));
  connect(volumeFlow_5.V_flow, to_m3hr5.u)
    annotation (Line(points={{130,127},{130,88},{186,88}},   color={0,0,127}));
  connect(pipe_2.port_b, volumeFlow_2.port_a)
    annotation (Line(points={{-60,50},{-66,50}}, color={0,127,255}));
  connect(volumeFlow_2.V_flow, to_m3hr2.u)
    annotation (Line(points={{-76,61},{-76,88}}, color={0,0,127}));
  connect(to_m3hr2.y, V_flow_2)
    annotation (Line(points={{-76,111},{-76,116},{-188,116}},color={0,0,127}));
  connect(idealJunction_2.port_2, idealJunction_1.port_1)
    annotation (Line(points={{-8,28},{-8,10},{-6,10}}, color={0,127,255}));
  connect(pipe_3.port_a, idealJunction_2.port_1)
    annotation (Line(points={{-8,66},{-8,48}}, color={0,127,255}));
  connect(pipe_8.port_b,idealJunction_6. port_2)
    annotation (Line(points={{60,0},{90,0}}, color={0,127,255}));
  connect(idealJunction_6.port_1,volumeFlow_6. port_a)
    annotation (Line(points={{110,0},{118,0},{118,-38},{122,-38}},
                                               color={0,127,255}));
  connect(idealJunction_3.port_1, pipe_3.port_b)
    annotation (Line(points={{-8,128},{-8,86}}, color={0,127,255}));
  connect(idealJunction_3.port_2, idealJunction_4.port_2)
    annotation (Line(points={{-8,148},{-8,156}}, color={0,127,255}));

  connect(idealJunction_1.port_3,pipe_8. port_a)
    annotation (Line(points={{4,0},{40,0}}, color={0,127,255}));
  connect(pipe_2.port_a, idealJunction_2.port_3)
    annotation (Line(points={{-40,50},{-24,50},{-24,38},{-18,38}},
                                                 color={0,127,255}));
  connect(idealJunction_3.port_3, pipe_6.port_a)
    annotation (Line(points={{2,138},{40,138}}, color={0,127,255}));
  connect(pipe_7.port_b,idealJunction_6. port_3)
    annotation (Line(points={{101,68},{100,64},{100,10}},
                                                 color={0,127,255}));
  connect(pipe_4.port_a, idealJunction_4.port_3)
    annotation (Line(points={{-38,208},{-24,208},{-24,166},{-18,166}},
                                                   color={0,127,255}));
  connect(pipe_5.port_b, idealJunction_4.port_1) annotation (Line(points={{40,186},
          {-8,186},{-8,176}},                   color={0,127,255}));
  connect(volumeFlow_1.port_b, pump_1.port_a)
    annotation (Line(points={{-72,-114},{-36,-114}},
                                                   color={0,127,255}));
  connect(pressure_1.port_b, pump_1.port_a) annotation (Line(points={{-50,-82},
          {-60,-82},{-60,-114},{-36,-114}},
                                         color={0,127,255}));
  connect(volumeFlow_4.port_b, pump_4.port_a)
    annotation (Line(points={{88,268},{88,250}}, color={0,127,255}));
  connect(pressure_4.port_b, pump_4.port_a) annotation (Line(points={{42,254},{
          42,260},{88,260},{88,250}}, color={0,127,255}));
  connect(pressure_4.port_a, pipe_5.port_a) annotation (Line(points={{42,234},{
          42,202},{66,202},{66,186},{60,186}}, color={0,127,255}));
  connect(volumeFlow_4.port_a, source_4.ports[1])
    annotation (Line(points={{88,288},{88,304},{66,304}},  color={0,127,255}));
  connect(V_flow_4, V_flow_4)
    annotation (Line(points={{2,322},{2,322}},       color={0,0,127}));
  connect(idealJunction_5.port_1, pipe_6.port_b)
    annotation (Line(points={{90,138},{60,138}}, color={0,127,255}));
  connect(idealJunction_5.port_2, volumeFlow_5.port_a)
    annotation (Line(points={{110,138},{120,138}}, color={0,127,255}));
  connect(idealJunction_5.port_3, pipe_7.port_a) annotation (Line(points={{100,128},
          {101,124},{101,86}}, color={0,127,255}));
  connect(fixedDelay.u, w_p_1)
    annotation (Line(points={{-152,-42},{-188,-42}}, color={0,0,127}));
  connect(fixedDelay5.u, w_p_4)
    annotation (Line(points={{138,296},{140,296},{140,326}},
                                                   color={0,0,127}));
  connect(switch1.u1, fixedDelay.y) annotation (Line(points={{-64,6},{-120,6},{-120,
          -42},{-129,-42}}, color={0,0,127}));
  connect(greaterThreshold.y, switch1.u2) annotation (Line(points={{-87,-30},{-76,
          -30},{-76,-2},{-64,-2}}, color={255,0,255}));
  connect(greaterThreshold.u, fixedDelay.y) annotation (Line(points={{-110,-30},
          {-120,-30},{-120,-42},{-129,-42}}, color={0,0,127}));
  connect(realExpression.y, switch1.u3) annotation (Line(points={{-41,-32},{-34,
          -32},{-34,-22},{-70,-22},{-70,-10},{-64,-10}}, color={0,0,127}));
  connect(switch1.y, pump_1.N_in) annotation (Line(points={{-41,-2},{-24,-2},{-24,
          -104},{-26,-104}},                                         color={0,0,
          127}));
  connect(pump_1.port_b, valveDiscreteRamp.port_a) annotation (Line(points={{-16,
          -114},{16,-114},{16,-102}}, color={0,127,255}));
  connect(valveDiscreteRamp.port_b, pipe_1.port_a) annotation (Line(points={{16,
          -82},{16,-50},{-8,-50},{-8,-44}}, color={0,127,255}));
  connect(valveDiscreteRamp.open, greaterThreshold.y) annotation (Line(points={{8,-92},
          {-20,-92},{-20,-46},{-68,-46},{-68,-30},{-87,-30}},
                                                  color={255,0,255}));
  connect(valveDiscreteRamp1.open, greaterThreshold1.y) annotation (Line(points
        ={{96,208},{188,208},{188,258},{181,258}}, color={255,0,255}));
  connect(greaterThreshold1.u, fixedDelay5.y)
    annotation (Line(points={{158,258},{138,258},{138,273}}, color={0,0,127}));
  connect(pump_4.port_b, valveDiscreteRamp1.port_a)
    annotation (Line(points={{88,230},{88,218}}, color={0,127,255}));
  connect(valveDiscreteRamp1.port_b, pipe_5.port_a)
    annotation (Line(points={{88,198},{88,186},{60,186}}, color={0,127,255}));
  connect(switch2.u2, greaterThreshold1.y) annotation (Line(points={{142,230},{
          188,230},{188,258},{181,258}}, color={255,0,255}));
  connect(switch2.u1, fixedDelay5.y) annotation (Line(points={{142,238},{142,
          258},{138,258},{138,273}}, color={0,0,127}));
  connect(realExpression1.y, switch2.u3) annotation (Line(points={{153,216},{142,
          216},{142,222}},                         color={0,0,127}));
  connect(switch2.y, pump_4.N_in)
    annotation (Line(points={{119,230},{98,230},{98,240}}, color={0,0,127}));
  connect(volumeFlow_2.port_b, massFlowRate.port_a)
    annotation (Line(points={{-86,50},{-92,50}},   color={0,127,255}));
  connect(massFlowRate.port_b, sink_2.port_a)
    annotation (Line(points={{-112,50},{-146,50}}, color={0,127,255}));
  connect(massFlowRate.m_flow, m_flow_2)
    annotation (Line(points={{-102,61},{-102,96},{-188,96}}, color={0,0,127}));
  connect(massFlowRate1.port_a, volumeFlow_3.port_b)
    annotation (Line(points={{-94,208},{-86,208}},   color={0,127,255}));
  connect(massFlowRate1.m_flow, m_flow_3) annotation (Line(points={{-104,219},{-104,
          258},{-186,258}},            color={0,0,127}));
  connect(volumeFlow_5.port_b, massFlowRate2.port_a)
    annotation (Line(points={{140,138},{154,138}}, color={0,127,255}));
  connect(massFlowRate2.port_b, sink_5.port_a) annotation (Line(points={{174,138},
          {218,138}},                     color={0,127,255}));
  connect(massFlowRate2.m_flow, m_flow5) annotation (Line(points={{164,149},{164,
          178},{324,178}},               color={0,0,127}));
  connect(volumeFlow_6.port_b, massFlowRate3.port_a)
    annotation (Line(points={{142,-38},{158,-38}}, color={0,127,255}));
  connect(massFlowRate3.port_b, sink_6.port_a)
    annotation (Line(points={{178,-38},{220,-38}}, color={0,127,255}));
  connect(m_flow_2, m_flow_2)
    annotation (Line(points={{-188,96},{-188,96}}, color={0,0,127}));
  connect(to_m3s_2.y, sink_2.Qf)
    annotation (Line(points={{-156,31},{-156,39.2}}, color={0,0,127}));
  connect(to_m3s_6.y, sink_6.Qf) annotation (Line(points={{253,-58},{230,-58},{
          230,-48.8}},
                   color={0,0,127}));
  connect(to_m3s.y, sink_3.Qf) annotation (Line(points={{-164,189},{-164,197.2}},
                                         color={0,0,127}));
  connect(to_m3s_5.y, sink_5.Qf) annotation (Line(points={{255,122},{228,122},{
          228,127.2}},
                   color={0,0,127}));
  connect(p_abs_2, p_abs_2)
    annotation (Line(points={{-186,76},{-186,76}}, color={0,0,127}));
  connect(massFlowRate1.port_b, sink_3.port_a)
    annotation (Line(points={{-114,208},{-154,208}}, color={0,127,255}));
  connect(pressure_3.port, sink_3.port_a) annotation (Line(points={{-134,226},{
          -134,208},{-154,208}}, color={0,127,255}));
  connect(pressure_5.port, massFlowRate2.port_b) annotation (Line(points={{196,
          152},{196,138},{174,138}}, color={0,127,255}));
  connect(pressure_2.port, sink_2.port_a) annotation (Line(points={{-128,66},{
          -128,50},{-146,50}}, color={0,127,255}));
  connect(massFlowRate3.port_b, pressure_6.port) annotation (Line(points={{178,
          -38},{200,-38},{200,-76}}, color={0,127,255}));
  connect(pressure_3.p, p_abs_3)
    annotation (Line(points={{-145,236},{-186,236}}, color={0,0,127}));
  connect(pressure_5.p, p_abs_5)
    annotation (Line(points={{207,162},{324,162}}, color={0,0,127}));
  connect(massFlowRate3.m_flow, m_flow6)
    annotation (Line(points={{168,-27},{168,-18},{324,-18}}, color={0,0,127}));
  connect(pressure_2.p, p_abs_2)
    annotation (Line(points={{-139,76},{-186,76}}, color={0,0,127}));
  connect(pressure_6.p, p_abs_6)
    annotation (Line(points={{211,-86},{324,-86}}, color={0,0,127}));
  connect(to_m3s_2.u, w_v_2)
    annotation (Line(points={{-156,8},{-156,-4},{-190,-4}}, color={0,0,127}));
  connect(to_m3s.u, w_v_3) annotation (Line(points={{-164,166},{-164,152},{-192,
          152}}, color={0,0,127}));
  connect(to_m3s_5.u, w_v_4)
    annotation (Line(points={{278,122},{326,122}}, color={0,0,127}));
  connect(to_m3s_6.u, w_v_6)
    annotation (Line(points={{276,-58},{320,-58}}, color={0,0,127}));
  connect(to_m3hr1.y, V_flow_6)
    annotation (Line(points={{132,-103},{132,-144}}, color={0,0,127}));
  connect(to_m3hr1.u, volumeFlow_6.V_flow)
    annotation (Line(points={{132,-80},{132,-49}}, color={0,0,127}));
 annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=180,
        origin={48,188})),
    Icon(coordinateSystem(preserveAspectRatio=false, extent={{-180,-140},{320,320}})),
    Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-180,-140},{320,
            320}})),
    uses(Modelica(version="4.0.0"), Custom_Pump_V2(version="1"),
      PressureDrivenDemand2(version="1"),
      PressureDrivenDemand_smooth(version="1")),
    version="1");
end mini_circular_water_network;

within ;
model mini_circular_water_network_wo_PI
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

  Modelica.Fluid.Sources.FixedBoundary sink_2(
    redeclare package Medium = Medium,
    p=system.p_ambient,
    T=system.T_ambient,
    nPorts=1)
    annotation (Placement(transformation(extent={{-174,46},{-154,66}})));

  Modelica.Fluid.Sources.FixedBoundary sink_3(
    redeclare package Medium = Medium,
    p=system.p_ambient,
    T=system.T_ambient,
    nPorts=1)
    annotation (Placement(transformation(extent={{-172,166},{-152,186}})));

  Modelica.Fluid.Sources.FixedBoundary source_4(
    nPorts=1,
    redeclare package Medium = Medium,
    use_T=true,
    T=system.T_ambient,
    p=system.p_ambient)
    annotation (Placement(transformation(extent={{46,294},{66,314}})));

   Modelica.Fluid.Sources.FixedBoundary sink_5(
    redeclare package Medium = Medium,
    p=system.p_ambient,
    T=system.T_ambient,
    nPorts=1)
    annotation (Placement(transformation(extent={{230,156},{210,176}})));

  Modelica.Fluid.Sources.FixedBoundary sink_6(
    redeclare package Medium = Medium,
    p=system.p_ambient,
    T=system.T_ambient,
    nPorts=1)
    annotation (Placement(transformation(extent={{248,-56},{228,-36}})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_1(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={-6,0})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_2(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=270,
        origin={-8,56})));

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

  Modelica.Fluid.Fittings.TeeJunctionIdeal teeJunctionIdeal_5(redeclare package
      Medium = Medium)
    annotation (Placement(transformation(extent={{90,148},{110,128}})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_6(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=180,
        origin={100,0})));

  Modelica.Fluid.Valves.ValveLinear valve_2(
    allowFlowReversal=false,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-114,66},{-134,46}})));

  Modelica.Fluid.Valves.ValveLinear valve_3(
    allowFlowReversal=false,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-112,186},{-132,166}})));

  Modelica.Fluid.Valves.ValveLinear valve_5(
    allowFlowReversal=false,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{178,156},{198,136}})));

  Modelica.Fluid.Valves.ValveLinear valve_6(
    allowFlowReversal=false,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{176,-28},{196,-48}})));

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
    annotation (Placement(transformation(extent={{-38,46},{-58,66}})));

  Modelica.Fluid.Pipes.StaticPipe pipe_3(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-8,94})));

  Modelica.Fluid.Pipes.StaticPipe pipe_4(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-36,166},{-56,186}})));

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

  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_1(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-92,-124},{-72,-104}})));

  Modelica.Fluid.Sensors.RelativePressure pressure_1(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-30,-72},{-50,-92}})));

  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_2(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-80,46},{-100,66}})));
  Modelica.Fluid.Sensors.RelativePressure pressure_2(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-114,94},{-134,74}})));

  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_3(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-78,166},{-98,186}})));

  Modelica.Fluid.Sensors.RelativePressure pressure_3(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-112,218},{-132,198}})));

  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_4(redeclare package Medium =
        Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=-90,
        origin={88,278})));

  Modelica.Fluid.Sensors.RelativePressure pressure_4(redeclare package Medium =
        Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=90,
        origin={42,244})));

  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_5(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{134,148},{154,128}})));
  Modelica.Fluid.Sensors.RelativePressure pressure_5(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{176,180},{196,160}})));

  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_6(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{122,-28},{142,-48}})));

  Modelica.Fluid.Sensors.RelativePressure pressure_6(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{176,-2},{196,-22}})));

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
    annotation (Placement(transformation(extent={{-190,110},{-210,130}})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_2 "Relative pressure signal"
    annotation (Placement(transformation(extent={{-188,88},{-208,108}})));
  Modelica.Blocks.Interfaces.RealInput w_v_2
    "Connector of setpoint input signal"
    annotation (Placement(transformation(extent={{-210,12},{-170,52}})));

  Modelica.Blocks.Interfaces.RealOutput V_flow_3
    "Connector of Real output signal containing input signal u in another unit"
    annotation (Placement(transformation(extent={{-178,244},{-198,264}})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_3 "Relative pressure signal"
    annotation (Placement(transformation(extent={{-180,220},{-200,240}})));
  Modelica.Blocks.Interfaces.RealInput w_v_3
    "Connector of setpoint input signal"    annotation (Placement(transformation(extent={{-214,
            136},{-174,176}})));

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
        origin={322,102})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_5 "Relative pressure signal"
    annotation (Placement(transformation(extent={{312,180},{332,200}})));
  Modelica.Blocks.Interfaces.RealInput w_v_5
    "=1: completely open, =0: completely closed"
    annotation (Placement(transformation(extent={{-20,-20},{20,20}},
        rotation=180,
        origin={324,134})));

  Modelica.Blocks.Interfaces.RealOutput V_flow_6
    "Volume flow rate from port_a to port_b"
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={132,-144})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_6 "Relative pressure signal"
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={260,-146})));
  Modelica.Blocks.Interfaces.RealInput w_v_6
    "=1: completely open, =0: completely closed"
    annotation (Placement(transformation(extent={{-20,-20},{20,20}},
        rotation=90,
        origin={186,-148})));

  Modelica.Blocks.Math.UnitConversions.To_bar to_bar annotation (Placement(transformation(extent={{-64,-74},
            {-84,-54}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar2 annotation (Placement(transformation(extent={{-148,88},
            {-168,108}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar3 annotation (Placement(transformation(extent={{-136,
            220},{-156,240}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar4 annotation (Placement(
        transformation(
        extent={{10,-10},{-10,10}},
        rotation=-90,
        origin={-28,266})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar5 annotation (Placement(transformation(extent={{236,180},
            {256,200}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar6 annotation (Placement(transformation(extent={{216,-26},{236,-6}})));

  To_m3hr to_m3hr  annotation (Placement(transformation(extent={{-90,-94},{-110,
            -74}})));
  To_m3hr to_m3hr2 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-90,100})));
  To_m3hr to_m3hr3 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-88,224})));
  To_m3hr to_m3hr4
    annotation (Placement(transformation(extent={{34,268},{14,288}})));
  To_m3hr to_m3hr5 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={198,96})));
  To_m3hr to_m3hr6 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={132,-76})));

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
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay5(delayTime=1)   annotation (
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
  Modelica.Fluid.Valves.ValveDiscrete     valveDiscreteRamp(
    redeclare package Medium = Medium,
    allowFlowReversal=true,
    dp_nominal=1,
    m_flow_nominal=100)
    annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=90,
        origin={16,-92})));
  Modelica.Blocks.Logical.GreaterThreshold greaterThreshold1(threshold=0.01)
    annotation (Placement(transformation(extent={{160,248},{180,268}})));
  Modelica.Blocks.Logical.Switch switch2
    annotation (Placement(transformation(extent={{140,220},{120,240}})));
  Modelica.Fluid.Valves.ValveDiscrete     valveDiscreteRamp1(
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
equation
  P_pum_1 = pump_1.W_total;
  P_pum_4 = pump_4.W_total;
  connect(volumeFlow_1.port_a, source_1.ports[1])
    annotation (Line(points={{-92,-114},{-154,-114}},
                                                   color={0,127,255}));
  connect(pressure_6.port_b,valve_6. port_b) annotation (Line(
      points={{196,-12},{200,-12},{200,-38},{196,-38}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pressure_6.port_a,valve_6. port_a) annotation (Line(
      points={{176,-12},{172,-12},{172,-38},{176,-38}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pipe_1.port_b,idealJunction_1. port_2)
    annotation (Line(points={{-8,-24},{-8,-10},{-6,-10}},
                                                 color={0,127,255}));
  connect(p_rel_4, p_rel_4)
    annotation (Line(points={{-28,324},{-28,324}},   color={0,0,127}));
  connect(volumeFlow_6.port_b,valve_6. port_a)
    annotation (Line(points={{142,-38},{176,-38}},
                                             color={0,127,255}));
  connect(valve_6.port_b,sink_6. ports[1])
    annotation (Line(points={{196,-38},{222,-38},{222,-46},{228,-46}},
                                               color={0,127,255}));
  connect(p_rel_6,p_rel_6)
    annotation (Line(points={{260,-146},{260,-146}},
                                                 color={0,0,127}));
  connect(pressure_1.port_a, pipe_1.port_a) annotation (Line(points={{-30,-82},
          {-8,-82},{-8,-44}},                    color={0,127,255}));
  connect(to_bar.y, p_rel_1)
    annotation (Line(points={{-85,-64},{-190,-64}}, color={0,0,127}));
  connect(to_bar.u, pressure_1.p_rel)
    annotation (Line(points={{-62,-64},{-40,-64},{-40,-73}}, color={0,0,127}));
  connect(to_bar6.u,pressure_6. p_rel)
    annotation (Line(points={{214,-16},{206,-16},{206,8},{186,8},{186,-3}},
                                                        color={0,0,127}));
  connect(to_bar6.y,p_rel_6)
    annotation (Line(points={{237,-16},{260,-16},{260,-146}},
                                                 color={0,0,127}));
  connect(pressure_4.p_rel, to_bar4.u)
    annotation (Line(points={{33,244},{-28,244},{-28,254}},
                                                        color={0,0,127}));
  connect(to_bar4.y, p_rel_4) annotation (Line(points={{-28,277},{-28,324}},
                                color={0,0,127}));
  connect(to_m3hr.u, volumeFlow_1.V_flow)
    annotation (Line(points={{-88,-84},{-88,-92},{-82,-92},{-82,-103}},
                                                             color={0,0,127}));
  connect(volumeFlow_4.V_flow, to_m3hr4.u)
    annotation (Line(points={{77,278},{36,278}}, color={0,0,127}));
  connect(V_flow_6,to_m3hr6. y) annotation (Line(points={{132,-144},{132,-87}},
                          color={0,0,127}));
  connect(volumeFlow_6.V_flow,to_m3hr6. u)
    annotation (Line(points={{132,-49},{132,-52},{130,-52},{130,-56},{134,-56},
          {132,-64}},                            color={0,0,127}));
  connect(pipe_4.port_b, volumeFlow_3.port_a)
    annotation (Line(points={{-56,176},{-78,176}},  color={0,127,255}));
  connect(volumeFlow_3.port_b, valve_3.port_a)
    annotation (Line(points={{-98,176},{-112,176}},color={0,127,255}));
  connect(pressure_3.port_a, valve_3.port_a) annotation (Line(points={{-112,208},
          {-106,208},{-106,176},{-112,176}},
                                          color={0,127,255}));
  connect(pressure_3.p_rel, to_bar3.u)
    annotation (Line(points={{-122,217},{-122,230},{-134,230}},
                                                             color={0,0,127}));
  connect(to_bar3.y, p_rel_3)
    annotation (Line(points={{-157,230},{-190,230}},
                                                   color={0,0,127}));
  connect(volumeFlow_3.V_flow, to_m3hr3.u)
    annotation (Line(points={{-88,187},{-88,212}},
                                                 color={0,0,127}));
  connect(to_m3hr3.y, V_flow_3)
    annotation (Line(points={{-88,235},{-88,254},{-188,254}},color={0,0,127}));

  connect(valve_3.port_b, pressure_3.port_b) annotation (Line(points={{-132,176},
          {-138,176},{-138,208},{-132,208}},
                                          color={0,127,255}));
  connect(sink_3.ports[1], valve_3.port_b)
    annotation (Line(points={{-152,176},{-132,176}},
                                                   color={0,127,255}));
  connect(to_m3hr.y, V_flow_1)
    annotation (Line(points={{-111,-84},{-190,-84}}, color={0,0,127}));
  connect(to_m3hr4.y, V_flow_4) annotation (Line(points={{13,278},{2,278},{2,
          322}},            color={0,0,127}));
  connect(pressure_5.port_b,valve_5. port_b) annotation (Line(
      points={{196,170},{198,170},{198,146}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pressure_5.port_a,valve_5. port_a) annotation (Line(
      points={{176,170},{176,146},{178,146}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(volumeFlow_5.port_b,valve_5. port_a)
    annotation (Line(points={{154,138},{178,138},{178,146}},
                                             color={0,127,255}));
  connect(valve_5.port_b,sink_5. ports[1])
    annotation (Line(points={{198,146},{204,146},{204,166},{210,166}},
                                               color={0,127,255}));
  connect(p_rel_5,p_rel_5)
    annotation (Line(points={{322,190},{322,190}},
                                                 color={0,0,127}));
  connect(to_bar5.u,pressure_5. p_rel)
    annotation (Line(points={{234,190},{186,190},{186,179}},
                                                        color={0,0,127}));
  connect(to_bar5.y,p_rel_5)
    annotation (Line(points={{257,190},{322,190}},
                                                 color={0,0,127}));
  connect(V_flow_5, to_m3hr5.y) annotation (Line(points={{322,102},{268,102},{268,
          96},{209,96}},   color={0,0,127}));
  connect(volumeFlow_5.V_flow, to_m3hr5.u)
    annotation (Line(points={{144,127},{144,96},{186,96}},   color={0,0,127}));
  connect(pipe_2.port_b, volumeFlow_2.port_a)
    annotation (Line(points={{-58,56},{-80,56}}, color={0,127,255}));
  connect(volumeFlow_2.port_b, valve_2.port_a)
    annotation (Line(points={{-100,56},{-114,56}}, color={0,127,255}));
  connect(pressure_2.port_a,valve_2. port_a) annotation (Line(points={{-114,84},
          {-108,84},{-108,56},{-114,56}}, color={0,127,255}));
  connect(pressure_2.p_rel, to_bar2.u)
    annotation (Line(points={{-124,93},{-124,98},{-146,98}}, color={0,0,127}));
  connect(to_bar2.y, p_rel_2)
    annotation (Line(points={{-169,98},{-198,98}}, color={0,0,127}));
  connect(volumeFlow_2.V_flow, to_m3hr2.u)
    annotation (Line(points={{-90,67},{-90,88}}, color={0,0,127}));
  connect(to_m3hr2.y, V_flow_2)
    annotation (Line(points={{-90,111},{-90,120},{-200,120}},color={0,0,127}));
  connect(valve_2.port_b,pressure_2. port_b) annotation (Line(points={{-134,56},
          {-140,56},{-140,84},{-134,84}}, color={0,127,255}));
  connect(sink_2.ports[1], valve_2.port_b)
    annotation (Line(points={{-154,56},{-134,56}}, color={0,127,255}));
  connect(idealJunction_2.port_2, idealJunction_1.port_1)
    annotation (Line(points={{-8,46},{-8,16},{-6,16},{-6,10}},
                                                       color={0,127,255}));
  connect(pipe_3.port_a, idealJunction_2.port_1)
    annotation (Line(points={{-8,84},{-8,66}}, color={0,127,255}));
  connect(pipe_8.port_b,idealJunction_6. port_2)
    annotation (Line(points={{60,0},{90,0}}, color={0,127,255}));
  connect(idealJunction_6.port_1,volumeFlow_6. port_a)
    annotation (Line(points={{110,0},{118,0},{118,-38},{122,-38}},
                                               color={0,127,255}));
  connect(idealJunction_3.port_1, pipe_3.port_b)
    annotation (Line(points={{-8,128},{-8,104}},color={0,127,255}));
  connect(idealJunction_3.port_2, idealJunction_4.port_2)
    annotation (Line(points={{-8,148},{-8,156}}, color={0,127,255}));

  connect(idealJunction_1.port_3,pipe_8. port_a)
    annotation (Line(points={{4,0},{40,0}}, color={0,127,255}));
  connect(pipe_2.port_a, idealJunction_2.port_3)
    annotation (Line(points={{-38,56},{-18,56}}, color={0,127,255}));
  connect(idealJunction_3.port_3, pipe_6.port_a)
    annotation (Line(points={{2,138},{40,138}}, color={0,127,255}));
  connect(pipe_7.port_b,idealJunction_6. port_3)
    annotation (Line(points={{101,68},{100,64},{100,10}},
                                                 color={0,127,255}));
  connect(pipe_4.port_a, idealJunction_4.port_3)
    annotation (Line(points={{-36,176},{-26,176},{-26,166},{-18,166}},
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
  connect(teeJunctionIdeal_5.port_1, pipe_6.port_b)
    annotation (Line(points={{90,138},{60,138}}, color={0,127,255}));
  connect(teeJunctionIdeal_5.port_2, volumeFlow_5.port_a) annotation (Line(
        points={{110,138},{134,138}},                     color={0,127,255}));
  connect(teeJunctionIdeal_5.port_3, pipe_7.port_a) annotation (Line(points={{100,
          128},{101,124},{101,86}}, color={0,127,255}));
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
  connect(switch1.y, pump_1.N_in) annotation (Line(points={{-41,-2},{-32,-2},{-32,
          -30},{-30,-30},{-30,-60},{-16,-60},{-16,-104},{-26,-104}}, color={0,0,
          127}));
  connect(pump_1.port_b, valveDiscreteRamp.port_a) annotation (Line(points={{-16,
          -114},{16,-114},{16,-102}}, color={0,127,255}));
  connect(valveDiscreteRamp.port_b, pipe_1.port_a) annotation (Line(points={{16,
          -82},{16,-50},{-8,-50},{-8,-44}}, color={0,127,255}));
  connect(valveDiscreteRamp.open, greaterThreshold.y) annotation (Line(points={{
          24,-92},{-28,-92},{-28,-30},{-87,-30}}, color={255,0,255}));
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
  connect(realExpression1.y, switch2.u3) annotation (Line(points={{153,216},{
          146,216},{146,214},{142,214},{142,222}}, color={0,0,127}));
  connect(switch2.y, pump_4.N_in)
    annotation (Line(points={{119,230},{98,230},{98,240}}, color={0,0,127}));
  connect(w_v_3, valve_3.opening) annotation (Line(points={{-194,156},{-122,156},
          {-122,168}}, color={0,0,127}));
  connect(w_v_5, valve_5.opening)
    annotation (Line(points={{324,134},{188,134},{188,138}}, color={0,0,127}));
  connect(w_v_6, valve_6.opening)
    annotation (Line(points={{186,-148},{186,-46}}, color={0,0,127}));
  connect(w_v_2, valve_2.opening)
    annotation (Line(points={{-190,32},{-124,32},{-124,48}}, color={0,0,127}));
  connect(V_flow_3, V_flow_3)
    annotation (Line(points={{-188,254},{-188,254}}, color={0,0,127}));
 annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=180,
        origin={48,188})),
    Icon(coordinateSystem(preserveAspectRatio=false, extent={{-180,-140},{320,320}})),
    Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-180,-140},{320,
            320}})),
    uses(Modelica(version="4.0.0"), Custom_Pump_V2(version="1")),
    version="1");
end mini_circular_water_network_wo_PI;

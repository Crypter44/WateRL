within ;
model simple_network "System consisting of a pump and a valve."

  //replaceable package Medium = Modelica.Media.Water.StandardWaterOnePhase
    //constrainedby Modelica.Media.Interfaces.PartialMedium;
  replaceable package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater
    constrainedby Modelica.Media.Interfaces.PartialMedium;

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
    annotation (Placement(transformation(extent={{-174,28},{-154,48}})));

  Modelica.Fluid.Sources.FixedBoundary source_4(
    redeclare package Medium = Medium,
    use_T=true,
    T=system.T_ambient,
    p=system.p_ambient,
    nPorts=1)
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={-8,112})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_2(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=270,
        origin={-8,38})));

  Modelica.Fluid.Valves.ValveLinear valve_2(
    allowFlowReversal=false,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-96,28},{-116,48}})));

  Modelica.Fluid.Pipes.StaticPipe pipe_1(
    allowFlowReversal=true,
    length=20,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium,
    height_ab=0)
    annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-8,-34})));

  Modelica.Fluid.Pipes.StaticPipe pipe_2(
    allowFlowReversal=true,
    length=20,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-38,28},{-58,48}})));

  Modelica.Fluid.Pipes.StaticPipe pipe_3(
    allowFlowReversal=true,
    length=20,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-8,76})));

  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_1(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-92,-124},{-72,-104}})));

  Modelica.Fluid.Sensors.RelativePressure pressure_1(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-30,-72},{-50,-92}})));

  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_2(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-64,28},{-84,48}})));
  Modelica.Fluid.Sensors.RelativePressure pressure_2(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-114,76},{-134,56}})));

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
    annotation (Placement(transformation(extent={{-178,90},{-198,110}})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_2 "Relative pressure signal"
    annotation (Placement(transformation(extent={{-178,70},{-198,90}})));
  Modelica.Blocks.Interfaces.RealInput w_v_2
    "Connector of setpoint input signal"
    annotation (Placement(transformation(extent={{-208,104},{-168,144}})));

  Modelica.Blocks.Math.UnitConversions.To_bar to_bar annotation (Placement(transformation(extent={{-64,-74},
            {-84,-54}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar2 annotation (Placement(transformation(extent={{-148,70},{-168,90}})));

  To_m3hr to_m3hr  annotation (Placement(transformation(extent={{-90,-94},{-110,
            -74}})));
  To_m3hr to_m3hr2 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-74,82})));

  Modelica.Blocks.Interfaces.RealOutput P_pum_1
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={-26,-144})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay(delayTime=0.1)
    annotation (Placement(transformation(extent={{-150,-52},{-130,-32}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay3(delayTime=0.1)
    annotation (Placement(transformation(extent={{-150,114},{-130,134}})));
  Modelica.Blocks.Logical.GreaterThreshold greaterThreshold(threshold=0.01)
    annotation (Placement(transformation(extent={{-108,-40},{-88,-20}})));
  Modelica.Blocks.Logical.Switch switch1
    annotation (Placement(transformation(extent={{-42,-10},{-22,10}})));
  Modelica.Blocks.Sources.RealExpression realExpression(y=0)
    annotation (Placement(transformation(extent={{-72,-26},{-52,-6}})));
  Modelica.Fluid.Valves.ValveDiscrete     valveDiscreteRamp(
    redeclare package Medium = Medium,
    allowFlowReversal=true,
    dp_nominal=1,
    m_flow_nominal=100)
    annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=90,
        origin={16,-92})));
equation
  P_pum_1 = pump_1.W_total;
  connect(volumeFlow_1.port_a, source_1.ports[1])
    annotation (Line(points={{-92,-114},{-154,-114}},
                                                   color={0,127,255}));
  connect(pressure_1.port_a, pipe_1.port_a) annotation (Line(points={{-30,-82},
          {-8,-82},{-8,-44}},                    color={0,127,255}));
  connect(to_bar.y, p_rel_1)
    annotation (Line(points={{-85,-64},{-190,-64}}, color={0,0,127}));
  connect(to_bar.u, pressure_1.p_rel)
    annotation (Line(points={{-62,-64},{-40,-64},{-40,-73}}, color={0,0,127}));
  connect(to_m3hr.u, volumeFlow_1.V_flow)
    annotation (Line(points={{-88,-84},{-88,-92},{-82,-92},{-82,-103}},
                                                             color={0,0,127}));

  connect(to_m3hr.y, V_flow_1)
    annotation (Line(points={{-111,-84},{-190,-84}}, color={0,0,127}));
  connect(pipe_2.port_b, volumeFlow_2.port_a)
    annotation (Line(points={{-58,38},{-64,38}}, color={0,127,255}));
  connect(volumeFlow_2.port_b, valve_2.port_a)
    annotation (Line(points={{-84,38},{-96,38}},   color={0,127,255}));
  connect(pressure_2.port_a,valve_2. port_a) annotation (Line(points={{-114,66},
          {-90,66},{-90,38},{-96,38}},    color={0,127,255}));
  connect(pressure_2.p_rel, to_bar2.u)
    annotation (Line(points={{-124,75},{-124,80},{-146,80}}, color={0,0,127}));
  connect(to_bar2.y, p_rel_2)
    annotation (Line(points={{-169,80},{-188,80}}, color={0,0,127}));
  connect(volumeFlow_2.V_flow, to_m3hr2.u)
    annotation (Line(points={{-74,49},{-74,70}}, color={0,0,127}));
  connect(to_m3hr2.y, V_flow_2)
    annotation (Line(points={{-74,93},{-74,100},{-188,100}}, color={0,0,127}));
  connect(valve_2.port_b,pressure_2. port_b) annotation (Line(points={{-116,38},
          {-140,38},{-140,66},{-134,66}}, color={0,127,255}));
  connect(sink_2.ports[1], valve_2.port_b)
    annotation (Line(points={{-154,38},{-116,38}}, color={0,127,255}));
  connect(pipe_3.port_a, idealJunction_2.port_1)
    annotation (Line(points={{-8,66},{-8,48}}, color={0,127,255}));

  connect(pipe_2.port_a, idealJunction_2.port_3)
    annotation (Line(points={{-38,38},{-18,38}}, color={0,127,255}));
  connect(volumeFlow_1.port_b, pump_1.port_a)
    annotation (Line(points={{-72,-114},{-36,-114}},
                                                   color={0,127,255}));
  connect(pressure_1.port_b, pump_1.port_a) annotation (Line(points={{-50,-82},
          {-60,-82},{-60,-114},{-36,-114}},
                                         color={0,127,255}));
  connect(fixedDelay.u, w_p_1)
    annotation (Line(points={{-152,-42},{-188,-42}}, color={0,0,127}));
  connect(w_v_2, fixedDelay3.u) annotation (Line(points={{-188,124},{-152,124}},
                                  color={0,0,127}));
  connect(switch1.u1, fixedDelay.y) annotation (Line(points={{-44,8},{-122,8},{-122,
          -42},{-129,-42}}, color={0,0,127}));
  connect(greaterThreshold.y, switch1.u2) annotation (Line(points={{-87,-30},{-84,
          -30},{-84,0},{-44,0}},   color={255,0,255}));
  connect(greaterThreshold.u, fixedDelay.y) annotation (Line(points={{-110,-30},
          {-120,-30},{-120,-42},{-129,-42}}, color={0,0,127}));
  connect(realExpression.y, switch1.u3) annotation (Line(points={{-51,-16},{-48,
          -16},{-48,-8},{-44,-8}},                       color={0,0,127}));
  connect(switch1.y, pump_1.N_in) annotation (Line(points={{-21,0},{-18,0},{-18,
          -20},{-24,-20},{-24,-96},{-26,-96},{-26,-104}},            color={0,0,
          127}));
  connect(pump_1.port_b, valveDiscreteRamp.port_a) annotation (Line(points={{-16,
          -114},{16,-114},{16,-102}}, color={0,127,255}));
  connect(valveDiscreteRamp.port_b, pipe_1.port_a) annotation (Line(points={{16,
          -82},{16,-50},{-8,-50},{-8,-44}}, color={0,127,255}));
  connect(valveDiscreteRamp.open, greaterThreshold.y) annotation (Line(points={{24,-92},
          {32,-92},{32,-30},{-87,-30}},           color={255,0,255}));
  connect(idealJunction_2.port_2, pipe_1.port_b)
    annotation (Line(points={{-8,28},{-8,-24}}, color={0,127,255}));
  connect(source_4.ports[1], pipe_3.port_b)
    annotation (Line(points={{-8,102},{-8,86}}, color={0,127,255}));
  connect(fixedDelay3.y, valve_2.opening) annotation (Line(points={{-129,124},{-106,
          124},{-106,46}}, color={0,0,127}));
 annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=180,
        origin={48,188})),
    Icon(coordinateSystem(preserveAspectRatio=false, extent={{-180,-140},{40,160}})),
    Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-180,-140},{40,
            160}})),
    uses(Modelica(version="4.0.0"), Custom_Pump_V2(version="1")),
    version="1");
end simple_network;

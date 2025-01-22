within ;
model simple_network_valve
  "System consisting of a pump and a valve, the valve is controlled."

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
    annotation (Placement(transformation(extent={{-44,-68},{-24,-88}})));


    //energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial, //only needed for Medium = Modelica.Media.Water.StandardWaterOnePhase
    //massDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial, //only needed for Medium = Modelica.Media.Water.StandardWaterOnePhase

  Modelica.Fluid.Sources.FixedBoundary source_1(
    nPorts=1,
    redeclare package Medium = Medium,
    use_T=true,
    T=system.T_ambient,
    p=system.p_ambient)
    annotation (Placement(transformation(extent={{-174,-88},{-154,-68}})));

  Modelica.Fluid.Sources.FixedBoundary sink_2(
    redeclare package Medium = Medium,
    p=system.p_ambient,
    T=system.T_ambient,
    nPorts=1)
    annotation (Placement(transformation(extent={{-174,28},{-154,48}})));

  Modelica.Fluid.Sources.FixedBoundary source_3(
    redeclare package Medium = Medium,
    use_T=true,
    T=system.T_ambient,
    p=system.p_ambient,
    nPorts=1)
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={-8,102})));

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
    annotation (Placement(transformation(extent={{-104,48},{-124,28}})));

  Modelica.Fluid.Pipes.StaticPipe pipe_1(
    allowFlowReversal=true,
    length=20,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium,
    height_ab=0)
    annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-8,-10})));

  Modelica.Fluid.Pipes.StaticPipe pipe_2(
    allowFlowReversal=true,
    length=20,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-32,28},{-52,48}})));

  Modelica.Fluid.Pipes.StaticPipe pipe_3(
    allowFlowReversal=true,
    length=20,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-8,70})));

  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_1(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-92,-88},{-72,-68}})));

  Modelica.Fluid.Sensors.RelativePressure pressure_1(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-30,-34},{-50,-54}})));

  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_2(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-64,28},{-84,48}})));
  Modelica.Fluid.Sensors.RelativePressure pressure_2(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-114,76},{-134,56}})));

 Modelica.Blocks.Interfaces.RealOutput V_flow_1
    "Connector of Real output signal containing input signal u in another unit"
    annotation (Placement(transformation(extent={{-180,-36},{-200,-56}})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_1 "Relative pressure signal"
    annotation (Placement(transformation(extent={{-180,-36},{-200,-16}})));

  Modelica.Blocks.Interfaces.RealOutput V_flow_2
    "Connector of Real output signal containing input signal u in another unit"
    annotation (Placement(transformation(extent={{-178,90},{-198,110}})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_2 "Relative pressure signal"
    annotation (Placement(transformation(extent={{-178,70},{-198,90}})));
  Modelica.Blocks.Interfaces.RealInput w_v_2
    "Connector of setpoint input signal"
    annotation (Placement(transformation(extent={{-208,-16},{-168,24}})));

  Modelica.Blocks.Math.UnitConversions.To_bar to_bar annotation (Placement(transformation(extent={{-64,-36},
            {-84,-16}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar2 annotation (Placement(transformation(extent={{-148,70},{-168,90}})));

  To_m3hr to_m3hr  annotation (Placement(transformation(extent={{-90,-56},{-110,
            -36}})));
  To_m3hr to_m3hr2 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-74,82})));

  Modelica.Blocks.Interfaces.RealOutput P_pum_1
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={-34,-108})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay3(delayTime=0.1)
    annotation (Placement(transformation(extent={{-146,-6},{-126,14}})));
  Modelica.Blocks.Sources.RealExpression pump_rpm(y=0.75)
    annotation (Placement(transformation(extent={{14,-104},{-6,-82}})));
equation
  P_pum_1 = pump_1.W_total;
  connect(volumeFlow_1.port_a, source_1.ports[1])
    annotation (Line(points={{-92,-78},{-154,-78}},color={0,127,255}));
  connect(pressure_1.port_a, pipe_1.port_a) annotation (Line(points={{-30,-44},
          {-8,-44},{-8,-20}},                    color={0,127,255}));
  connect(to_bar.y, p_rel_1)
    annotation (Line(points={{-85,-26},{-190,-26}}, color={0,0,127}));
  connect(to_bar.u, pressure_1.p_rel)
    annotation (Line(points={{-62,-26},{-40,-26},{-40,-35}}, color={0,0,127}));
  connect(to_m3hr.u, volumeFlow_1.V_flow)
    annotation (Line(points={{-88,-46},{-80,-46},{-80,-64},{-82,-64},{-82,-67}},
                                                             color={0,0,127}));

  connect(to_m3hr.y, V_flow_1)
    annotation (Line(points={{-111,-46},{-190,-46}}, color={0,0,127}));
  connect(pipe_2.port_b, volumeFlow_2.port_a)
    annotation (Line(points={{-52,38},{-64,38}}, color={0,127,255}));
  connect(volumeFlow_2.port_b, valve_2.port_a)
    annotation (Line(points={{-84,38},{-104,38}},  color={0,127,255}));
  connect(pressure_2.port_a,valve_2. port_a) annotation (Line(points={{-114,66},
          {-90,66},{-90,38},{-104,38}},   color={0,127,255}));
  connect(pressure_2.p_rel, to_bar2.u)
    annotation (Line(points={{-124,75},{-124,80},{-146,80}}, color={0,0,127}));
  connect(to_bar2.y, p_rel_2)
    annotation (Line(points={{-169,80},{-188,80}}, color={0,0,127}));
  connect(volumeFlow_2.V_flow, to_m3hr2.u)
    annotation (Line(points={{-74,49},{-74,70}}, color={0,0,127}));
  connect(to_m3hr2.y, V_flow_2)
    annotation (Line(points={{-74,93},{-74,100},{-188,100}}, color={0,0,127}));
  connect(valve_2.port_b,pressure_2. port_b) annotation (Line(points={{-124,38},
          {-140,38},{-140,66},{-134,66}}, color={0,127,255}));
  connect(sink_2.ports[1], valve_2.port_b)
    annotation (Line(points={{-154,38},{-124,38}}, color={0,127,255}));
  connect(pipe_3.port_a, idealJunction_2.port_1)
    annotation (Line(points={{-8,60},{-8,48}}, color={0,127,255}));

  connect(pipe_2.port_a, idealJunction_2.port_3)
    annotation (Line(points={{-32,38},{-18,38}}, color={0,127,255}));
  connect(volumeFlow_1.port_b, pump_1.port_a)
    annotation (Line(points={{-72,-78},{-44,-78}}, color={0,127,255}));
  connect(pressure_1.port_b, pump_1.port_a) annotation (Line(points={{-50,-44},
          {-58,-44},{-58,-78},{-44,-78}},color={0,127,255}));
  connect(w_v_2, fixedDelay3.u) annotation (Line(points={{-188,4},{-148,4}},
                                  color={0,0,127}));
  connect(idealJunction_2.port_2, pipe_1.port_b)
    annotation (Line(points={{-8,28},{-8,0}},   color={0,127,255}));
  connect(source_3.ports[1], pipe_3.port_b)
    annotation (Line(points={{-8,92},{-8,80}},  color={0,127,255}));
  connect(fixedDelay3.y, valve_2.opening) annotation (Line(points={{-125,4},{
          -114,4},{-114,30}},
                           color={0,0,127}));
  connect(pump_1.port_b, pipe_1.port_a)
    annotation (Line(points={{-24,-78},{-8,-78},{-8,-20}}, color={0,127,255}));
  connect(pump_rpm.y, pump_1.N_in) annotation (Line(points={{-7,-93},{-12,-93},
          {-12,-88},{-34,-88}}, color={0,0,127}));
 annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=180,
        origin={48,188})),
    Icon(coordinateSystem(preserveAspectRatio=false, extent={{-180,-100},{20,
            120}})),
    Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-180,-100},{20,
            120}})),
    uses(Modelica(version="4.0.0"), Custom_Pump_V2(version="1")),
    version="1");
end simple_network_valve;

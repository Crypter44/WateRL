within ;
model mini_tank
  "System consisting of two pumps, a tank, and multiple sinks (consumer)."

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
    annotation (Placement(transformation(extent={{-80,86},{-60,106}})));

  // Power Curve -2.58707135371286 7.63830983123264 29.4033336406988 57.46
  // beta_1 * Q^3 +  beta_2 * Q^2 * n + beta_3 * Q * n^2 + beta_4 * n^3  with the pumps electrical Power "P" in kW, the pumps volumeflow "Q" in m³/h

    //energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial, //only needed for Medium = Modelica.Media.Water.StandardWaterOnePhase
    //massDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial, //only needed for Medium = Modelica.Media.Water.StandardWaterOnePhase

    Custom_Pump_V2.BaseClasses_Custom.Pump_vs pump_4(
    redeclare package Medium = Medium,
    allowFlowReversal=true,
    m_flow_start=0,
    redeclare function flowCharacteristic =
        Custom_Pump_V2.BaseClasses_Custom.PumpCharacteristics.quadraticFlow (c=
            {-0.962830634026511, 0.33404915911756, 15.48238267}),
    redeclare function powerCharacteristic =
        Custom_Pump_V2.BaseClasses_Custom.PumpCharacteristics.cubicPower (
         c={-2.58707135371286, 7.63830983123264, 29.4033336406988, 57.46, 0}),
    checkValve=false,
    rpm_rel=0.93969,
    use_N_in=true,
    V=0.1,
    use_HeatTransfer=false,
    redeclare model HeatTransfer =
        Modelica.Fluid.Vessels.BaseClasses.HeatTransfer.IdealHeatTransfer)
   "Pump_ID = 1 from pressure_booster_pumps_for_buildings"
    annotation (Placement(transformation(extent={{10,10},{-10,-10}},
        rotation=90,
        origin={92,458})));

  Modelica.Fluid.Vessels.OpenTank tank_9(
    height=5,
    crossArea=5,
    redeclare package Medium = Medium,
    use_portsData=false,
    nPorts=1) annotation (Placement(transformation(extent={{-54,124},{-14,164}})));

  Modelica.Fluid.Sources.FixedBoundary source_4(
    nPorts=1,
    redeclare package Medium = Medium,
    use_T=true,
    T=system.T_ambient,
    p=system.p_ambient)
    annotation (Placement(transformation(extent={{50,514},{70,534}})));

   Modelica.Fluid.Sources.FixedBoundary sink_5(
    redeclare package Medium = Medium,
    p=system.p_ambient,
    T=system.T_ambient,
    nPorts=1)
    annotation (Placement(transformation(extent={{236,386},{216,406}})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal teeJunctionIdeal_5(redeclare package
      Medium = Medium)
    annotation (Placement(transformation(extent={{26,366},{46,346}})));

  Modelica.Fluid.Valves.ValveLinear valve_5(
    allowFlowReversal=true,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{182,374},{202,354}})));

  Modelica.Fluid.Valves.ValveLinear valve_7_1(
    allowFlowReversal=true,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={114,252})));

  Modelica.Fluid.Valves.ValveDiscrete valveDiscreteRamp_4(
    redeclare package Medium = Medium,
    allowFlowReversal=true,
    dp_nominal=1,
    m_flow_nominal=100) annotation (Placement(transformation(
        extent={{10,10},{-10,-10}},
        rotation=90,
        origin={92,426})));

  Modelica.Fluid.Pipes.StaticPipe pipe_5(
    allowFlowReversal=true,
    length=30,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium,
    height_ab=0)
    annotation (Placement(transformation(extent={{36,396},{16,416}})));

  Modelica.Fluid.Pipes.StaticPipe pipe_9(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.05,
    redeclare package Medium = Medium,
    height_ab=-7)
    annotation (Placement(transformation(extent={{38,66},{58,86}})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_4(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=-90,
        origin={92,496})));
  Modelica.Fluid.Sensors.RelativePressure pressure_4(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=90,
        origin={46,462})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_5(
    redeclare package Medium = Medium, allowFlowReversal=false)
    annotation (Placement(transformation(extent={{138,366},{158,346}})));
  Modelica.Fluid.Sensors.RelativePressure pressure_5(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{180,398},{200,378}})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_7_1(redeclare package Medium
      = Medium, allowFlowReversal=true)
                annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={114,176})));
  Modelica.Fluid.Sensors.RelativePressure pressure_7_1(redeclare package Medium
      = Medium) annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=90,
        origin={140,282})));
  Modelica.Blocks.Continuous.LimPID PI_5(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{262,342},{242,362}})));

  Modelica.Blocks.Interfaces.RealOutput V_flow_4
    "Volume flow rate from port_a to port_b" annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={4,546})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_4 "Relative pressure signal"
    annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=-90,
        origin={-26,548})));
  Modelica.Blocks.Interfaces.RealInput w_p_4
    "=1: completely open, =0: completely closed" annotation (Placement(
        transformation(
        extent={{20,-20},{-20,20}},
        rotation=90,
        origin={144,546})));

  Modelica.Blocks.Interfaces.RealOutput V_flow_5
    "Volume flow rate from port_a to port_b"
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=0,
        origin={330,320})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_5 "Relative pressure signal"
    annotation (Placement(transformation(extent={{318,406},{338,426}})));
  Modelica.Blocks.Interfaces.RealInput w_v_5
    "=1: completely open, =0: completely closed"
    annotation (Placement(transformation(extent={{-20,-20},{20,20}},
        rotation=180,
        origin={328,352})));
  Modelica.Blocks.Interfaces.RealOutput u_v_5 "Connector of Real output signal"
    annotation (Placement(transformation(extent={{320,364},{340,384}})));

  Modelica.Blocks.Interfaces.RealOutput V_flow_7
    "Volume flow rate from port_a to port_b" annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={330,176})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_7 "Relative pressure signal"
    annotation (Placement(transformation(extent={{320,272},{340,292}})));

  Modelica.Blocks.Interfaces.RealInput w_v_7
    "=1: completely open, =0: completely closed"
    annotation (Placement(transformation(extent={{-20,-20},{20,20}},
        rotation=180,
        origin={338,252})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar4 annotation (Placement(
        transformation(
        extent={{10,-10},{-10,10}},
        rotation=-90,
        origin={-24,484})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar5 annotation (Placement(transformation(extent={{242,406},
            {262,426}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar7
    annotation (Placement(transformation(extent={{260,272},{280,292}})));

  To_m3hr to_m3hr4 annotation (Placement(transformation(extent={{38,486},{18,
            506}})));
  To_m3hr to_m3hr5 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={202,320})));
  To_m3hr to_m3hr7 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={152,176})));

  Modelica.Blocks.Interfaces.RealOutput P_pum_4 annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={98,544})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay2(delayTime=1)
    annotation (Placement(transformation(extent={{296,342},{276,362}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay5(delayTime=1)
    annotation (
      Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={142,502})));
  Modelica.Blocks.Logical.GreaterThreshold greaterThreshold1(threshold=0.01)
    annotation (Placement(transformation(extent={{164,466},{184,486}})));
  Modelica.Blocks.Logical.Switch switch_4
    annotation (Placement(transformation(extent={{144,438},{124,458}})));
  Modelica.Blocks.Sources.RealExpression realExpression1(y=0)
    annotation (Placement(transformation(extent={{178,424},{158,444}})));

  Modelica.Blocks.Interfaces.RealOutput level_tank_9
    annotation (Placement(transformation(extent={{10,-10},{-10,10}},
        rotation=90,
        origin={-28,50})));


  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay1(delayTime=1)
    annotation (Placement(transformation(extent={{224,242},{204,262}})));
  Modelica.Fluid.Pipes.StaticPipe pipe_1(
    allowFlowReversal=false,
    length=10,
    diameter(displayUnit="m") = 0.05,
    redeclare package Medium = Medium,
    height_ab=-7)
    annotation (Placement(transformation(extent={{88,346},{108,366}})));
equation
  P_pum_4 = pump_4.W_total;
  level_tank_9 = tank_9.level;
  connect(p_rel_4, p_rel_4)
    annotation (Line(points={{-26,548},{-26,548}},   color={0,0,127}));
  connect(pressure_4.p_rel, to_bar4.u)
    annotation (Line(points={{37,462},{-24,462},{-24,472}},
                                                        color={0,0,127}));
  connect(to_bar4.y, p_rel_4) annotation (Line(points={{-24,495},{-24,548},{-26,
          548}},                color={0,0,127}));
  connect(volumeFlow_4.V_flow, to_m3hr4.u)
    annotation (Line(points={{81,496},{40,496}}, color={0,0,127}));

  connect(to_m3hr4.y, V_flow_4) annotation (Line(points={{17,496},{4,496},{4,
          546}},            color={0,0,127}));
  connect(pressure_5.port_b,valve_5. port_b) annotation (Line(
      points={{200,388},{202,388},{202,364}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pressure_5.port_a,valve_5. port_a) annotation (Line(
      points={{180,388},{180,364},{182,364}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(volumeFlow_5.port_b,valve_5. port_a)
    annotation (Line(points={{158,356},{182,356},{182,364}},
                                             color={0,127,255}));
  connect(valve_5.port_b,sink_5. ports[1])
    annotation (Line(points={{202,364},{210,364},{210,396},{216,396}},
                                               color={0,127,255}));
  connect(p_rel_5,p_rel_5)
    annotation (Line(points={{328,416},{328,416}},
                                                 color={0,0,127}));
  connect(to_bar5.u,pressure_5. p_rel)
    annotation (Line(points={{240,416},{190,416},{190,397}},
                                                        color={0,0,127}));
  connect(to_bar5.y,p_rel_5)
    annotation (Line(points={{263,416},{328,416}},
                                                 color={0,0,127}));
  connect(V_flow_5, to_m3hr5.y) annotation (Line(points={{330,320},{213,320}},
                           color={0,0,127}));
  connect(volumeFlow_5.V_flow, to_m3hr5.u)
    annotation (Line(points={{148,345},{148,320},{190,320}}, color={0,0,127}));
  connect(PI_5.y,u_v_5)
    annotation (Line(points={{241,352},{230,352},{230,374},{330,374}},
                                                   color={0,0,127}));
  connect(PI_5.y,valve_5. opening) annotation (Line(points={{241,352},{192,352},
          {192,356}},                    color={0,0,127}));
  connect(PI_5.u_m, to_m3hr5.y)
    annotation (Line(points={{252,340},{252,320},{213,320}}, color={0,0,127}));

  connect(volumeFlow_4.port_b, pump_4.port_a)
    annotation (Line(points={{92,486},{92,468}}, color={0,127,255}));
  connect(pressure_4.port_b, pump_4.port_a) annotation (Line(points={{46,472},{
          46,478},{92,478},{92,468}}, color={0,127,255}));
  connect(pressure_4.port_a, pipe_5.port_a) annotation (Line(points={{46,452},{
          46,406},{36,406}},                   color={0,127,255}));
  connect(volumeFlow_4.port_a, source_4.ports[1])
    annotation (Line(points={{92,506},{92,524},{70,524}},  color={0,127,255}));
  connect(V_flow_4, V_flow_4)
    annotation (Line(points={{4,546},{4,546}},       color={0,0,127}));
  connect(fixedDelay2.y, PI_5.u_s)
    annotation (Line(points={{275,352},{264,352}},           color={0,0,127}));
  connect(fixedDelay2.u, w_v_5)
    annotation (Line(points={{298,352},{328,352}},           color={0,0,127}));
  connect(fixedDelay5.u, w_p_4)
    annotation (Line(points={{142,514},{144,514},{144,546}},
                                                   color={0,0,127}));
  connect(valveDiscreteRamp_4.open, greaterThreshold1.y) annotation (Line(
        points={{100,426},{192,426},{192,476},{185,476}}, color={255,0,255}));
  connect(greaterThreshold1.u, fixedDelay5.y)
    annotation (Line(points={{162,476},{142,476},{142,491}}, color={0,0,127}));
  connect(pump_4.port_b, valveDiscreteRamp_4.port_a)
    annotation (Line(points={{92,448},{92,436}}, color={0,127,255}));
  connect(valveDiscreteRamp_4.port_b, pipe_5.port_a)
    annotation (Line(points={{92,416},{92,406},{36,406}}, color={0,127,255}));
  connect(switch_4.u2, greaterThreshold1.y) annotation (Line(points={{146,448},{
          192,448},{192,476},{185,476}}, color={255,0,255}));
  connect(switch_4.u1, fixedDelay5.y) annotation (Line(points={{146,456},{146,476},
          {142,476},{142,491}}, color={0,0,127}));
  connect(realExpression1.y, switch_4.u3)
    annotation (Line(points={{157,434},{146,434},{146,440}}, color={0,0,127}));
  connect(switch_4.y, pump_4.N_in)
    annotation (Line(points={{123,448},{102,448},{102,458}}, color={0,0,127}));
  connect(pressure_7_1.port_b, valve_7_1.port_b) annotation (Line(
      points={{140,272},{140,234},{114,234},{114,242}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pressure_7_1.port_a, valve_7_1.port_a) annotation (Line(
      points={{140,292},{140,298},{114,298},{114,262}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(to_bar7.u, pressure_7_1.p_rel)
    annotation (Line(points={{258,282},{149,282}}, color={0,0,127}));
  connect(V_flow_7, to_m3hr7.y)
    annotation (Line(points={{330,176},{163,176}}, color={0,0,127}));
  connect(to_bar7.y, p_rel_7)
    annotation (Line(points={{281,282},{330,282}}, color={0,0,127}));
  connect(pipe_9.port_a,tank_9. ports[1])
    annotation (Line(points={{38,76},{-34,76},{-34,124}}, color={0,127,255}));
  connect(V_flow_7, V_flow_7)
    annotation (Line(points={{330,176},{330,176}}, color={0,0,127}));
  connect(pipe_5.port_b, teeJunctionIdeal_5.port_1) annotation (Line(points={{16,406},
          {10,406},{10,356},{26,356}},    color={0,127,255}));
  connect(valve_7_1.port_b, volumeFlow_7_1.port_a)
    annotation (Line(points={{114,242},{114,186}}, color={0,127,255}));
  connect(to_m3hr7.u, volumeFlow_7_1.V_flow)
    annotation (Line(points={{140,176},{125,176}}, color={0,0,127}));
  connect(volumeFlow_7_1.port_b, pipe_9.port_b)
    annotation (Line(points={{114,166},{114,76},{58,76}}, color={0,127,255}));
  connect(teeJunctionIdeal_5.port_3, valve_7_1.port_a) annotation (Line(points={{36,346},
          {36,268},{114,268},{114,262}},           color={0,127,255}));
  connect(fixedDelay1.u, w_v_7)
    annotation (Line(points={{226,252},{338,252}}, color={0,0,127}));
  connect(fixedDelay1.y, valve_7_1.opening)
    annotation (Line(points={{203,252},{122,252}}, color={0,0,127}));
  connect(teeJunctionIdeal_5.port_2, pipe_1.port_a)
    annotation (Line(points={{46,356},{88,356}}, color={0,127,255}));
  connect(pipe_1.port_b, volumeFlow_5.port_a)
    annotation (Line(points={{108,356},{138,356}}, color={0,127,255}));
 annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=180,
        origin={48,188})),
    Icon(coordinateSystem(preserveAspectRatio=false, extent={{-100,60},{320,540}})),
    Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-100,60},{320,
            540}})),
    uses(Modelica(version="4.0.0"), Custom_Pump_V2(version="1")),
    version="1");
end mini_tank;

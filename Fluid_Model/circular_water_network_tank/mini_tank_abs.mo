within ;
model mini_tank_abs
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
    annotation (Placement(transformation(extent={{-118,138},{-98,158}})));

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
    crossArea=10,
    level_start=0.5*tank_9.height,
    redeclare package Medium = Medium,
    use_portsData=false,
    nPorts=1) annotation (Placement(transformation(extent={{-66,250},{-26,290}})));

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
    annotation (Placement(transformation(extent={{224,376},{204,396}})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal teeJunctionIdeal_5(redeclare package
      Medium = Medium)
    annotation (Placement(transformation(extent={{82,366},{102,346}})));

  Modelica.Fluid.Valves.ValveLinear valve_5(
    allowFlowReversal=true,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{152,366},{172,346}})));

  Modelica.Fluid.Valves.ValveLinear valve_7_1(
    allowFlowReversal=true,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={104,254})));

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
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium,
    height_ab=-5)
    annotation (Placement(transformation(extent={{0,108},{20,128}})));
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
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{116,366},{136,346}})));
  Modelica.Fluid.Sensors.RelativePressure pressure_5(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{154,400},{174,380}})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_7(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={106,290})));
  Modelica.Fluid.Sensors.RelativePressure pressure_7(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{10,-10},{-10,10}},
        rotation=90,
        origin={142,196})));

  Modelica.Blocks.Continuous.LimPID PI_5(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{238,338},{218,358}})));

 Modelica.Blocks.Continuous.LimPID PI_7_1(
    controllerType=Modelica.Blocks.Types.SimpleController.PID,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{208,264},{188,244}})));

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
        origin={326,320})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_5 "Relative pressure signal"
    annotation (Placement(transformation(extent={{316,398},{336,418}})));
  Modelica.Blocks.Interfaces.RealInput w_v_5
    "=1: completely open, =0: completely closed"
    annotation (Placement(transformation(extent={{-20,-20},{20,20}},
        rotation=180,
        origin={328,352})));
  Modelica.Blocks.Interfaces.RealOutput u_v_5 "Connector of Real output signal"
    annotation (Placement(transformation(extent={{314,364},{334,384}})));

  Modelica.Blocks.Interfaces.RealOutput V_flow_7_1
    "Volume flow rate from port_a to port_b" annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={328,290})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_7 "Relative pressure signal"
    annotation (Placement(transformation(extent={{322,186},{342,206}})));
  Modelica.Blocks.Interfaces.RealInput w_v_7
    "=1: completely open, =0: completely closed"
    annotation (Placement(transformation(extent={{-20,-20},{20,20}},
        rotation=180,
        origin={330,254})));
  Modelica.Blocks.Interfaces.RealOutput u_v_7_1
    "Connector of Real output signal"
    annotation (Placement(transformation(extent={{320,208},{340,228}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar4 annotation (Placement(
        transformation(
        extent={{10,-10},{-10,10}},
        rotation=-90,
        origin={-24,484})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar5 annotation (Placement(transformation(extent={{240,398},
            {260,418}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar7 annotation (Placement(transformation(extent={{292,186},
            {312,206}})));

  To_m3hr to_m3hr4 annotation (Placement(transformation(extent={{38,486},{18,
            506}})));
  To_m3hr to_m3hr5 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={202,320})));
  To_m3hr to_m3hr7 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={142,290})));

  Modelica.Blocks.Interfaces.RealOutput P_pum_4 annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={98,544})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay2(delayTime=1)
    annotation (Placement(transformation(extent={{296,338},{276,358}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay5(delayTime=1)
    annotation (
      Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={142,502})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay7(delayTime=1)
    annotation (Placement(transformation(extent={{286,244},{266,264}})));
  Modelica.Blocks.Logical.GreaterThreshold greaterThreshold1(threshold=0.01)
    annotation (Placement(transformation(extent={{164,466},{184,486}})));
  Modelica.Blocks.Logical.Switch switch2
    annotation (Placement(transformation(extent={{144,438},{124,458}})));
  Modelica.Fluid.Valves.ValveDiscrete valveDiscreteRamp1(
    redeclare package Medium = Medium,
    allowFlowReversal=true,
    dp_nominal=1,
    m_flow_nominal=100)
    annotation (Placement(transformation(
        extent={{10,10},{-10,-10}},
        rotation=90,
        origin={92,426})));
  Modelica.Blocks.Sources.RealExpression realExpression1(y=0)
    annotation (Placement(transformation(extent={{178,424},{158,444}})));

  Modelica.Blocks.Interfaces.RealOutput level_tank_9
    annotation (Placement(transformation(extent={{-180,236},{-200,256}})));

  AbsoluteValueConnection absoluteValueConnection
    annotation (Placement(transformation(extent={{228,244},{248,264}})));
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
      points={{174,390},{184,390},{184,356},{172,356}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pressure_5.port_a,valve_5. port_a) annotation (Line(
      points={{154,390},{136,390},{136,356},{152,356}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(volumeFlow_5.port_b,valve_5. port_a)
    annotation (Line(points={{136,356},{152,356}},
                                             color={0,127,255}));
  connect(valve_5.port_b,sink_5. ports[1])
    annotation (Line(points={{172,356},{184,356},{184,386},{204,386}},
                                               color={0,127,255}));
  connect(p_rel_5,p_rel_5)
    annotation (Line(points={{326,408},{326,408}},
                                                 color={0,0,127}));
  connect(to_bar5.u,pressure_5. p_rel)
    annotation (Line(points={{238,408},{192,408},{192,412},{164,412},{164,399}},
                                                        color={0,0,127}));
  connect(to_bar5.y,p_rel_5)
    annotation (Line(points={{261,408},{326,408}},
                                                 color={0,0,127}));
  connect(V_flow_5, to_m3hr5.y) annotation (Line(points={{326,320},{213,320}},
                           color={0,0,127}));
  connect(volumeFlow_5.V_flow, to_m3hr5.u)
    annotation (Line(points={{126,345},{126,320},{190,320}}, color={0,0,127}));
  connect(PI_5.y,u_v_5)
    annotation (Line(points={{217,348},{208,348},{208,374},{324,374}},
                                                   color={0,0,127}));
  connect(PI_5.y,valve_5. opening) annotation (Line(points={{217,348},{180,348},
          {180,340},{162,340},{162,348}},color={0,0,127}));
  connect(PI_5.u_m, to_m3hr5.y)
    annotation (Line(points={{228,336},{220,336},{220,320},{213,320}},
                                                             color={0,0,127}));

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
  connect(teeJunctionIdeal_5.port_2, volumeFlow_5.port_a) annotation (Line(
        points={{102,356},{116,356}},                     color={0,127,255}));
  connect(fixedDelay2.u, w_v_5)
    annotation (Line(points={{298,348},{298,352},{328,352}}, color={0,0,127}));
  connect(fixedDelay5.u, w_p_4)
    annotation (Line(points={{142,514},{144,514},{144,546}},
                                                   color={0,0,127}));
  connect(valveDiscreteRamp1.open, greaterThreshold1.y) annotation (Line(points={{100,426},
          {192,426},{192,476},{185,476}},          color={255,0,255}));
  connect(greaterThreshold1.u, fixedDelay5.y)
    annotation (Line(points={{162,476},{142,476},{142,491}}, color={0,0,127}));
  connect(pump_4.port_b, valveDiscreteRamp1.port_a)
    annotation (Line(points={{92,448},{92,436}}, color={0,127,255}));
  connect(valveDiscreteRamp1.port_b, pipe_5.port_a)
    annotation (Line(points={{92,416},{92,406},{36,406}}, color={0,127,255}));
  connect(switch2.u2, greaterThreshold1.y) annotation (Line(points={{146,448},{
          192,448},{192,476},{185,476}}, color={255,0,255}));
  connect(switch2.u1, fixedDelay5.y) annotation (Line(points={{146,456},{146,
          476},{142,476},{142,491}}, color={0,0,127}));
  connect(realExpression1.y, switch2.u3) annotation (Line(points={{157,434},{
          150,434},{150,432},{146,432},{146,440}}, color={0,0,127}));
  connect(switch2.y, pump_4.N_in)
    annotation (Line(points={{123,448},{102,448},{102,458}},
                                                           color={0,0,127}));
  connect(pressure_7.port_b, valve_7_1.port_b) annotation (Line(
      points={{142,186},{144,186},{144,180},{104,180},{104,244}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pressure_7.port_a, valve_7_1.port_a) annotation (Line(
      points={{142,206},{142,272},{104,272},{104,264}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(to_bar7.u,pressure_7. p_rel)
    annotation (Line(points={{290,196},{151,196}},      color={0,0,127}));
  connect(V_flow_7_1, to_m3hr7.y)
    annotation (Line(points={{328,290},{153,290}}, color={0,0,127}));
  connect(volumeFlow_7.V_flow,to_m3hr7. u)
    annotation (Line(points={{117,290},{130,290}},           color={0,0,127}));
  connect(PI_7_1.u_m, to_m3hr7.y)
    annotation (Line(points={{198,266},{198,290},{153,290}}, color={0,0,127}));
  connect(fixedDelay7.u,w_v_7)
    annotation (Line(points={{288,254},{300,254},{300,256},{310,256},{310,252},{
          330,254}},                                         color={0,0,127}));
  connect(to_bar7.y, p_rel_7)
    annotation (Line(points={{313,196},{316,196},{316,194},{318,194},{318,196},{
          332,196}},                               color={0,0,127}));
  connect(pipe_9.port_a,tank_9. ports[1])
    annotation (Line(points={{0,118},{-48,118},{-48,250},{-46,250}},
                                                          color={0,127,255}));
  connect(V_flow_7_1, V_flow_7_1)
    annotation (Line(points={{328,290},{328,290}}, color={0,0,127}));
  connect(valve_7_1.port_a, volumeFlow_7.port_b) annotation (Line(points={{104,264},
          {104,270},{106,270},{106,280}},      color={0,127,255}));
  connect(PI_7_1.y, valve_7_1.opening) annotation (Line(points={{187,254},{112,254}},
                                                         color={0,0,127}));
  connect(PI_7_1.y, u_v_7_1) annotation (Line(points={{187,254},{158,254},{158,218},
          {330,218}},                          color={0,0,127}));
  connect(pipe_5.port_b, teeJunctionIdeal_5.port_1) annotation (Line(points={{16,
          406},{8,406},{8,356},{82,356}}, color={0,127,255}));
  connect(valve_7_1.port_b, pipe_9.port_b) annotation (Line(points={{104,244},{104,
          118},{20,118}}, color={0,127,255}));
  connect(volumeFlow_7.port_a, teeJunctionIdeal_5.port_3) annotation (Line(
        points={{106,300},{106,340},{92,340},{92,346}}, color={0,127,255}));
  connect(fixedDelay7.y, absoluteValueConnection.u)
    annotation (Line(points={{265,254},{250,254}}, color={0,0,127}));
  connect(absoluteValueConnection.y, PI_7_1.u_s)
    annotation (Line(points={{227,254},{210,254}}, color={0,0,127}));
 annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=180,
        origin={48,188})),
    Icon(coordinateSystem(preserveAspectRatio=false, extent={{-180,-160},{320,
            540}})),
    Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-180,-160},{
            320,540}})),
    uses(Modelica(version="4.0.0"), Custom_Pump_V2(version="1")),
    version="1");
end mini_tank_abs;

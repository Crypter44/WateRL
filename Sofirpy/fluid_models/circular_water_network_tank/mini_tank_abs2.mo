within ;
model mini_tank_abs2
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
    annotation (Placement(transformation(extent={{234,374},{214,394}})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal teeJunctionIdeal_5(redeclare package
      Medium = Medium)
    annotation (Placement(transformation(extent={{82,366},{102,346}})));

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
        origin={106,246})));

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
    annotation (Placement(transformation(extent={{138,366},{158,346}})));
  Modelica.Fluid.Sensors.RelativePressure pressure_5(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{180,398},{200,378}})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_7(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={106,290})));
  Modelica.Fluid.Sensors.RelativePressure pressure_7(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{10,-10},{-10,10}},
        rotation=90,
        origin={142,240})));

  Modelica.Blocks.Continuous.LimPID PI_5(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{262,334},{242,354}})));

 Modelica.Blocks.Continuous.LimPID PI_7_1(
    controllerType=Modelica.Blocks.Types.SimpleController.PID,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{208,244},{188,224}})));

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
        origin={328,292})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_7 "Relative pressure signal"
    annotation (Placement(transformation(extent={{318,164},{338,184}})));
  Modelica.Blocks.Interfaces.RealInput w_v_7
    "=1: completely open, =0: completely closed"
    annotation (Placement(transformation(extent={{-20,-20},{20,20}},
        rotation=180,
        origin={326,230})));
  Modelica.Blocks.Interfaces.RealOutput u_v_7_1
    "Connector of Real output signal"
    annotation (Placement(transformation(extent={{320,188},{340,208}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar4 annotation (Placement(
        transformation(
        extent={{10,-10},{-10,10}},
        rotation=-90,
        origin={-24,484})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar5 annotation (Placement(transformation(extent={{240,398},
            {260,418}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar7 annotation (Placement(transformation(extent={{288,164},
            {308,184}})));

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
    annotation (Placement(transformation(extent={{296,336},{276,356}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay5(delayTime=1)
    annotation (
      Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={142,502})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay7(delayTime=1)
    annotation (Placement(transformation(extent={{300,220},{280,240}})));
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
    annotation (Placement(transformation(extent={{234,224},{254,244}})));
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
    annotation (Line(points={{202,364},{208,364},{208,384},{214,384}},
                                               color={0,127,255}));
  connect(p_rel_5,p_rel_5)
    annotation (Line(points={{326,408},{326,408}},
                                                 color={0,0,127}));
  connect(to_bar5.u,pressure_5. p_rel)
    annotation (Line(points={{238,408},{190,408},{190,397}},
                                                        color={0,0,127}));
  connect(to_bar5.y,p_rel_5)
    annotation (Line(points={{261,408},{326,408}},
                                                 color={0,0,127}));
  connect(V_flow_5, to_m3hr5.y) annotation (Line(points={{326,320},{213,320}},
                           color={0,0,127}));
  connect(volumeFlow_5.V_flow, to_m3hr5.u)
    annotation (Line(points={{148,345},{148,320},{190,320}}, color={0,0,127}));
  connect(PI_5.y,u_v_5)
    annotation (Line(points={{241,344},{236,344},{236,342},{226,342},{226,356},
          {230,356},{230,374},{324,374}},          color={0,0,127}));
  connect(PI_5.y,valve_5. opening) annotation (Line(points={{241,344},{192,344},
          {192,356}},                    color={0,0,127}));
  connect(PI_5.u_m, to_m3hr5.y)
    annotation (Line(points={{252,332},{252,320},{213,320}}, color={0,0,127}));

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
        points={{102,356},{138,356}},                     color={0,127,255}));
  connect(fixedDelay2.y, PI_5.u_s)
    annotation (Line(points={{275,346},{276,344},{264,344}}, color={0,0,127}));
  connect(fixedDelay2.u, w_v_5)
    annotation (Line(points={{298,346},{298,352},{328,352}}, color={0,0,127}));
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
      points={{142,230},{144,230},{144,216},{106,216},{106,236}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pressure_7.port_a, valve_7_1.port_a) annotation (Line(
      points={{142,250},{142,260},{106,260},{106,256}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(to_bar7.u,pressure_7. p_rel)
    annotation (Line(points={{286,174},{172,174},{172,240},{151,240}},
                                                        color={0,0,127}));
  connect(V_flow_7_1, to_m3hr7.y)
    annotation (Line(points={{328,292},{240,292},{240,290},{153,290}},
                                                   color={0,0,127}));
  connect(volumeFlow_7.V_flow,to_m3hr7. u)
    annotation (Line(points={{117,290},{130,290}},           color={0,0,127}));
  connect(PI_7_1.u_m, to_m3hr7.y)
    annotation (Line(points={{198,246},{198,290},{153,290}}, color={0,0,127}));
  connect(fixedDelay7.u,w_v_7)
    annotation (Line(points={{302,230},{326,230}},           color={0,0,127}));
  connect(to_bar7.y, p_rel_7)
    annotation (Line(points={{309,174},{328,174}}, color={0,0,127}));
  connect(pipe_9.port_a,tank_9. ports[1])
    annotation (Line(points={{0,118},{-48,118},{-48,250},{-46,250}},
                                                          color={0,127,255}));
  connect(V_flow_7_1, V_flow_7_1)
    annotation (Line(points={{328,292},{328,292}}, color={0,0,127}));
  connect(valve_7_1.port_a, volumeFlow_7.port_b) annotation (Line(points={{106,256},
          {106,280}},                          color={0,127,255}));
  connect(PI_7_1.y, valve_7_1.opening) annotation (Line(points={{187,234},{160,
          234},{160,224},{124,224},{124,246},{114,246}}, color={0,0,127}));
  connect(PI_7_1.y, u_v_7_1) annotation (Line(points={{187,234},{180,234},{180,
          198},{330,198}},                     color={0,0,127}));
  connect(pipe_5.port_b, teeJunctionIdeal_5.port_1) annotation (Line(points={{16,
          406},{8,406},{8,356},{82,356}}, color={0,127,255}));
  connect(valve_7_1.port_b, pipe_9.port_b) annotation (Line(points={{106,236},{
          106,118},{20,118}}, color={0,127,255}));
  connect(volumeFlow_7.port_a, teeJunctionIdeal_5.port_3) annotation (Line(
        points={{106,300},{106,340},{92,340},{92,346}}, color={0,127,255}));
  connect(absoluteValueConnection.u, fixedDelay7.y) annotation (Line(points={{
          256,234},{270,234},{270,230},{279,230}}, color={0,0,127}));
  connect(absoluteValueConnection.y, PI_7_1.u_s)
    annotation (Line(points={{233,234},{210,234}}, color={0,0,127}));
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
end mini_tank_abs2;

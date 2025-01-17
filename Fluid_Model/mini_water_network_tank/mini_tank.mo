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
    level_start=0.5*tank_9.height,
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
        origin={114,250})));
  Modelica.Fluid.Valves.ValveLinear valve_7_2(
    allowFlowReversal=true,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium) annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=-90,
        origin={62,166})));

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
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{138,366},{158,346}})));
  Modelica.Fluid.Sensors.RelativePressure pressure_5(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{180,398},{200,378}})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_7_1(redeclare package Medium =
        Medium) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={114,198})));
  Modelica.Fluid.Sensors.RelativePressure pressure_7_1(redeclare package Medium =
        Medium) annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=90,
        origin={140,282})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_7_2(redeclare package Medium =
        Medium) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={62,118})));
  Modelica.Fluid.Sensors.RelativePressure pressure_7_2(redeclare package Medium =
        Medium) annotation (Placement(transformation(
        extent={{10,10},{-10,-10}},
        rotation=90,
        origin={28,168})));
  Modelica.Blocks.Continuous.LimPID PI_5(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{262,342},{242,362}})));

 Modelica.Blocks.Continuous.LimPID PI_7_1(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{190,238},{170,258}})));
 Modelica.Blocks.Continuous.LimPID PI_7_2(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{166,156},{146,176}})));

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
    annotation (Placement(transformation(extent={{318,406},{338,426}})));
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
        origin={330,198})));
  Modelica.Blocks.Interfaces.RealOutput V_flow_7_2
    annotation (Placement(transformation(extent={{320,108},{340,128}})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_7_1 "Relative pressure signal"
    annotation (Placement(transformation(extent={{320,272},{340,292}})));

  Modelica.Blocks.Interfaces.RealOutput p_rel_7_2 "Relative pressure signal"
    annotation (Placement(transformation(extent={{320,76},{340,96}})));
  Modelica.Blocks.Interfaces.RealInput w_v_7
    "=1: completely open, =0: completely closed"
    annotation (Placement(transformation(extent={{-20,-20},{20,20}},
        rotation=180,
        origin={340,234})));
  Modelica.Blocks.Interfaces.RealOutput u_v_7_1
    "Connector of Real output signal"
    annotation (Placement(transformation(extent={{320,256},{340,276}})));
  Modelica.Blocks.Interfaces.RealOutput u_v_7_2
    "Connector of Real output signal"
    annotation (Placement(transformation(extent={{320,174},{340,194}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar4 annotation (Placement(
        transformation(
        extent={{10,-10},{-10,10}},
        rotation=-90,
        origin={-24,484})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar5 annotation (Placement(transformation(extent={{242,406},
            {262,426}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar7_1 annotation (Placement(transformation(extent={{260,272},{280,292}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar7_2 annotation (Placement(transformation(extent={{154,76},{174,96}})));

  To_m3hr to_m3hr4 annotation (Placement(transformation(extent={{38,486},{18,
            506}})));
  To_m3hr to_m3hr5 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={202,320})));
  To_m3hr to_m3hr7_1 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={152,198})));
  To_m3hr to_m3hr7_2 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={132,118})));

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
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay7(delayTime=1)
    annotation (Placement(transformation(extent={{312,224},{292,244}})));
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
  Modelica.Blocks.Logical.GreaterEqualThreshold greaterEqualThreshold_7_1(threshold
      =0.01)
    annotation (Placement(transformation(extent={{262,224},{242,244}})));
  Modelica.Blocks.Logical.Switch switch_7_1
    annotation (Placement(transformation(extent={{222,238},{202,258}})));
  Modelica.Blocks.Sources.RealExpression realExpression_7_1
    annotation (Placement(transformation(extent={{204,214},{224,234}})));
  Modelica.Blocks.Logical.Switch switch_7_2
    annotation (Placement(transformation(extent={{194,176},{174,156}})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal teeJunctionIdeal_1(redeclare package
      Medium = Medium)
    annotation (Placement(transformation(extent={{82,306},{102,326}})));

  Modelica.Blocks.Math.Gain gain_7_2(k=-1)
    annotation (Placement(transformation(extent={{290,156},{270,176}})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal teeJunctionIdeal_2(redeclare package
      Medium = Medium)
    annotation (Placement(transformation(extent={{88,108},{108,88}})));
  Modelica.Blocks.Logical.GreaterEqualThreshold greaterEqualThreshold_7_2(threshold
      =0.01)
    annotation (Placement(transformation(extent={{252,156},{232,176}})));
  AbsoluteValueConnection absoluteValueConnection_7_1 annotation (Placement(
        transformation(
        extent={{10,10},{-10,-10}},
        rotation=90,
        origin={180,218})));
  AbsoluteValueConnection absoluteValueConnection_7_2 annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={156,138})));
  Modelica.Blocks.Sources.RealExpression realExpression_7_2
    annotation (Placement(transformation(extent={{222,164},{202,184}})));


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
  connect(V_flow_5, to_m3hr5.y) annotation (Line(points={{326,320},{213,320}},
                           color={0,0,127}));
  connect(volumeFlow_5.V_flow, to_m3hr5.u)
    annotation (Line(points={{148,345},{148,320},{190,320}}, color={0,0,127}));
  connect(PI_5.y,u_v_5)
    annotation (Line(points={{241,352},{230,352},{230,374},{324,374}},
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
  connect(teeJunctionIdeal_5.port_2, volumeFlow_5.port_a) annotation (Line(
        points={{102,356},{138,356}},                     color={0,127,255}));
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
      points={{140,272},{140,234},{114,234},{114,240}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pressure_7_1.port_a, valve_7_1.port_a) annotation (Line(
      points={{140,292},{140,298},{114,298},{114,260}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(to_bar7_1.u, pressure_7_1.p_rel)
    annotation (Line(points={{258,282},{149,282}}, color={0,0,127}));
  connect(V_flow_7_1, to_m3hr7_1.y)
    annotation (Line(points={{330,198},{163,198}}, color={0,0,127}));
  connect(fixedDelay7.u,w_v_7)
    annotation (Line(points={{314,234},{340,234}},           color={0,0,127}));
  connect(to_bar7_1.y, p_rel_7_1)
    annotation (Line(points={{281,282},{330,282}}, color={0,0,127}));
  connect(pipe_9.port_a,tank_9. ports[1])
    annotation (Line(points={{38,76},{-34,76},{-34,124}}, color={0,127,255}));
  connect(V_flow_7_1, V_flow_7_1)
    annotation (Line(points={{330,198},{330,198}}, color={0,0,127}));
  connect(greaterEqualThreshold_7_1.y, switch_7_1.u2) annotation (Line(points={{
          241,234},{234,234},{234,248},{224,248}}, color={255,0,255}));
  connect(greaterEqualThreshold_7_1.u, fixedDelay7.y)
    annotation (Line(points={{264,234},{291,234}}, color={0,0,127}));
  connect(realExpression_7_1.y, switch_7_1.u3) annotation (Line(points={{225,224},
          {232,224},{232,240},{224,240}}, color={0,0,127}));
  connect(teeJunctionIdeal_1.port_3, teeJunctionIdeal_5.port_3)
    annotation (Line(points={{92,326},{92,346}}, color={0,127,255}));
  connect(gain_7_2.u, fixedDelay7.y) annotation (Line(points={{292,166},{298,166},
          {298,218},{286,218},{286,234},{291,234}}, color={0,0,127}));
  connect(to_m3hr7_2.u, volumeFlow_7_2.V_flow)
    annotation (Line(points={{120,118},{73,118}}, color={0,0,127}));
  connect(fixedDelay7.y, switch_7_1.u1) annotation (Line(points={{291,234},{286,
          234},{286,256},{224,256}}, color={0,0,127}));
  connect(PI_7_1.u_s, switch_7_1.y)
    annotation (Line(points={{192,248},{201,248}}, color={0,0,127}));
  connect(PI_7_1.y, valve_7_1.opening) annotation (Line(points={{169,248},{122,248},
          {122,250}},                                    color={0,0,127}));
  connect(PI_7_1.y, u_v_7_1) annotation (Line(points={{169,248},{160,248},{160,266},
          {330,266}},                          color={0,0,127}));
  connect(switch_7_2.y, PI_7_2.u_s)
    annotation (Line(points={{173,166},{168,166}}, color={0,0,127}));
  connect(PI_7_2.y, valve_7_2.opening) annotation (Line(points={{145,166},{70,166}},
                                   color={0,0,127}));
  connect(PI_7_2.y, u_v_7_2) annotation (Line(points={{145,166},{138,166},{138,184},
          {330,184}},      color={0,0,127}));
  connect(pipe_5.port_b, teeJunctionIdeal_5.port_1) annotation (Line(points={{16,
          406},{8,406},{8,356},{82,356}}, color={0,127,255}));
  connect(pipe_9.port_b, teeJunctionIdeal_2.port_3)
    annotation (Line(points={{58,76},{98,76},{98,88}},    color={0,127,255}));
  connect(greaterEqualThreshold_7_2.u, gain_7_2.y)
    annotation (Line(points={{254,166},{269,166}}, color={0,0,127}));
  connect(greaterEqualThreshold_7_2.y, switch_7_2.u2)
    annotation (Line(points={{231,166},{196,166}}, color={255,0,255}));
  connect(absoluteValueConnection_7_1.y, PI_7_1.u_m)
    annotation (Line(points={{180,229},{180,236}}, color={0,0,127}));
  connect(absoluteValueConnection_7_2.y, PI_7_2.u_m)
    annotation (Line(points={{156,149},{156,154}}, color={0,0,127}));
  connect(to_m3hr7_2.y, absoluteValueConnection_7_2.u)
    annotation (Line(points={{143,118},{156,118},{156,126}}, color={0,0,127}));
  connect(absoluteValueConnection_7_2.u, V_flow_7_2)
    annotation (Line(points={{156,126},{156,118},{330,118}}, color={0,0,127}));
  connect(gain_7_2.y, switch_7_2.u1) annotation (Line(points={{269,166},{256,166},
          {256,150},{196,150},{196,158}}, color={0,0,127}));
  connect(teeJunctionIdeal_1.port_2, valve_7_1.port_a) annotation (Line(points={
          {102,316},{114,316},{114,260}}, color={0,127,255}));
  connect(valve_7_1.port_b, volumeFlow_7_1.port_a)
    annotation (Line(points={{114,240},{114,208}}, color={0,127,255}));
  connect(volumeFlow_7_1.port_b, teeJunctionIdeal_2.port_2)
    annotation (Line(points={{114,188},{114,98},{108,98}}, color={0,127,255}));
  connect(to_m3hr7_1.u, volumeFlow_7_1.V_flow)
    annotation (Line(points={{140,198},{125,198}}, color={0,0,127}));
  connect(to_m3hr7_1.y, absoluteValueConnection_7_1.u)
    annotation (Line(points={{163,198},{180,198},{180,206}}, color={0,0,127}));
  connect(volumeFlow_7_2.port_b, teeJunctionIdeal_2.port_1)
    annotation (Line(points={{62,108},{62,98},{88,98}}, color={0,127,255}));
  connect(realExpression_7_2.y, switch_7_2.u3)
    annotation (Line(points={{201,174},{196,174}}, color={0,0,127}));
  connect(pressure_7_2.port_b, valve_7_2.port_a) annotation (Line(
      points={{28,158},{28,148},{62,148},{62,156}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pressure_7_2.port_a, valve_7_2.port_b) annotation (Line(
      points={{28,178},{28,184},{62,184},{62,176}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(valve_7_2.port_a, volumeFlow_7_2.port_a)
    annotation (Line(points={{62,156},{62,128}}, color={0,127,255}));
  connect(teeJunctionIdeal_1.port_1, valve_7_2.port_b)
    annotation (Line(points={{82,316},{62,316},{62,176}}, color={0,127,255}));
  connect(to_bar7_2.u, pressure_7_2.p_rel) annotation (Line(points={{152,86},{10,
          86},{10,168},{19,168}}, color={0,0,127}));
  connect(to_bar7_2.y, p_rel_7_2)
    annotation (Line(points={{175,86},{330,86}}, color={0,0,127}));
  connect(p_rel_7_2, p_rel_7_2)
    annotation (Line(points={{330,86},{330,86}}, color={0,0,127}));
 annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=180,
        origin={48,188})),
    Icon(coordinateSystem(preserveAspectRatio=false, extent={{-180,60},{320,540}})),
    Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-180,60},{320,
            540}})),
    uses(Modelica(version="4.0.0"), Custom_Pump_V2(version="1")),
    version="1");
end mini_tank;

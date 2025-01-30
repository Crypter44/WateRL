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
    annotation (Placement(transformation(extent={{-74,170},{-54,190}})));

  // Power Curve -2.58707135371286 7.63830983123264 29.4033336406988 57.46
  // beta_1 * Q^3 +  beta_2 * Q^2 * n + beta_3 * Q * n^2 + beta_4 * n^3  with the pumps electrical Power "P" in kW, the pumps volumeflow "Q" in m³/h

    //energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial, //only needed for Medium = Modelica.Media.Water.StandardWaterOnePhase
    //massDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial, //only needed for Medium = Modelica.Media.Water.StandardWaterOnePhase

  Modelica.Fluid.Vessels.OpenTank tank_9(
    height=5,
    crossArea=10,
    redeclare package Medium = Medium,
    use_portsData=false,
    portsData={Modelica.Fluid.Vessels.BaseClasses.VesselPortsData(diameter=0.1)},
    use_HeatTransfer=false,
    nPorts=1) annotation (Placement(transformation(extent={{-48,208},{-8,248}})));

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
    annotation (Placement(transformation(extent={{238,398},{218,418}})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal teeJunctionIdeal_5(redeclare package
      Medium = Medium)
    annotation (Placement(transformation(extent={{28,378},{48,358}})));

  Modelica.Fluid.Valves.ValveIncompressible
                                    valve_5(
    allowFlowReversal=true,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{184,386},{204,366}})));

  Modelica.Fluid.Valves.ValveLinear valve_7_1(
    allowFlowReversal=true,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={114,252})));

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
    annotation (Placement(transformation(extent={{42,176},{62,196}})));
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
    redeclare package Medium = Medium, allowFlowReversal=true)
    annotation (Placement(transformation(extent={{140,378},{160,358}})));
  Modelica.Fluid.Sensors.RelativePressure pressure_5(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{182,410},{202,390}})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_7_1(redeclare package Medium =
        Medium, allowFlowReversal=true)
                annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={114,204})));
  Modelica.Fluid.Sensors.RelativePressure pressure_7_1(redeclare package Medium =
        Medium) annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=90,
        origin={140,276})));
  Modelica.Blocks.Continuous.LimPID PI_5(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{264,354},{244,374}})));

  Modelica.Blocks.Interfaces.RealOutput V_flow_4
    "Volume flow rate from port_a to port_b" annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={4,550})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_4 "Relative pressure signal"
    annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=-90,
        origin={-24,550})));
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
        origin={330,330})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_5 "Relative pressure signal"
    annotation (Placement(transformation(extent={{320,418},{340,438}})));
  Modelica.Blocks.Interfaces.RealInput w_v_5
    "=1: completely open, =0: completely closed"
    annotation (Placement(transformation(extent={{-20,-20},{20,20}},
        rotation=180,
        origin={330,364})));
  Modelica.Blocks.Interfaces.RealOutput u_v_5 "Connector of Real output signal"
    annotation (Placement(transformation(extent={{320,380},{340,400}})));

  Modelica.Blocks.Interfaces.RealOutput V_flow_7
    "Volume flow rate from port_a to port_b" annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={330,204})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_7 "Relative pressure signal"
    annotation (Placement(transformation(extent={{320,266},{340,286}})));

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
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar5 annotation (Placement(transformation(extent={{244,418},
            {264,438}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar7
    annotation (Placement(transformation(extent={{260,266},{280,286}})));

  To_m3hr to_m3hr4 annotation (Placement(transformation(extent={{38,486},{18,
            506}})));
  To_m3hr to_m3hr5 annotation (Placement(transformation(
        extent={{-12,-12},{12,12}},
        rotation=0,
        origin={202,330})));
  To_m3hr to_m3hr7 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={156,204})));

  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay2(delayTime=1)
    annotation (Placement(transformation(extent={{298,354},{278,374}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay5(delayTime=1)
    annotation (
      Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={142,502})));

  Modelica.Blocks.Interfaces.RealOutput level_tank_9
    annotation (Placement(transformation(extent={{10,-10},{-10,10}},
        rotation=90,
        origin={-30,150})));


  Modelica.Fluid.Pipes.StaticPipe pipe_1(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.05,
    redeclare package Medium = Medium,
    height_ab=0)
    annotation (Placement(transformation(extent={{60,358},{80,378}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay1(delayTime=1)
    annotation (Placement(transformation(extent={{240,242},{220,262}})));

  //MAGNA3 D 100-120 F
  Modelica.Fluid.Machines.PrescribedPump pump_4(
    redeclare package Medium = Medium,
    m_flow_start=0.001,
    checkValveHomotopy=Modelica.Fluid.Types.CheckValveHomotopyType.Closed,
    N_nominal=3033,
    use_powerCharacteristic=true,
    redeclare function powerCharacteristic =
        Modelica.Fluid.Machines.BaseClasses.PumpCharacteristics.quadraticPower
        (V_flow_nominal(displayUnit="m3/h") = {0,0.0027777777777778,0.0055555555555556},
          W_nominal={500,750,1000}),
    checkValve=true,
    redeclare function flowCharacteristic =
        Modelica.Fluid.Machines.BaseClasses.PumpCharacteristics.quadraticFlow (
          V_flow_nominal(displayUnit="m3/h") = {0,0.0027777777777778,0.0055555555555556},
          head_nominal={13,12,10}),
    V(displayUnit="l") = 0.01,
    energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
    massDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
    use_HeatTransfer=false,
    use_N_in=true) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={92,454})));
  Modelica.Blocks.Interfaces.RealOutput P_pum_4 annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={100,550})));
equation
  P_pum_4 = pump_4.W_total;
  level_tank_9 = tank_9.level;
  connect(p_rel_4, p_rel_4)
    annotation (Line(points={{-24,550},{-24,550}},   color={0,0,127}));
  connect(pressure_4.p_rel, to_bar4.u)
    annotation (Line(points={{37,462},{-24,462},{-24,472}},
                                                        color={0,0,127}));
  connect(to_bar4.y, p_rel_4) annotation (Line(points={{-24,495},{-24,550}},
                                color={0,0,127}));
  connect(volumeFlow_4.V_flow, to_m3hr4.u)
    annotation (Line(points={{81,496},{40,496}}, color={0,0,127}));

  connect(to_m3hr4.y, V_flow_4) annotation (Line(points={{17,496},{4,496},{4,550}},
                            color={0,0,127}));
  connect(pressure_5.port_b,valve_5. port_b) annotation (Line(
      points={{202,400},{204,400},{204,376}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pressure_5.port_a,valve_5. port_a) annotation (Line(
      points={{182,400},{182,376},{184,376}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(volumeFlow_5.port_b,valve_5. port_a)
    annotation (Line(points={{160,368},{170,368},{170,376},{184,376}},
                                             color={0,127,255}));
  connect(valve_5.port_b,sink_5. ports[1])
    annotation (Line(points={{204,376},{212,376},{212,408},{218,408}},
                                               color={0,127,255}));
  connect(p_rel_5,p_rel_5)
    annotation (Line(points={{330,428},{330,428}},
                                                 color={0,0,127}));
  connect(to_bar5.u,pressure_5. p_rel)
    annotation (Line(points={{242,428},{192,428},{192,409}},
                                                        color={0,0,127}));
  connect(to_bar5.y,p_rel_5)
    annotation (Line(points={{265,428},{330,428}},
                                                 color={0,0,127}));
  connect(V_flow_5, to_m3hr5.y) annotation (Line(points={{330,330},{215.2,330}},
                           color={0,0,127}));
  connect(volumeFlow_5.V_flow, to_m3hr5.u)
    annotation (Line(points={{150,357},{150,330},{187.6,330}},
                                                             color={0,0,127}));
  connect(PI_5.y,u_v_5)
    annotation (Line(points={{243,364},{232,364},{232,390},{330,390}},
                                                   color={0,0,127}));
  connect(PI_5.y,valve_5. opening) annotation (Line(points={{243,364},{194,364},
          {194,368}},                    color={0,0,127}));
  connect(PI_5.u_m, to_m3hr5.y)
    annotation (Line(points={{254,352},{254,330},{215.2,330}},
                                                             color={0,0,127}));

  connect(pressure_4.port_a, pipe_5.port_a) annotation (Line(points={{46,452},{
          46,406},{36,406}},                   color={0,127,255}));
  connect(volumeFlow_4.port_a, source_4.ports[1])
    annotation (Line(points={{92,506},{92,524},{70,524}},  color={0,127,255}));
  connect(V_flow_4, V_flow_4)
    annotation (Line(points={{4,550},{4,550}},       color={0,0,127}));
  connect(fixedDelay2.y, PI_5.u_s)
    annotation (Line(points={{277,364},{266,364}},           color={0,0,127}));
  connect(fixedDelay2.u, w_v_5)
    annotation (Line(points={{300,364},{330,364}},           color={0,0,127}));
  connect(fixedDelay5.u, w_p_4)
    annotation (Line(points={{142,514},{144,514},{144,546}},
                                                   color={0,0,127}));
  connect(pressure_7_1.port_b, valve_7_1.port_b) annotation (Line(
      points={{140,266},{140,234},{114,234},{114,242}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pressure_7_1.port_a, valve_7_1.port_a) annotation (Line(
      points={{140,286},{140,298},{114,298},{114,262}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(to_bar7.u, pressure_7_1.p_rel)
    annotation (Line(points={{258,276},{149,276}}, color={0,0,127}));
  connect(V_flow_7, to_m3hr7.y)
    annotation (Line(points={{330,204},{167,204}}, color={0,0,127}));
  connect(to_bar7.y, p_rel_7)
    annotation (Line(points={{281,276},{330,276}}, color={0,0,127}));
  connect(pipe_9.port_a,tank_9. ports[1])
    annotation (Line(points={{42,186},{-28,186},{-28,208}},
                                                          color={0,127,255}));
  connect(V_flow_7, V_flow_7)
    annotation (Line(points={{330,204},{330,204}}, color={0,0,127}));
  connect(pipe_5.port_b, teeJunctionIdeal_5.port_1) annotation (Line(points={{16,406},
          {10,406},{10,368},{28,368}},    color={0,127,255}));
  connect(valve_7_1.port_b, volumeFlow_7_1.port_a)
    annotation (Line(points={{114,242},{114,214}}, color={0,127,255}));
  connect(to_m3hr7.u, volumeFlow_7_1.V_flow)
    annotation (Line(points={{144,204},{125,204}}, color={0,0,127}));
  connect(volumeFlow_7_1.port_b, pipe_9.port_b)
    annotation (Line(points={{114,194},{114,186},{62,186}},
                                                          color={0,127,255}));
  connect(teeJunctionIdeal_5.port_3, valve_7_1.port_a) annotation (Line(points={{38,358},
          {38,268},{114,268},{114,262}},           color={0,127,255}));
  connect(teeJunctionIdeal_5.port_2, pipe_1.port_a)
    annotation (Line(points={{48,368},{60,368}}, color={0,127,255}));
  connect(fixedDelay1.y, valve_7_1.opening)
    annotation (Line(points={{219,252},{122,252}}, color={0,0,127}));
  connect(fixedDelay1.u, w_v_7)
    annotation (Line(points={{242,252},{338,252}}, color={0,0,127}));
  connect(pipe_1.port_b, volumeFlow_5.port_a)
    annotation (Line(points={{80,368},{140,368}}, color={0,127,255}));
  connect(volumeFlow_4.port_b, pump_4.port_a)
    annotation (Line(points={{92,486},{92,464}}, color={0,127,255}));
  connect(pump_4.port_b, pipe_5.port_a)
    annotation (Line(points={{92,444},{92,406},{36,406}}, color={0,127,255}));
  connect(fixedDelay5.y, pump_4.N_in)
    annotation (Line(points={{142,491},{142,454},{102,454}}, color={0,0,127}));
  connect(pressure_4.port_b, pump_4.port_a)
    annotation (Line(points={{46,472},{92,472},{92,464}}, color={0,127,255}));
 annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=180,
        origin={48,188})),
    Icon(coordinateSystem(preserveAspectRatio=false, extent={{-100,160},{320,
            540}})),
    Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-100,160},{320,
            540}})),
    uses(Modelica(version="4.0.0"), Custom_Pump_V2(version="1")),
    version="1");
end mini_tank;

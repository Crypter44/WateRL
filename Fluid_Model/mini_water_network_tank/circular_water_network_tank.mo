within ;
model circular_water_network_tank
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
    annotation (Placement(transformation(extent={{288,-66},{308,-46}})));

  // Power Curve -2.58707135371286 7.63830983123264 29.4033336406988 57.46
  // beta_1 * Q^3 +  beta_2 * Q^2 * n + beta_3 * Q * n^2 + beta_4 * n^3  with the pumps electrical Power "P" in kW, the pumps volumeflow "Q" in m³/h
  Custom_Pump_V2.BaseClasses_Custom.Pump_vs pump_1(
    redeclare package Medium = Medium,
    allowFlowReversal=true,
    m_flow_start=0,
    redeclare function flowCharacteristic =
        Custom_Pump_V2.BaseClasses_Custom.PumpCharacteristics.quadraticFlow (c=
            {-0.962830634026511,0.33404915911756,15.48238267}),
    redeclare function powerCharacteristic =
        Custom_Pump_V2.BaseClasses_Custom.PumpCharacteristics.cubicPower (c=
        {-2.58707135371286, 7.63830983123264, 29.4033336406988, 57.46, 0}),
    checkValve=false,
    rpm_rel=0.93969,
    use_N_in=true,
    V=0.1,
    use_HeatTransfer=false,
    redeclare model HeatTransfer =
        Modelica.Fluid.Vessels.BaseClasses.HeatTransfer.IdealHeatTransfer)
    "Pump_ID = 1 from pressure_booster_pumps_for_buildings"
    annotation (Placement(transformation(extent={{-54,-150},{-34,-130}})));

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
    nPorts=2) annotation (Placement(transformation(extent={{10,204},{50,244}})));

  Modelica.Fluid.Sources.FixedBoundary source_1(
    nPorts=1,
    redeclare package Medium = Medium,
    use_T=true,
    T=system.T_ambient,
    p=system.p_ambient)
    annotation (Placement(transformation(extent={{-174,-150},{-154,-130}})));

  Modelica.Fluid.Sources.FixedBoundary sink_2(
    redeclare package Medium = Medium,
    p=system.p_ambient,
    T=system.T_ambient,
    nPorts=1)
    annotation (Placement(transformation(extent={{-174,28},{-154,48}})));

  Modelica.Fluid.Sources.FixedBoundary sink_3(
    redeclare package Medium = Medium,
    p=system.p_ambient,
    T=system.T_ambient,
    nPorts=1)
    annotation (Placement(transformation(extent={{-176,380},{-156,400}})));

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

  Modelica.Fluid.Sources.FixedBoundary sink_6(
    redeclare package Medium = Medium,
    p=system.p_ambient,
    T=system.T_ambient,
    nPorts=1)
    annotation (Placement(transformation(extent={{248,-94},{228,-74}})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_1(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={-8,-46})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_2(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=270,
        origin={-8,38})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_3(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=90,
        origin={-8,356})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_4(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=270,
        origin={-8,390})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal teeJunctionIdeal_5(redeclare package
      Medium = Medium)
    annotation (Placement(transformation(extent={{82,366},{102,346}})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_6(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=180,
        origin={80,-46})));

  Modelica.Fluid.Valves.ValveLinear valve_2(
    allowFlowReversal=true,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-114,28},{-134,48}})));

  Modelica.Fluid.Valves.ValveLinear valve_3(
    allowFlowReversal=true,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-116,380},{-136,400}})));

  Modelica.Fluid.Valves.ValveLinear valve_5(
    allowFlowReversal=true,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{182,374},{202,354}})));

  Modelica.Fluid.Valves.ValveLinear valve_6(
    allowFlowReversal=true,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{176,-78},{196,-98}})));

  Modelica.Fluid.Valves.ValveLinear valve_7_1(
    allowFlowReversal=true,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={106,236})));
 Modelica.Fluid.Valves.ValveLinear valve_7_2(
    allowFlowReversal=true,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium) annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=-90,
        origin={74,182})));
  Modelica.Fluid.Valves.ValveLinear valve_8_1(
    allowFlowReversal=true,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium) annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=-90,
        origin={100,82})));
 Modelica.Fluid.Valves.ValveLinear valve_8_2(
    allowFlowReversal=true,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={50,60})));
  Modelica.Fluid.Pipes.StaticPipe pipe_1(
    allowFlowReversal=true,
    length=30,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium,
    height_ab=0)
    annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-8,-70})));

  Modelica.Fluid.Pipes.StaticPipe pipe_2(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-38,28},{-58,48}})));

  Modelica.Fluid.Pipes.StaticPipe pipe_3(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-8,74})));

  Modelica.Fluid.Pipes.StaticPipe pipe_4(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-32,380},{-52,400}})));

  Modelica.Fluid.Pipes.StaticPipe pipe_5(
    allowFlowReversal=true,
    length=30,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium,
    height_ab=0)
    annotation (Placement(transformation(extent={{36,396},{16,416}})));
  Modelica.Fluid.Pipes.StaticPipe pipe_6(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{32,346},{52,366}})));

  Modelica.Fluid.Pipes.StaticPipe pipe_7(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{9,-11},{-9,11}},
        rotation=90,
        origin={81,-23})));

  Modelica.Fluid.Pipes.StaticPipe pipe_8(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{38,-56},{58,-36}})));
  Modelica.Fluid.Pipes.StaticPipe pipe_9(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium,
    height_ab=-5)
    annotation (Placement(transformation(extent={{40,144},{60,164}})));
  Modelica.Fluid.Pipes.StaticPipe pipe_10(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium,
    height_ab=-5)
    annotation (Placement(transformation(extent={{38,116},{58,136}})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_1(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-116,-150},{-96,-130}})));
  Modelica.Fluid.Sensors.RelativePressure pressure_1(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-42,-96},{-62,-116}})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_2(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-80,28},{-100,48}})));
  Modelica.Fluid.Sensors.RelativePressure pressure_2(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-114,76},{-134,56}})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_3(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-82,380},{-102,400}})));
  Modelica.Fluid.Sensors.RelativePressure pressure_3(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-116,432},{-136,412}})));
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
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_6(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{122,-78},{142,-98}})));
  Modelica.Fluid.Sensors.RelativePressure pressure_6(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{176,-48},{196,-68}})));
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
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_8(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{10,-10},{-10,10}},
        rotation=-90,
        origin={100,110})));
  Modelica.Fluid.Sensors.RelativePressure pressure_8(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=90,
        origin={126,82})));
  Modelica.Blocks.Continuous.LimPID PI_2(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{-100,114},{-80,134}})));

  Modelica.Blocks.Continuous.LimPID PI_3(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.005,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{-102,478},{-82,498}})));

  Modelica.Blocks.Continuous.LimPID PI_5(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{262,334},{242,354}})));

  Modelica.Blocks.Continuous.LimPID PI_6(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{202,-138},{222,-118}})));

 Modelica.Blocks.Continuous.LimPID PI_7_1(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{196,244},{176,224}})));
 Modelica.Blocks.Continuous.LimPID PI_7_2(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{184,154},{164,174}})));
 Modelica.Blocks.Continuous.LimPID PI_8_1(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{186,96},{166,76}})));
 Modelica.Blocks.Continuous.LimPID PI_8_2(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{190,20},{170,40}})));
 Modelica.Blocks.Interfaces.RealOutput V_flow_1
    "Connector of Real output signal containing input signal u in another unit"
    annotation (Placement(transformation(extent={{-180,-100},{-200,-120}})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_1 "Relative pressure signal"
    annotation (Placement(transformation(extent={{-180,-100},{-200,-80}})));
  Modelica.Blocks.Interfaces.RealInput w_p_1
    "Prescribed rotational speed"
    annotation (Placement(transformation(extent={{-210,-80},{-170,-40}})));
  Modelica.Blocks.Interfaces.RealOutput V_flow_2
    "Connector of Real output signal containing input signal u in another unit"
    annotation (Placement(transformation(extent={{-190,92},{-210,112}})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_2 "Relative pressure signal"
    annotation (Placement(transformation(extent={{-188,70},{-208,90}})));
  Modelica.Blocks.Interfaces.RealInput w_v_2
    "Connector of setpoint input signal"
    annotation (Placement(transformation(extent={{-218,104},{-178,144}})));
  Modelica.Blocks.Interfaces.RealOutput u_v_2
    "Connector of actuator output signal" annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=0,
        origin={-184,144})));

  Modelica.Blocks.Interfaces.RealOutput V_flow_3
    "Connector of Real output signal containing input signal u in another unit"
    annotation (Placement(transformation(extent={{-176,454},{-196,474}})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_3 "Relative pressure signal"
    annotation (Placement(transformation(extent={{-176,434},{-196,454}})));
  Modelica.Blocks.Interfaces.RealInput w_v_3
    "Connector of setpoint input signal"    annotation (Placement(transformation(extent={{-206,
            468},{-166,508}})));

  Modelica.Blocks.Interfaces.RealOutput u_v_3
    "Connector of actuator output signal" annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=0,
        origin={-184,512})));

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

  Modelica.Blocks.Interfaces.RealOutput V_flow_6
    "Volume flow rate from port_a to port_b"
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={132,-170})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_6 "Relative pressure signal"
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={260,-172})));
  Modelica.Blocks.Interfaces.RealInput w_v_6
    "=1: completely open, =0: completely closed"
    annotation (Placement(transformation(extent={{-20,-20},{20,20}},
        rotation=90,
        origin={168,-174})));
  Modelica.Blocks.Interfaces.RealOutput u_v_6 "Connector of Real output signal"
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={232,-172})));
  Modelica.Blocks.Interfaces.RealOutput V_flow_7_1
    "Volume flow rate from port_a to port_b" annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={328,292})));
  Modelica.Blocks.Interfaces.RealOutput V_flow_7_2
    annotation (Placement(transformation(extent={{320,160},{340,180}})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_7 "Relative pressure signal"
    annotation (Placement(transformation(extent={{320,186},{340,206}})));
  Modelica.Blocks.Interfaces.RealInput w_v_7
    "=1: completely open, =0: completely closed"
    annotation (Placement(transformation(extent={{-20,-20},{20,20}},
        rotation=180,
        origin={326,230})));
  Modelica.Blocks.Interfaces.RealOutput u_v_7_1
    "Connector of Real output signal"
    annotation (Placement(transformation(extent={{318,272},{338,292}})));
  Modelica.Blocks.Interfaces.RealOutput V_flow_8_1
    "Volume flow rate from port_a to port_b" annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={330,110})));
    Modelica.Blocks.Interfaces.RealOutput V_flow_8_2
    annotation (Placement(transformation(extent={{320,14},{340,34}})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_8 "Relative pressure signal"
    annotation (Placement(transformation(extent={{316,-30},{336,-10}})));
  Modelica.Blocks.Interfaces.RealInput w_v_8
    "=1: completely open, =0: completely closed"
    annotation (Placement(transformation(extent={{-20,-20},{20,20}},
        rotation=180,
        origin={320,70})));
  Modelica.Blocks.Interfaces.RealOutput u_v_8_1
    "Connector of Real output signal"
    annotation (Placement(transformation(extent={{320,84},{340,104}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar annotation (Placement(transformation(extent={{-88,
            -100},{-108,-80}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar2 annotation (Placement(transformation(extent={{-148,70},{-168,90}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar3 annotation (Placement(transformation(extent={{-140,
            434},{-160,454}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar4 annotation (Placement(
        transformation(
        extent={{10,-10},{-10,10}},
        rotation=-90,
        origin={-24,484})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar5 annotation (Placement(transformation(extent={{240,398},
            {260,418}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar6 annotation (Placement(transformation(extent={{220,-62},
            {240,-42}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar7 annotation (Placement(transformation(extent={{290,186},
            {310,206}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar8 annotation (Placement(transformation(extent={{216,-26},
            {236,-6}})));

  To_m3hr to_m3hr  annotation (Placement(transformation(extent={{-112,-120},{
            -132,-100}})));
  To_m3hr to_m3hr2 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-90,82})));
  To_m3hr to_m3hr3 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-92,438})));
  To_m3hr to_m3hr4 annotation (Placement(transformation(extent={{38,486},{18,
            506}})));
  To_m3hr to_m3hr5 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={202,320})));
  To_m3hr to_m3hr6 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={132,-124})));
  To_m3hr to_m3hr7 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={142,290})));
  To_m3hr to_m3hr8 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={136,110})));

  Modelica.Blocks.Interfaces.RealOutput P_pum_1
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={-26,-170})));
  Modelica.Blocks.Interfaces.RealOutput P_pum_4 annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={98,544})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay(delayTime=1)
    annotation (Placement(transformation(extent={{-160,-70},{-140,-50}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay1(delayTime=1)
    annotation (Placement(transformation(extent={{172,-138},{192,-118}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay2(delayTime=1)
    annotation (Placement(transformation(extent={{296,336},{276,356}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay3(delayTime=1)
    annotation (Placement(transformation(extent={{-150,114},{-130,134}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay4(delayTime=1)
    annotation (Placement(transformation(extent={{-144,478},{-124,498}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay5(delayTime=1)
    annotation (
      Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={142,502})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay7(delayTime=1)
    annotation (Placement(transformation(extent={{300,220},{280,240}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay8(delayTime=1)
    annotation (Placement(transformation(extent={{294,60},{274,80}})));
  Modelica.Blocks.Logical.GreaterThreshold greaterThreshold(threshold=0.01)
    annotation (Placement(transformation(extent={{-122,-70},{-102,-50}})));
  Modelica.Blocks.Logical.Switch switch1
    annotation (Placement(transformation(extent={{-58,-40},{-38,-20}})));
  Modelica.Blocks.Sources.RealExpression realExpression(y=0)
    annotation (Placement(transformation(extent={{-38,-64},{-66,-46}})));
  Modelica.Fluid.Valves.ValveDiscrete valveDiscreteRamp(
    redeclare package Medium = Medium,
    allowFlowReversal=true,
    dp_nominal=1,
    m_flow_nominal=100)
    annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-8,-122})));
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
  Modelica.Blocks.Logical.GreaterThreshold      greaterThreshold2(threshold=
        0.01)
    annotation (Placement(transformation(extent={{254,68},{234,88}})));
  Modelica.Blocks.Logical.Switch switch_8_1
    annotation (Placement(transformation(extent={{218,76},{198,96}})));
  Modelica.Blocks.Logical.Switch switch_8_2
    annotation (Placement(transformation(extent={{220,18},{200,38}})));
  Modelica.Blocks.Sources.RealExpression realExpression3
    annotation (Placement(transformation(extent={{182,50},{202,70}})));
  Modelica.Blocks.Logical.GreaterEqualThreshold greaterEqualThreshold1
    annotation (Placement(transformation(extent={{264,206},{244,226}})));
  Modelica.Blocks.Logical.Switch switch_7_1
    annotation (Placement(transformation(extent={{222,224},{202,244}})));
  Modelica.Blocks.Sources.RealExpression realExpression4
    annotation (Placement(transformation(extent={{188,196},{208,216}})));
  Modelica.Blocks.Logical.Switch switch_7_2
    annotation (Placement(transformation(extent={{218,154},{198,174}})));
  Modelica.Blocks.Interfaces.RealOutput u_v_8_2
    "Connector of Real output signal"
    annotation (Placement(transformation(extent={{316,-10},{336,10}})));
  Modelica.Blocks.Interfaces.RealOutput u_v_7_2
    "Connector of Real output signal"
    annotation (Placement(transformation(extent={{318,124},{338,144}})));
  Modelica.Fluid.Fittings.TeeJunctionIdeal teeJunctionIdeal_1(redeclare package
      Medium = Medium)
    annotation (Placement(transformation(extent={{82,306},{102,326}})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_9(redeclare package Medium
      = Medium)
    annotation (Placement(transformation(extent={{10,10},{-10,-10}},
        rotation=-90,
        origin={72,264})));
  Modelica.Blocks.Math.Gain gain(k=-1)
    annotation (Placement(transformation(extent={{284,146},{264,166}})));

  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_10(redeclare package Medium
      = Medium)
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={46,24})));
  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_5(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=180,
        origin={78,2})));
  To_m3hr to_m3hr1 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={138,196})));
  To_m3hr to_m3hr9 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={84,22})));

  Modelica.Blocks.Math.Gain gain1(k=-1)
    annotation (Placement(transformation(extent={{256,10},{236,30}})));
  Modelica.Blocks.Logical.LessThreshold         lessThreshold(threshold=-0.01)
    annotation (Placement(transformation(extent={{254,36},{234,56}})));
  Modelica.Blocks.Logical.Switch switch_8_2_2
    annotation (Placement(transformation(extent={{160,32},{140,52}})));
equation
  P_pum_1 = pump_1.W_total;
  P_pum_4 = pump_4.W_total;
  level_tank_9 = tank_9.level;
  connect(volumeFlow_1.port_a, source_1.ports[1])
    annotation (Line(points={{-116,-140},{-154,-140}},
                                                   color={0,127,255}));
  connect(pressure_6.port_b,valve_6. port_b) annotation (Line(
      points={{196,-58},{196,-88}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pressure_6.port_a,valve_6. port_a) annotation (Line(
      points={{176,-58},{176,-88}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pipe_1.port_b,idealJunction_1. port_2)
    annotation (Line(points={{-8,-60},{-8,-56}}, color={0,127,255}));
  connect(p_rel_4, p_rel_4)
    annotation (Line(points={{-26,548},{-26,548}},   color={0,0,127}));
  connect(volumeFlow_6.port_b,valve_6. port_a)
    annotation (Line(points={{142,-88},{176,-88}},
                                             color={0,127,255}));
  connect(valve_6.port_b,sink_6. ports[1])
    annotation (Line(points={{196,-88},{222,-88},{222,-84},{228,-84}},
                                               color={0,127,255}));
  connect(p_rel_6,p_rel_6)
    annotation (Line(points={{260,-172},{260,-172}},
                                                 color={0,0,127}));
  connect(pressure_1.port_a, pipe_1.port_a) annotation (Line(points={{-42,-106},
          {-8,-106},{-8,-80}},                   color={0,127,255}));
  connect(to_bar.y, p_rel_1)
    annotation (Line(points={{-109,-90},{-190,-90}},color={0,0,127}));
  connect(to_bar.u, pressure_1.p_rel)
    annotation (Line(points={{-86,-90},{-52,-90},{-52,-97}}, color={0,0,127}));
  connect(to_bar6.u,pressure_6. p_rel)
    annotation (Line(points={{218,-52},{218,-49},{186,-49}},
                                                        color={0,0,127}));
  connect(to_bar6.y,p_rel_6)
    annotation (Line(points={{241,-52},{260,-52},{260,-172}},
                                                 color={0,0,127}));
  connect(pressure_4.p_rel, to_bar4.u)
    annotation (Line(points={{37,462},{-24,462},{-24,472}},
                                                        color={0,0,127}));
  connect(to_bar4.y, p_rel_4) annotation (Line(points={{-24,495},{-24,548},{-26,
          548}},                color={0,0,127}));
  connect(to_m3hr.u, volumeFlow_1.V_flow)
    annotation (Line(points={{-110,-110},{-106,-110},{-106,-129}},
                                                             color={0,0,127}));
  connect(volumeFlow_4.V_flow, to_m3hr4.u)
    annotation (Line(points={{81,496},{40,496}}, color={0,0,127}));
  connect(V_flow_6,to_m3hr6. y) annotation (Line(points={{132,-170},{132,-135}},
                          color={0,0,127}));
  connect(volumeFlow_6.V_flow,to_m3hr6. u)
    annotation (Line(points={{132,-99},{132,-112}},
                                                 color={0,0,127}));
  connect(PI_6.y,u_v_6)
    annotation (Line(points={{223,-128},{223,-130},{236,-130},{236,-154},{232,
          -154},{232,-172}},                       color={0,0,127}));
  connect(PI_6.y,valve_6. opening) annotation (Line(points={{223,-128},{226,
          -128},{226,-100},{186,-100},{186,-96}},
                                         color={0,0,127}));
  connect(pipe_4.port_b, volumeFlow_3.port_a)
    annotation (Line(points={{-52,390},{-82,390}},  color={0,127,255}));
  connect(volumeFlow_3.port_b, valve_3.port_a)
    annotation (Line(points={{-102,390},{-116,390}},
                                                   color={0,127,255}));
  connect(pressure_3.port_a, valve_3.port_a) annotation (Line(points={{-116,422},
          {-110,422},{-110,390},{-116,390}},
                                          color={0,127,255}));
  connect(pressure_3.p_rel, to_bar3.u)
    annotation (Line(points={{-126,431},{-126,444},{-138,444}},
                                                             color={0,0,127}));
  connect(to_bar3.y, p_rel_3)
    annotation (Line(points={{-161,444},{-186,444}},
                                                   color={0,0,127}));
  connect(volumeFlow_3.V_flow, to_m3hr3.u)
    annotation (Line(points={{-92,401},{-92,426}},
                                                 color={0,0,127}));
  connect(to_m3hr3.y, V_flow_3)
    annotation (Line(points={{-92,449},{-92,464},{-186,464}},color={0,0,127}));

  connect(valve_3.port_b, pressure_3.port_b) annotation (Line(points={{-136,390},
          {-142,390},{-142,422},{-136,422}},
                                          color={0,127,255}));
  connect(sink_3.ports[1], valve_3.port_b)
    annotation (Line(points={{-156,390},{-136,390}},
                                                   color={0,127,255}));
  connect(PI_6.u_m,to_m3hr6. y)
    annotation (Line(points={{212,-140},{212,-148},{132,-148},{132,-135}},
                                                           color={0,0,127}));
  connect(to_m3hr3.y, PI_3.u_m)
    annotation (Line(points={{-92,449},{-92,476}},color={0,0,127}));
  connect(PI_3.y, u_v_3)
    annotation (Line(points={{-81,488},{-74,488},{-74,512},{-184,512}},
                                                             color={0,0,127}));
  connect(to_m3hr.y, V_flow_1)
    annotation (Line(points={{-133,-110},{-190,-110}},
                                                     color={0,0,127}));
  connect(valve_3.opening, PI_3.y) annotation (Line(points={{-126,398},{-126,
          406},{-60,406},{-60,488},{-81,488}},
                                         color={0,0,127}));
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
  connect(pipe_2.port_b, volumeFlow_2.port_a)
    annotation (Line(points={{-58,38},{-80,38}}, color={0,127,255}));
  connect(volumeFlow_2.port_b, valve_2.port_a)
    annotation (Line(points={{-100,38},{-114,38}}, color={0,127,255}));
  connect(pressure_2.port_a,valve_2. port_a) annotation (Line(points={{-114,66},
          {-108,66},{-108,38},{-114,38}}, color={0,127,255}));
  connect(pressure_2.p_rel, to_bar2.u)
    annotation (Line(points={{-124,75},{-124,80},{-146,80}}, color={0,0,127}));
  connect(to_bar2.y, p_rel_2)
    annotation (Line(points={{-169,80},{-198,80}}, color={0,0,127}));
  connect(volumeFlow_2.V_flow, to_m3hr2.u)
    annotation (Line(points={{-90,49},{-90,70}}, color={0,0,127}));
  connect(to_m3hr2.y, V_flow_2)
    annotation (Line(points={{-90,93},{-90,102},{-200,102}}, color={0,0,127}));
  connect(valve_2.port_b,pressure_2. port_b) annotation (Line(points={{-134,38},
          {-140,38},{-140,66},{-134,66}}, color={0,127,255}));
  connect(sink_2.ports[1], valve_2.port_b)
    annotation (Line(points={{-154,38},{-134,38}}, color={0,127,255}));
  connect(to_m3hr2.y, PI_2.u_m)
    annotation (Line(points={{-90,93},{-90,112}}, color={0,0,127}));
  connect(PI_2.y, u_v_2) annotation (Line(points={{-79,124},{-70,124},{-70,144},
          {-184,144}},                     color={0,0,127}));
  connect(valve_2.opening, PI_2.y) annotation (Line(points={{-124,46},{-96,46},{
          -96,52},{-66,52},{-66,124},{-79,124}}, color={0,0,127}));
  connect(idealJunction_2.port_2, idealJunction_1.port_1)
    annotation (Line(points={{-8,28},{-8,-36}},        color={0,127,255}));
  connect(pipe_3.port_a, idealJunction_2.port_1)
    annotation (Line(points={{-8,64},{-8,48}}, color={0,127,255}));
  connect(pipe_8.port_b,idealJunction_6. port_2)
    annotation (Line(points={{58,-46},{70,-46}},
                                             color={0,127,255}));
  connect(idealJunction_6.port_1,volumeFlow_6. port_a)
    annotation (Line(points={{90,-46},{112,-46},{112,-88},{122,-88}},
                                               color={0,127,255}));
  connect(idealJunction_3.port_1, pipe_3.port_b)
    annotation (Line(points={{-8,346},{-8,84}}, color={0,127,255}));
  connect(idealJunction_3.port_2, idealJunction_4.port_2)
    annotation (Line(points={{-8,366},{-8,380}}, color={0,127,255}));
  connect(u_v_6,u_v_6)
    annotation (Line(points={{232,-172},{232,-172}}, color={0,0,127}));

  connect(idealJunction_1.port_3,pipe_8. port_a)
    annotation (Line(points={{2,-46},{38,-46}},
                                            color={0,127,255}));
  connect(pipe_2.port_a, idealJunction_2.port_3)
    annotation (Line(points={{-38,38},{-18,38}}, color={0,127,255}));
  connect(idealJunction_3.port_3, pipe_6.port_a)
    annotation (Line(points={{2,356},{32,356}}, color={0,127,255}));
  connect(pipe_7.port_b,idealJunction_6. port_3)
    annotation (Line(points={{81,-32},{80,-32},{80,-36}},
                                                 color={0,127,255}));
  connect(pipe_4.port_a, idealJunction_4.port_3)
    annotation (Line(points={{-32,390},{-18,390}}, color={0,127,255}));
  connect(pipe_5.port_b, idealJunction_4.port_1) annotation (Line(points={{16,406},
          {-8,406},{-8,400}},                   color={0,127,255}));
  connect(volumeFlow_1.port_b, pump_1.port_a)
    annotation (Line(points={{-96,-140},{-54,-140}},
                                                   color={0,127,255}));
  connect(pressure_1.port_b, pump_1.port_a) annotation (Line(points={{-62,-106},
          {-72,-106},{-72,-140},{-54,-140}},
                                         color={0,127,255}));
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
  connect(teeJunctionIdeal_5.port_1, pipe_6.port_b)
    annotation (Line(points={{82,356},{52,356}}, color={0,127,255}));
  connect(teeJunctionIdeal_5.port_2, volumeFlow_5.port_a) annotation (Line(
        points={{102,356},{138,356}},                     color={0,127,255}));
  connect(fixedDelay.u, w_p_1)
    annotation (Line(points={{-162,-60},{-170,-60},{-170,-64},{-176,-64},{-176,
          -60},{-190,-60}},                          color={0,0,127}));
  connect(fixedDelay1.y, PI_6.u_s)
    annotation (Line(points={{193,-128},{200,-128}},
                                                   color={0,0,127}));
  connect(fixedDelay1.u, w_v_6) annotation (Line(points={{170,-128},{168,-128},
          {168,-174}},color={0,0,127}));
  connect(fixedDelay2.y, PI_5.u_s)
    annotation (Line(points={{275,346},{276,344},{264,344}}, color={0,0,127}));
  connect(fixedDelay2.u, w_v_5)
    annotation (Line(points={{298,346},{298,352},{328,352}}, color={0,0,127}));
  connect(w_v_2, fixedDelay3.u) annotation (Line(points={{-198,124},{-152,124}},
                                  color={0,0,127}));
  connect(fixedDelay3.y, PI_2.u_s) annotation (Line(points={{-129,124},{-102,124}},
                                       color={0,0,127}));
  connect(fixedDelay4.y, PI_3.u_s)
    annotation (Line(points={{-123,488},{-104,488}}, color={0,0,127}));
  connect(fixedDelay4.u, w_v_3)
    annotation (Line(points={{-146,488},{-186,488}}, color={0,0,127}));
  connect(fixedDelay5.u, w_p_4)
    annotation (Line(points={{142,514},{144,514},{144,546}},
                                                   color={0,0,127}));
  connect(switch1.u1, fixedDelay.y) annotation (Line(points={{-60,-22},{-132,
          -22},{-132,-60},{-139,-60}},
                            color={0,0,127}));
  connect(greaterThreshold.y, switch1.u2) annotation (Line(points={{-101,-60},{
          -80,-60},{-80,-30},{-60,-30}},
                                   color={255,0,255}));
  connect(greaterThreshold.u, fixedDelay.y) annotation (Line(points={{-124,-60},
          {-139,-60}},                       color={0,0,127}));
  connect(realExpression.y, switch1.u3) annotation (Line(points={{-67.4,-55},{
          -67.4,-56},{-68,-56},{-68,-38},{-60,-38}},     color={0,0,127}));
  connect(switch1.y, pump_1.N_in) annotation (Line(points={{-37,-30},{-28,-30},
          {-28,-130},{-44,-130}},                                    color={0,0,
          127}));
  connect(pump_1.port_b, valveDiscreteRamp.port_a) annotation (Line(points={{-34,
          -140},{-8,-140},{-8,-132}}, color={0,127,255}));
  connect(valveDiscreteRamp.port_b, pipe_1.port_a) annotation (Line(points={{-8,-112},
          {-8,-80}},                        color={0,127,255}));
  connect(valveDiscreteRamp.open, greaterThreshold.y) annotation (Line(points={{-16,
          -122},{-22,-122},{-22,-74},{-70,-74},{-70,-60},{-101,-60}},
                                                  color={255,0,255}));
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
      points={{142,230},{144,230},{144,216},{106,216},{106,226}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pressure_7.port_a, valve_7_1.port_a) annotation (Line(
      points={{142,250},{142,260},{106,260},{106,246}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(to_bar7.u,pressure_7. p_rel)
    annotation (Line(points={{288,196},{168,196},{168,240},{151,240}},
                                                        color={0,0,127}));
  connect(V_flow_7_1, to_m3hr7.y)
    annotation (Line(points={{328,292},{240,292},{240,290},{153,290}},
                                                   color={0,0,127}));
  connect(volumeFlow_7.V_flow,to_m3hr7. u)
    annotation (Line(points={{117,290},{130,290}},           color={0,0,127}));
  connect(PI_7_1.u_m, to_m3hr7.y)
    annotation (Line(points={{186,246},{186,290},{153,290}}, color={0,0,127}));
  connect(fixedDelay7.u,w_v_7)
    annotation (Line(points={{302,230},{326,230}},           color={0,0,127}));
  connect(to_bar7.y, p_rel_7)
    annotation (Line(points={{311,196},{330,196}}, color={0,0,127}));
  connect(pressure_8.port_b, valve_8_1.port_b) annotation (Line(
      points={{126,92},{100,92}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pressure_8.port_a, valve_8_1.port_a) annotation (Line(
      points={{126,72},{100,72}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(V_flow_8_1, to_m3hr8.y)
    annotation (Line(points={{330,110},{147,110}}, color={0,0,127}));
  connect(volumeFlow_8.V_flow,to_m3hr8. u)
    annotation (Line(points={{111,110},{124,110}},           color={0,0,127}));
  connect(PI_8_1.u_m, to_m3hr8.y)
    annotation (Line(points={{176,98},{176,110},{147,110}}, color={0,0,127}));
  connect(fixedDelay8.u,w_v_8)
    annotation (Line(points={{296,70},{320,70}},             color={0,0,127}));
  connect(to_bar8.y, p_rel_8)
    annotation (Line(points={{237,-16},{304,-16},{304,-20},{326,-20}},
                                                 color={0,0,127}));
  connect(volumeFlow_8.port_a, valve_8_1.port_b)
    annotation (Line(points={{100,100},{100,92}},color={0,127,255}));
  connect(pipe_9.port_b, valve_7_1.port_b)
    annotation (Line(points={{60,154},{74,154},{74,160},{106,160},{106,226}},
                                                          color={0,127,255}));
  connect(pipe_9.port_a,tank_9. ports[1])
    annotation (Line(points={{40,154},{32,154},{32,204},{28,204}},
                                                          color={0,127,255}));
  connect(pipe_10.port_a,tank_9. ports[2])
    annotation (Line(points={{38,126},{32,126},{32,204}}, color={0,127,255}));
  connect(pipe_10.port_b, volumeFlow_8.port_b) annotation (Line(points={{58,126},
          {100,126},{100,120}},                   color={0,127,255}));
  connect(greaterThreshold2.y, switch_8_1.u2)
    annotation (Line(points={{233,78},{233,86},{220,86}}, color={255,0,255}));
  connect(realExpression3.y, switch_8_1.u3)
    annotation (Line(points={{203,60},{220,60},{220,78}}, color={0,0,127}));
  connect(to_bar8.u, pressure_8.p_rel)
    annotation (Line(points={{214,-16},{136,-16},{136,64},{148,64},{148,82},{135,
          82}},                                         color={0,0,127}));
  connect(V_flow_7_1, V_flow_7_1)
    annotation (Line(points={{328,292},{328,292}}, color={0,0,127}));
  connect(greaterEqualThreshold1.y, switch_7_1.u2) annotation (Line(points={{
          243,216},{236,216},{236,234},{224,234}}, color={255,0,255}));
  connect(valve_7_1.port_b, valve_7_2.port_a)
    annotation (Line(points={{106,226},{106,160},{74,160},{74,172}},
                                                 color={0,127,255}));
  connect(realExpression3.y, switch_8_2.u1)
    annotation (Line(points={{203,60},{222,60},{222,36}}, color={0,0,127}));
  connect(realExpression4.y, switch_7_2.u1) annotation (Line(points={{209,206},
          {228,206},{228,172},{220,172}}, color={0,0,127}));
  connect(switch_7_2.u2, greaterEqualThreshold1.y) annotation (Line(points={{
          220,164},{236,164},{236,216},{243,216}}, color={255,0,255}));
  connect(greaterEqualThreshold1.u, fixedDelay7.y) annotation (Line(points={{266,216},
          {272,216},{272,230},{279,230}},                              color={0,
          0,127}));
  connect(realExpression4.y, switch_7_1.u3) annotation (Line(points={{209,206},
          {232,206},{232,226},{224,226}}, color={0,0,127}));
  connect(fixedDelay8.y, greaterThreshold2.u) annotation (Line(points={{273,70},
          {273,72},{256,72},{256,78}}, color={0,0,127}));
  connect(teeJunctionIdeal_1.port_3, teeJunctionIdeal_5.port_3)
    annotation (Line(points={{92,326},{92,346}}, color={0,127,255}));
  connect(teeJunctionIdeal_1.port_2, volumeFlow_7.port_a) annotation (Line(
        points={{102,316},{106,316},{106,300}}, color={0,127,255}));
  connect(valve_7_1.port_a, volumeFlow_7.port_b) annotation (Line(points={{106,246},
          {106,280}},                          color={0,127,255}));
  connect(volumeFlow_9.port_a, valve_7_2.port_b) annotation (Line(points={{72,254},
          {72,192},{74,192}},               color={0,127,255}));
  connect(volumeFlow_9.port_b, teeJunctionIdeal_1.port_1)
    annotation (Line(points={{72,274},{72,316},{82,316}}, color={0,127,255}));
  connect(gain.u, fixedDelay7.y) annotation (Line(points={{286,156},{292,156},{
          292,176},{276,176},{276,216},{272,216},{272,230},{279,230}},
                               color={0,0,127}));
  connect(pipe_7.port_a, idealJunction_5.port_3)
    annotation (Line(points={{81,-14},{78,-14},{78,-8}},
                                                      color={0,127,255}));
  connect(idealJunction_5.port_1, valve_8_1.port_a) annotation (Line(points={{88,2},{
          116,2},{116,32},{100,32},{100,72}},       color={0,127,255}));
  connect(to_m3hr1.u, volumeFlow_9.V_flow) annotation (Line(points={{126,196},{
          92,196},{92,198},{56,198},{56,264},{61,264}},
                                                      color={0,0,127}));
  connect(to_m3hr1.y, PI_7_2.u_m) annotation (Line(points={{149,196},{156,196},
          {156,140},{174,140},{174,152}},
        color={0,0,127}));
  connect(to_m3hr9.u, volumeFlow_10.V_flow)
    annotation (Line(points={{72,22},{57,22},{57,24}}, color={0,0,127}));
  connect(to_m3hr9.y, PI_8_2.u_m) annotation (Line(points={{95,22},{144,22},{
          144,8},{180,8},{180,18}},    color={0,0,127}));
  connect(valve_8_2.port_b, volumeFlow_10.port_a)
    annotation (Line(points={{50,50},{50,34},{46,34}},
                                               color={0,127,255}));
  connect(volumeFlow_10.port_b, idealJunction_5.port_2)
    annotation (Line(points={{46,14},{46,2},{68,2}},   color={0,127,255}));
  connect(valve_8_2.port_a, pipe_10.port_b) annotation (Line(points={{50,70},{50,
          108},{68,108},{68,126},{58,126}}, color={0,127,255}));
  connect(PI_7_2.u_m, V_flow_7_2) annotation (Line(points={{174,152},{174,138},
          {254,138},{254,140},{308,140},{308,170},{330,170}},
                                         color={0,0,127}));
  connect(V_flow_8_2, to_m3hr9.y) annotation (Line(points={{330,24},{280,24},{
          280,6},{142,6},{142,22},{95,22}}, color={0,0,127}));
  connect(gain1.u, fixedDelay8.y) annotation (Line(points={{258,20},{268,20},{
          268,70},{273,70}}, color={0,0,127}));
  connect(fixedDelay7.y, switch_7_1.u1) annotation (Line(points={{279,230},{272,
          230},{272,242},{224,242}}, color={0,0,127}));
  connect(PI_7_1.u_s, switch_7_1.y)
    annotation (Line(points={{198,234},{201,234}}, color={0,0,127}));
  connect(PI_7_1.y, valve_7_1.opening) annotation (Line(points={{175,234},{160,
          234},{160,224},{124,224},{124,236},{114,236}}, color={0,0,127}));
  connect(PI_7_1.y, u_v_7_1) annotation (Line(points={{175,234},{168,234},{168,
          260},{296,260},{296,282},{328,282}}, color={0,0,127}));
  connect(gain.y, switch_7_2.u3)
    annotation (Line(points={{263,156},{220,156}}, color={0,0,127}));
  connect(switch_7_2.y, PI_7_2.u_s)
    annotation (Line(points={{197,164},{186,164}}, color={0,0,127}));
  connect(PI_7_2.y, valve_7_2.opening) annotation (Line(points={{163,164},{96,
          164},{96,182},{82,182}}, color={0,0,127}));
  connect(PI_7_2.y, u_v_7_2) annotation (Line(points={{163,164},{152,164},{152,
          134},{328,134}}, color={0,0,127}));
  connect(gain1.y, switch_8_2.u3)
    annotation (Line(points={{235,20},{222,20}}, color={0,0,127}));
  connect(switch_8_2.y, PI_8_2.u_s) annotation (Line(points={{199,28},{194,28},
          {194,30},{192,30}}, color={0,0,127}));
  connect(V_flow_8_1, PI_8_1.u_m)
    annotation (Line(points={{330,110},{176,110},{176,98}}, color={0,0,127}));
  connect(fixedDelay8.y, switch_8_1.u1) annotation (Line(points={{273,70},{270,
          70},{270,94},{220,94}}, color={0,0,127}));
  connect(switch_8_1.y, PI_8_1.u_s)
    annotation (Line(points={{197,86},{188,86}}, color={0,0,127}));
  connect(PI_8_1.y, u_v_8_1) annotation (Line(points={{165,86},{160,86},{160,
          104},{288,104},{288,94},{330,94}}, color={0,0,127}));
  connect(PI_8_1.y, valve_8_1.opening) annotation (Line(points={{165,86},{152,
          86},{152,60},{108,60},{108,82}}, color={0,0,127}));
  connect(fixedDelay8.y, lessThreshold.u) annotation (Line(points={{273,70},{
          268,70},{268,46},{256,46}}, color={0,0,127}));
  connect(lessThreshold.y, switch_8_2.u2)
    annotation (Line(points={{233,46},{233,28},{222,28}}, color={255,0,255}));
  connect(switch_8_2_2.y, valve_8_2.opening) annotation (Line(points={{139,42},
          {72,42},{72,60},{58,60}}, color={0,0,127}));
  connect(PI_8_2.y, switch_8_2_2.u1)
    annotation (Line(points={{169,30},{162,30},{162,50}}, color={0,0,127}));
  connect(switch_8_2_2.u3, realExpression3.y) annotation (Line(points={{162,34},
          {164,34},{164,4},{264,4},{264,12},{272,12},{272,52},{264,52},{264,100},
          {228,100},{228,60},{203,60}}, color={0,0,127}));
  connect(switch_8_2_2.u2, lessThreshold.y) annotation (Line(points={{162,42},{
          200,42},{200,46},{233,46}}, color={255,0,255}));
  connect(u_v_8_2, switch_8_2_2.y) annotation (Line(points={{326,0},{126,0},{
          126,42},{139,42}}, color={0,0,127}));
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
end circular_water_network_tank;

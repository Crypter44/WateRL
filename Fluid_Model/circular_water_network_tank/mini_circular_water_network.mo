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
    annotation (Placement(transformation(extent={{288,-40},{308,-20}})));

  Custom_Pump_V2.BaseClasses_Custom.Pump_vs pump_1(
    redeclare package Medium = Medium,
    allowFlowReversal=true,
    m_flow_start=0,
    redeclare function flowCharacteristic =
        Custom_Pump_V2.BaseClasses_Custom.PumpCharacteristics.quadraticFlow (c=
            {-0.962830634026511,0.33404915911756,15.48238267}),
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
    "Pump_ID = 1 from pressure_booster_pumps_for_buildings"
    annotation (Placement(transformation(extent={{-54,-124},{-34,-104}})));

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
        Custom_Pump_V2.BaseClasses_Custom.PumpCharacteristics.cubicPower (c={-0.14637,
            1.1881,23.0824,53.0304,6.0431}),
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
        origin={92,354})));

  Modelica.Fluid.Vessels.OpenTank tank(
    height=5,
    crossArea=10,
    level_start=0.2*tank.height,
    redeclare package Medium = Medium,
    use_portsData=false,
    nPorts=2) annotation (Placement(transformation(extent={{10,164},{50,204}})));

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

  Modelica.Fluid.Sources.FixedBoundary sink_3(
    redeclare package Medium = Medium,
    p=system.p_ambient,
    T=system.T_ambient,
    nPorts=1)
    annotation (Placement(transformation(extent={{-176,276},{-156,296}})));

  Modelica.Fluid.Sources.FixedBoundary source_4(
    nPorts=1,
    redeclare package Medium = Medium,
    use_T=true,
    T=system.T_ambient,
    p=system.p_ambient)
    annotation (Placement(transformation(extent={{50,410},{70,430}})));

   Modelica.Fluid.Sources.FixedBoundary sink_5(
    redeclare package Medium = Medium,
    p=system.p_ambient,
    T=system.T_ambient,
    nPorts=1)
    annotation (Placement(transformation(extent={{234,274},{214,294}})));

  Modelica.Fluid.Sources.FixedBoundary sink_6(
    redeclare package Medium = Medium,
    p=system.p_ambient,
    T=system.T_ambient,
    nPorts=1)
    annotation (Placement(transformation(extent={{248,-68},{228,-48}})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_1(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={-6,-10})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_2(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=270,
        origin={-8,38})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_3(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=90,
        origin={-8,256})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_4(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=270,
        origin={-8,286})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal teeJunctionIdeal_5(redeclare package
      Medium = Medium)
    annotation (Placement(transformation(extent={{82,266},{102,246}})));

  Modelica.Fluid.Fittings.TeeJunctionIdeal idealJunction_6(redeclare package
      Medium = Medium) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=180,
        origin={102,-10})));

  Modelica.Fluid.Valves.ValveLinear valve_2(
    allowFlowReversal=false,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-114,28},{-134,48}})));

  Modelica.Fluid.Valves.ValveLinear valve_3(
    allowFlowReversal=false,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-116,276},{-136,296}})));

  Modelica.Fluid.Valves.ValveLinear valve_5(
    allowFlowReversal=false,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{182,274},{202,254}})));

  Modelica.Fluid.Valves.ValveLinear valve_6(
    allowFlowReversal=false,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{176,-52},{196,-72}})));

  Modelica.Fluid.Valves.ValveLinear valve_7(
    allowFlowReversal=true,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={92,150})));

  Modelica.Fluid.Valves.ValveLinear valve_8(
    allowFlowReversal=true,
    dp_nominal=100000,
    m_flow_nominal=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{10,-10},{-10,10}},
        rotation=-90,
        origin={102,60})));

  Modelica.Fluid.Pipes.StaticPipe pipe_1(
    allowFlowReversal=true,
    length=30,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium,
    height_ab=0)
    annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-8,-44})));

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
    annotation (Placement(transformation(extent={{-36,276},{-56,296}})));

  Modelica.Fluid.Pipes.StaticPipe pipe_5(
    allowFlowReversal=true,
    length=30,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium,
    height_ab=0)
    annotation (Placement(transformation(extent={{36,292},{16,312}})));
  Modelica.Fluid.Pipes.StaticPipe pipe_6(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{32,246},{52,266}})));

  Modelica.Fluid.Pipes.StaticPipe pipe_7(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{9,-11},{-9,11}},
        rotation=90,
        origin={103,21})));

  Modelica.Fluid.Pipes.StaticPipe pipe_8(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{40,-20},{60,0}})));

  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_1(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-116,-124},{-96,-104}})));
  Modelica.Fluid.Sensors.RelativePressure pressure_1(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-42,-70},{-62,-90}})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_2(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-80,28},{-100,48}})));
  Modelica.Fluid.Sensors.RelativePressure pressure_2(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-114,76},{-134,56}})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_3(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-82,276},{-102,296}})));
  Modelica.Fluid.Sensors.RelativePressure pressure_3(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-116,328},{-136,308}})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_4(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=-90,
        origin={92,392})));
  Modelica.Fluid.Sensors.RelativePressure pressure_4(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=90,
        origin={46,358})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_5(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{138,266},{158,246}})));
  Modelica.Fluid.Sensors.RelativePressure pressure_5(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{180,298},{200,278}})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_6(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{122,-52},{142,-72}})));
  Modelica.Fluid.Sensors.RelativePressure pressure_6(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{176,-22},{196,-42}})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_7(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={92,192})));
  Modelica.Fluid.Sensors.RelativePressure pressure_7(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{10,-10},{-10,10}},
        rotation=90,
        origin={142,148})));
  Modelica.Fluid.Sensors.VolumeFlowRate volumeFlow_8(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{10,-10},{-10,10}},
        rotation=-90,
        origin={102,100})));
  Modelica.Fluid.Sensors.RelativePressure pressure_8(
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=90,
        origin={152,56})));
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
    annotation (Placement(transformation(extent={{-102,374},{-82,394}})));

  Modelica.Blocks.Continuous.LimPID PI_5(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{262,234},{242,254}})));

  Modelica.Blocks.Continuous.LimPID PI_6(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{202,-112},{222,-92}})));

 Modelica.Blocks.Continuous.LimPID PI_7(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{238,190},{218,170}})));

 Modelica.Blocks.Continuous.LimPID PI_8(
    controllerType=Modelica.Blocks.Types.SimpleController.PI,
    k=0.01,
    Ti=0.01,
    Td=0.1,
    yMax=1,
    yMin=0,
    initType=Modelica.Blocks.Types.Init.InitialState)
    annotation (Placement(transformation(extent={{254,96},{234,76}})));

 Modelica.Blocks.Interfaces.RealOutput V_flow_1
    "Connector of Real output signal containing input signal u in another unit"
    annotation (Placement(transformation(extent={{-180,-74},{-200,-94}})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_1 "Relative pressure signal"
    annotation (Placement(transformation(extent={{-180,-74},{-200,-54}})));
  Modelica.Blocks.Interfaces.RealInput w_p_1
    "Prescribed rotational speed"
    annotation (Placement(transformation(extent={{-210,-54},{-170,-14}})));
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
    annotation (Placement(transformation(extent={{-176,350},{-196,370}})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_3 "Relative pressure signal"
    annotation (Placement(transformation(extent={{-176,330},{-196,350}})));
  Modelica.Blocks.Interfaces.RealInput w_v_3
    "Connector of setpoint input signal"    annotation (Placement(transformation(extent={{-206,
            364},{-166,404}})));

  Modelica.Blocks.Interfaces.RealOutput u_v_3
    "Connector of actuator output signal" annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=0,
        origin={-184,408})));

  Modelica.Blocks.Interfaces.RealOutput V_flow_4
    "Volume flow rate from port_a to port_b" annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={4,442})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_4 "Relative pressure signal"
    annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=-90,
        origin={-26,444})));
  Modelica.Blocks.Interfaces.RealInput w_p_4
    "=1: completely open, =0: completely closed" annotation (Placement(
        transformation(
        extent={{20,-20},{-20,20}},
        rotation=90,
        origin={144,442})));

  Modelica.Blocks.Interfaces.RealOutput V_flow_5
    "Volume flow rate from port_a to port_b"
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=0,
        origin={326,220})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_5 "Relative pressure signal"
    annotation (Placement(transformation(extent={{316,298},{336,318}})));
  Modelica.Blocks.Interfaces.RealInput w_v_5
    "=1: completely open, =0: completely closed"
    annotation (Placement(transformation(extent={{-20,-20},{20,20}},
        rotation=180,
        origin={328,252})));
  Modelica.Blocks.Interfaces.RealOutput u_v_5 "Connector of Real output signal"
    annotation (Placement(transformation(extent={{314,264},{334,284}})));

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
        origin={168,-148})));
  Modelica.Blocks.Interfaces.RealOutput u_v_6 "Connector of Real output signal"
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={232,-146})));
  Modelica.Blocks.Interfaces.RealOutput V_flow_7
    "Volume flow rate from port_a to port_b"
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=0,
        origin={326,196})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_7 "Relative pressure signal"
    annotation (Placement(transformation(extent={{316,124},{336,144}})));
  Modelica.Blocks.Interfaces.RealInput w_v_7
    "=1: completely open, =0: completely closed"
    annotation (Placement(transformation(extent={{-20,-20},{20,20}},
        rotation=180,
        origin={324,178})));
  Modelica.Blocks.Interfaces.RealOutput u_v_7 "Connector of Real output signal"
    annotation (Placement(transformation(extent={{316,140},{336,160}})));
  Modelica.Blocks.Interfaces.RealOutput V_flow_8
    "Volume flow rate from port_a to port_b"
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=0,
        origin={324,104})));
  Modelica.Blocks.Interfaces.RealOutput p_rel_8 "Relative pressure signal"
    annotation (Placement(transformation(extent={{314,32},{334,52}})));
  Modelica.Blocks.Interfaces.RealInput w_v_8
    "=1: completely open, =0: completely closed"
    annotation (Placement(transformation(extent={{-20,-20},{20,20}},
        rotation=180,
        origin={322,86})));
  Modelica.Blocks.Interfaces.RealOutput u_v_8 "Connector of Real output signal"
    annotation (Placement(transformation(extent={{314,48},{334,68}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar annotation (Placement(transformation(extent={{-88,-74},
            {-108,-54}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar2 annotation (Placement(transformation(extent={{-148,70},{-168,90}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar3 annotation (Placement(transformation(extent={{-140,
            330},{-160,350}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar4 annotation (Placement(
        transformation(
        extent={{10,-10},{-10,10}},
        rotation=-90,
        origin={-24,380})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar5 annotation (Placement(transformation(extent={{240,298},
            {260,318}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar6 annotation (Placement(transformation(extent={{220,-36},
            {240,-16}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar7 annotation (Placement(transformation(extent={{240,124},
            {260,144}})));
  Modelica.Blocks.Math.UnitConversions.To_bar to_bar8 annotation (Placement(transformation(extent={{250,32},
            {270,52}})));

  To_m3hr to_m3hr  annotation (Placement(transformation(extent={{-112,-94},{-132,
            -74}})));
  To_m3hr to_m3hr2 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-90,82})));
  To_m3hr to_m3hr3 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-92,334})));
  To_m3hr to_m3hr4 annotation (Placement(transformation(extent={{38,382},{18,402}})));
  To_m3hr to_m3hr5 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={202,220})));
  To_m3hr to_m3hr6 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={132,-98})));
  To_m3hr to_m3hr7 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={130,196})));
  To_m3hr to_m3hr8 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={140,104})));

  Modelica.Blocks.Interfaces.RealOutput P_pum_1
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={-26,-144})));
  Modelica.Blocks.Interfaces.RealOutput P_pum_4 annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={98,440})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay(delayTime=1)
    annotation (Placement(transformation(extent={{-160,-44},{-140,-24}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay1(delayTime=1)
    annotation (Placement(transformation(extent={{172,-112},{192,-92}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay2(delayTime=1)
    annotation (Placement(transformation(extent={{296,236},{276,256}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay3(delayTime=1)
    annotation (Placement(transformation(extent={{-150,114},{-130,134}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay4(delayTime=1)
    annotation (Placement(transformation(extent={{-144,374},{-124,394}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay5(delayTime=1)
    annotation (
      Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={142,398})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay7(delayTime=1)
    annotation (Placement(transformation(extent={{276,168},{256,188}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay8(delayTime=1)
    annotation (Placement(transformation(extent={{286,76},{266,96}})));
  Modelica.Blocks.Logical.GreaterThreshold greaterThreshold(threshold=0.01)
    annotation (Placement(transformation(extent={{-122,-44},{-102,-24}})));
  Modelica.Blocks.Logical.Switch switch1
    annotation (Placement(transformation(extent={{-58,-14},{-38,6}})));
  Modelica.Blocks.Sources.RealExpression realExpression(y=0)
    annotation (Placement(transformation(extent={{-38,-38},{-66,-20}})));
  Modelica.Fluid.Valves.ValveDiscrete valveDiscreteRamp(
    redeclare package Medium = Medium,
    allowFlowReversal=true,
    dp_nominal=1,
    m_flow_nominal=100)
    annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-8,-96})));
  Modelica.Blocks.Logical.GreaterThreshold greaterThreshold1(threshold=0.01)
    annotation (Placement(transformation(extent={{164,362},{184,382}})));
  Modelica.Blocks.Logical.Switch switch2
    annotation (Placement(transformation(extent={{144,334},{124,354}})));
  Modelica.Fluid.Valves.ValveDiscrete valveDiscreteRamp1(
    redeclare package Medium = Medium,
    allowFlowReversal=true,
    dp_nominal=1,
    m_flow_nominal=100)
    annotation (Placement(transformation(
        extent={{10,10},{-10,-10}},
        rotation=90,
        origin={92,322})));
  Modelica.Blocks.Sources.RealExpression realExpression1(y=0)
    annotation (Placement(transformation(extent={{178,320},{158,340}})));



  ApproxAbs approxAbs_7
    annotation (Placement(transformation(extent={{184,170},{164,190}})));
  ApproxAbs approxAbs_8
    annotation (Placement(transformation(extent={{196,66},{176,86}})));
  Modelica.Fluid.Pipes.StaticPipe pipe_9(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium,
    height_ab=-5)
    annotation (Placement(transformation(extent={{46,120},{66,140}})));
  Modelica.Fluid.Pipes.StaticPipe pipe_10(
    allowFlowReversal=true,
    length=10,
    diameter(displayUnit="m") = 0.025,
    redeclare package Medium = Medium,
    height_ab=-5)
    annotation (Placement(transformation(extent={{46,96},{66,116}})));
equation
  P_pum_1 = pump_1.W_total;
  P_pum_4 = pump_4.W_total;
  connect(volumeFlow_1.port_a, source_1.ports[1])
    annotation (Line(points={{-116,-114},{-154,-114}},
                                                   color={0,127,255}));
  connect(pressure_6.port_b,valve_6. port_b) annotation (Line(
      points={{196,-32},{196,-62}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pressure_6.port_a,valve_6. port_a) annotation (Line(
      points={{176,-32},{176,-62}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pipe_1.port_b,idealJunction_1. port_2)
    annotation (Line(points={{-8,-34},{-6,-34},{-6,-20}},
                                                 color={0,127,255}));
  connect(p_rel_4, p_rel_4)
    annotation (Line(points={{-26,444},{-26,444}},   color={0,0,127}));
  connect(volumeFlow_6.port_b,valve_6. port_a)
    annotation (Line(points={{142,-62},{176,-62}},
                                             color={0,127,255}));
  connect(valve_6.port_b,sink_6. ports[1])
    annotation (Line(points={{196,-62},{222,-62},{222,-58},{228,-58}},
                                               color={0,127,255}));
  connect(p_rel_6,p_rel_6)
    annotation (Line(points={{260,-146},{260,-146}},
                                                 color={0,0,127}));
  connect(pressure_1.port_a, pipe_1.port_a) annotation (Line(points={{-42,-80},{
          -8,-80},{-8,-54}},                     color={0,127,255}));
  connect(to_bar.y, p_rel_1)
    annotation (Line(points={{-109,-64},{-190,-64}},color={0,0,127}));
  connect(to_bar.u, pressure_1.p_rel)
    annotation (Line(points={{-86,-64},{-52,-64},{-52,-71}}, color={0,0,127}));
  connect(to_bar6.u,pressure_6. p_rel)
    annotation (Line(points={{218,-26},{218,-23},{186,-23}},
                                                        color={0,0,127}));
  connect(to_bar6.y,p_rel_6)
    annotation (Line(points={{241,-26},{260,-26},{260,-146}},
                                                 color={0,0,127}));
  connect(pressure_4.p_rel, to_bar4.u)
    annotation (Line(points={{37,358},{-24,358},{-24,368}},
                                                        color={0,0,127}));
  connect(to_bar4.y, p_rel_4) annotation (Line(points={{-24,391},{-24,444},{-26,
          444}},                color={0,0,127}));
  connect(to_m3hr.u, volumeFlow_1.V_flow)
    annotation (Line(points={{-110,-84},{-106,-84},{-106,-103}},
                                                             color={0,0,127}));
  connect(volumeFlow_4.V_flow, to_m3hr4.u)
    annotation (Line(points={{81,392},{40,392}}, color={0,0,127}));
  connect(V_flow_6,to_m3hr6. y) annotation (Line(points={{132,-144},{132,-109}},
                          color={0,0,127}));
  connect(volumeFlow_6.V_flow,to_m3hr6. u)
    annotation (Line(points={{132,-73},{132,-86}},
                                                 color={0,0,127}));
  connect(PI_6.y,u_v_6)
    annotation (Line(points={{223,-102},{223,-104},{236,-104},{236,-128},{232,-128},
          {232,-146}},                             color={0,0,127}));
  connect(PI_6.y,valve_6. opening) annotation (Line(points={{223,-102},{226,-102},
          {226,-74},{186,-74},{186,-70}},color={0,0,127}));
  connect(pipe_4.port_b, volumeFlow_3.port_a)
    annotation (Line(points={{-56,286},{-82,286}},  color={0,127,255}));
  connect(volumeFlow_3.port_b, valve_3.port_a)
    annotation (Line(points={{-102,286},{-116,286}},
                                                   color={0,127,255}));
  connect(pressure_3.port_a, valve_3.port_a) annotation (Line(points={{-116,318},
          {-110,318},{-110,286},{-116,286}},
                                          color={0,127,255}));
  connect(pressure_3.p_rel, to_bar3.u)
    annotation (Line(points={{-126,327},{-126,340},{-138,340}},
                                                             color={0,0,127}));
  connect(to_bar3.y, p_rel_3)
    annotation (Line(points={{-161,340},{-186,340}},
                                                   color={0,0,127}));
  connect(volumeFlow_3.V_flow, to_m3hr3.u)
    annotation (Line(points={{-92,297},{-92,322}},
                                                 color={0,0,127}));
  connect(to_m3hr3.y, V_flow_3)
    annotation (Line(points={{-92,345},{-92,360},{-186,360}},color={0,0,127}));

  connect(valve_3.port_b, pressure_3.port_b) annotation (Line(points={{-136,286},
          {-142,286},{-142,318},{-136,318}},
                                          color={0,127,255}));
  connect(sink_3.ports[1], valve_3.port_b)
    annotation (Line(points={{-156,286},{-136,286}},
                                                   color={0,127,255}));
  connect(PI_6.u_m,to_m3hr6. y)
    annotation (Line(points={{212,-114},{212,-122},{132,-122},{132,-109}},
                                                           color={0,0,127}));
  connect(to_m3hr3.y, PI_3.u_m)
    annotation (Line(points={{-92,345},{-92,372}},color={0,0,127}));
  connect(PI_3.y, u_v_3)
    annotation (Line(points={{-81,384},{-74,384},{-74,408},{-184,408}},
                                                             color={0,0,127}));
  connect(to_m3hr.y, V_flow_1)
    annotation (Line(points={{-133,-84},{-190,-84}}, color={0,0,127}));
  connect(valve_3.opening, PI_3.y) annotation (Line(points={{-126,294},{-126,302},
          {-60,302},{-60,384},{-81,384}},color={0,0,127}));
  connect(to_m3hr4.y, V_flow_4) annotation (Line(points={{17,392},{4,392},{4,442}},
                            color={0,0,127}));
  connect(pressure_5.port_b,valve_5. port_b) annotation (Line(
      points={{200,288},{202,288},{202,264}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pressure_5.port_a,valve_5. port_a) annotation (Line(
      points={{180,288},{180,264},{182,264}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(volumeFlow_5.port_b,valve_5. port_a)
    annotation (Line(points={{158,256},{182,256},{182,264}},
                                             color={0,127,255}));
  connect(valve_5.port_b,sink_5. ports[1])
    annotation (Line(points={{202,264},{208,264},{208,284},{214,284}},
                                               color={0,127,255}));
  connect(p_rel_5,p_rel_5)
    annotation (Line(points={{326,308},{326,308}},
                                                 color={0,0,127}));
  connect(to_bar5.u,pressure_5. p_rel)
    annotation (Line(points={{238,308},{190,308},{190,297}},
                                                        color={0,0,127}));
  connect(to_bar5.y,p_rel_5)
    annotation (Line(points={{261,308},{326,308}},
                                                 color={0,0,127}));
  connect(V_flow_5, to_m3hr5.y) annotation (Line(points={{326,220},{213,220}},
                           color={0,0,127}));
  connect(volumeFlow_5.V_flow, to_m3hr5.u)
    annotation (Line(points={{148,245},{148,220},{190,220}}, color={0,0,127}));
  connect(PI_5.y,u_v_5)
    annotation (Line(points={{241,244},{236,244},{236,242},{226,242},{226,256},{
          230,256},{230,274},{324,274}},           color={0,0,127}));
  connect(PI_5.y,valve_5. opening) annotation (Line(points={{241,244},{192,244},
          {192,256}},                    color={0,0,127}));
  connect(PI_5.u_m, to_m3hr5.y)
    annotation (Line(points={{252,232},{252,220},{213,220}}, color={0,0,127}));
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
    annotation (Line(points={{-8,28},{-8,0},{-6,0}},   color={0,127,255}));
  connect(pipe_3.port_a, idealJunction_2.port_1)
    annotation (Line(points={{-8,64},{-8,48}}, color={0,127,255}));
  connect(pipe_8.port_b,idealJunction_6. port_2)
    annotation (Line(points={{60,-10},{92,-10}},
                                             color={0,127,255}));
  connect(idealJunction_6.port_1,volumeFlow_6. port_a)
    annotation (Line(points={{112,-10},{118,-10},{118,-62},{122,-62}},
                                               color={0,127,255}));
  connect(idealJunction_3.port_1, pipe_3.port_b)
    annotation (Line(points={{-8,246},{-8,84}}, color={0,127,255}));
  connect(idealJunction_3.port_2, idealJunction_4.port_2)
    annotation (Line(points={{-8,266},{-8,276}}, color={0,127,255}));
  connect(u_v_6,u_v_6)
    annotation (Line(points={{232,-146},{232,-146}}, color={0,0,127}));

  connect(idealJunction_1.port_3,pipe_8. port_a)
    annotation (Line(points={{4,-10},{40,-10}},
                                            color={0,127,255}));
  connect(pipe_2.port_a, idealJunction_2.port_3)
    annotation (Line(points={{-38,38},{-18,38}}, color={0,127,255}));
  connect(idealJunction_3.port_3, pipe_6.port_a)
    annotation (Line(points={{2,256},{32,256}}, color={0,127,255}));
  connect(pipe_7.port_b,idealJunction_6. port_3)
    annotation (Line(points={{103,12},{102,12},{102,0}},
                                                 color={0,127,255}));
  connect(pipe_4.port_a, idealJunction_4.port_3)
    annotation (Line(points={{-36,286},{-18,286}}, color={0,127,255}));
  connect(pipe_5.port_b, idealJunction_4.port_1) annotation (Line(points={{16,302},
          {-8,302},{-8,296}},                   color={0,127,255}));
  connect(volumeFlow_1.port_b, pump_1.port_a)
    annotation (Line(points={{-96,-114},{-54,-114}},
                                                   color={0,127,255}));
  connect(pressure_1.port_b, pump_1.port_a) annotation (Line(points={{-62,-80},{
          -72,-80},{-72,-114},{-54,-114}},
                                         color={0,127,255}));
  connect(volumeFlow_4.port_b, pump_4.port_a)
    annotation (Line(points={{92,382},{92,364}}, color={0,127,255}));
  connect(pressure_4.port_b, pump_4.port_a) annotation (Line(points={{46,368},{46,
          374},{92,374},{92,364}},    color={0,127,255}));
  connect(pressure_4.port_a, pipe_5.port_a) annotation (Line(points={{46,348},{46,
          302},{36,302}},                      color={0,127,255}));
  connect(volumeFlow_4.port_a, source_4.ports[1])
    annotation (Line(points={{92,402},{92,420},{70,420}},  color={0,127,255}));
  connect(V_flow_4, V_flow_4)
    annotation (Line(points={{4,442},{4,442}},       color={0,0,127}));
  connect(teeJunctionIdeal_5.port_1, pipe_6.port_b)
    annotation (Line(points={{82,256},{52,256}}, color={0,127,255}));
  connect(teeJunctionIdeal_5.port_2, volumeFlow_5.port_a) annotation (Line(
        points={{102,256},{138,256}},                     color={0,127,255}));
  connect(fixedDelay.u, w_p_1)
    annotation (Line(points={{-162,-34},{-170,-34},{-170,-38},{-176,-38},{-176,-34},
          {-190,-34}},                               color={0,0,127}));
  connect(fixedDelay1.y, PI_6.u_s)
    annotation (Line(points={{193,-102},{200,-102}},
                                                   color={0,0,127}));
  connect(fixedDelay1.u, w_v_6) annotation (Line(points={{170,-102},{168,-102},{
          168,-148}}, color={0,0,127}));
  connect(fixedDelay2.y, PI_5.u_s)
    annotation (Line(points={{275,246},{276,244},{264,244}}, color={0,0,127}));
  connect(fixedDelay2.u, w_v_5)
    annotation (Line(points={{298,246},{298,252},{328,252}}, color={0,0,127}));
  connect(w_v_2, fixedDelay3.u) annotation (Line(points={{-198,124},{-152,124}},
                                  color={0,0,127}));
  connect(fixedDelay3.y, PI_2.u_s) annotation (Line(points={{-129,124},{-102,124}},
                                       color={0,0,127}));
  connect(fixedDelay4.y, PI_3.u_s)
    annotation (Line(points={{-123,384},{-104,384}}, color={0,0,127}));
  connect(fixedDelay4.u, w_v_3)
    annotation (Line(points={{-146,384},{-186,384}}, color={0,0,127}));
  connect(fixedDelay5.u, w_p_4)
    annotation (Line(points={{142,410},{144,410},{144,442}},
                                                   color={0,0,127}));
  connect(switch1.u1, fixedDelay.y) annotation (Line(points={{-60,4},{-132,4},{-132,
          -34},{-139,-34}}, color={0,0,127}));
  connect(greaterThreshold.y, switch1.u2) annotation (Line(points={{-101,-34},{-80,
          -34},{-80,-4},{-60,-4}}, color={255,0,255}));
  connect(greaterThreshold.u, fixedDelay.y) annotation (Line(points={{-124,-34},
          {-139,-34}},                       color={0,0,127}));
  connect(realExpression.y, switch1.u3) annotation (Line(points={{-67.4,-29},{-67.4,
          -30},{-68,-30},{-68,-12},{-60,-12}},           color={0,0,127}));
  connect(switch1.y, pump_1.N_in) annotation (Line(points={{-37,-4},{-28,-4},{-28,
          -104},{-44,-104}},                                         color={0,0,
          127}));
  connect(pump_1.port_b, valveDiscreteRamp.port_a) annotation (Line(points={{-34,
          -114},{-8,-114},{-8,-106}}, color={0,127,255}));
  connect(valveDiscreteRamp.port_b, pipe_1.port_a) annotation (Line(points={{-8,-86},
          {-8,-54}},                        color={0,127,255}));
  connect(valveDiscreteRamp.open, greaterThreshold.y) annotation (Line(points={{-16,-96},
          {-22,-96},{-22,-48},{-70,-48},{-70,-34},{-101,-34}},
                                                  color={255,0,255}));
  connect(valveDiscreteRamp1.open, greaterThreshold1.y) annotation (Line(points={{100,322},
          {192,322},{192,372},{185,372}},          color={255,0,255}));
  connect(greaterThreshold1.u, fixedDelay5.y)
    annotation (Line(points={{162,372},{142,372},{142,387}}, color={0,0,127}));
  connect(pump_4.port_b, valveDiscreteRamp1.port_a)
    annotation (Line(points={{92,344},{92,332}}, color={0,127,255}));
  connect(valveDiscreteRamp1.port_b, pipe_5.port_a)
    annotation (Line(points={{92,312},{92,302},{36,302}}, color={0,127,255}));
  connect(switch2.u2, greaterThreshold1.y) annotation (Line(points={{146,344},{192,
          344},{192,372},{185,372}},     color={255,0,255}));
  connect(switch2.u1, fixedDelay5.y) annotation (Line(points={{146,352},{146,372},
          {142,372},{142,387}},      color={0,0,127}));
  connect(realExpression1.y, switch2.u3) annotation (Line(points={{157,330},{150,
          330},{150,328},{146,328},{146,336}},     color={0,0,127}));
  connect(switch2.y, pump_4.N_in)
    annotation (Line(points={{123,344},{102,344},{102,354}},
                                                           color={0,0,127}));
  connect(pressure_7.port_b,valve_7. port_b) annotation (Line(
      points={{142,138},{144,138},{144,132},{92,132},{92,140}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pressure_7.port_a,valve_7. port_a) annotation (Line(
      points={{142,158},{142,164},{140,164},{140,168},{92,168},{92,160}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(volumeFlow_7.port_b,valve_7. port_a)
    annotation (Line(points={{92,182},{92,160}},
                                             color={0,127,255}));
  connect(to_bar7.u,pressure_7. p_rel)
    annotation (Line(points={{238,134},{164,134},{164,148},{151,148}},
                                                        color={0,0,127}));
  connect(V_flow_7,to_m3hr7. y) annotation (Line(points={{326,196},{141,196}},
                           color={0,0,127}));
  connect(volumeFlow_7.V_flow,to_m3hr7. u)
    annotation (Line(points={{103,192},{103,196},{118,196}}, color={0,0,127}));
  connect(PI_7.y,u_v_7)
    annotation (Line(points={{217,180},{208,180},{208,150},{326,150}},
                                                   color={0,0,127}));
  connect(PI_7.u_m,to_m3hr7. y)
    annotation (Line(points={{228,192},{228,196},{141,196}}, color={0,0,127}));
  connect(fixedDelay7.y,PI_7. u_s)
    annotation (Line(points={{255,178},{255,180},{240,180}}, color={0,0,127}));
  connect(fixedDelay7.u,w_v_7)
    annotation (Line(points={{278,178},{324,178}},           color={0,0,127}));
  connect(teeJunctionIdeal_5.port_3, volumeFlow_7.port_a)
    annotation (Line(points={{92,246},{92,202}}, color={0,127,255}));
  connect(to_bar7.y, p_rel_7)
    annotation (Line(points={{261,134},{326,134}}, color={0,0,127}));
  connect(pressure_8.port_b,valve_8. port_b) annotation (Line(
      points={{152,66},{154,66},{154,78},{102,78},{102,70}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(pressure_8.port_a,valve_8. port_a) annotation (Line(
      points={{152,46},{152,40},{102,40},{102,50}},
      color={0,127,255},
      pattern=LinePattern.Dot));
  connect(to_bar8.u,pressure_8. p_rel)
    annotation (Line(points={{248,42},{174,42},{174,56},{161,56}},
                                                        color={0,0,127}));
  connect(V_flow_8,to_m3hr8. y) annotation (Line(points={{324,104},{151,104}},
                           color={0,0,127}));
  connect(volumeFlow_8.V_flow,to_m3hr8. u)
    annotation (Line(points={{113,100},{113,104},{128,104}}, color={0,0,127}));
  connect(PI_8.y,u_v_8)
    annotation (Line(points={{233,86},{224,86},{224,58},{324,58}},
                                                   color={0,0,127}));
  connect(PI_8.u_m,to_m3hr8. y)
    annotation (Line(points={{244,98},{244,104},{151,104}},  color={0,0,127}));
  connect(fixedDelay8.y,PI_8. u_s)
    annotation (Line(points={{265,86},{256,86}},             color={0,0,127}));
  connect(fixedDelay8.u,w_v_8)
    annotation (Line(points={{288,86},{322,86}},             color={0,0,127}));
  connect(to_bar8.y, p_rel_8)
    annotation (Line(points={{271,42},{324,42}}, color={0,0,127}));
  connect(volumeFlow_8.port_a, valve_8.port_b)
    annotation (Line(points={{102,90},{102,70}}, color={0,127,255}));
  connect(valve_8.port_a, pipe_7.port_a) annotation (Line(points={{102,50},{102,
          40},{103,40},{103,30}}, color={0,127,255}));
  connect(PI_7.y, approxAbs_7.u)
    annotation (Line(points={{217,180},{186,180}}, color={0,0,127}));
  connect(approxAbs_7.y, valve_7.opening) annotation (Line(points={{163,180},{120,
          180},{120,150},{100,150}}, color={0,0,127}));
  connect(PI_8.y, approxAbs_8.u) annotation (Line(points={{233,86},{224,86},{224,
          76},{198,76}}, color={0,0,127}));
  connect(approxAbs_8.y, valve_8.opening) annotation (Line(points={{175,76},{120,
          76},{120,60},{110,60}}, color={0,0,127}));
  connect(pipe_9.port_b, valve_7.port_b)
    annotation (Line(points={{66,130},{92,130},{92,140}}, color={0,127,255}));
  connect(pipe_9.port_a, tank.ports[1])
    annotation (Line(points={{46,130},{28,130},{28,164}}, color={0,127,255}));
  connect(pipe_10.port_a, tank.ports[2])
    annotation (Line(points={{46,106},{32,106},{32,164}}, color={0,127,255}));
  connect(pipe_10.port_b, volumeFlow_8.port_b) annotation (Line(points={{66,106},
          {80,106},{80,120},{102,120},{102,110}}, color={0,127,255}));
 annotation (Placement(transformation(
        extent={{10,-10},{-10,10}},
        rotation=180,
        origin={48,188})),
    Icon(coordinateSystem(preserveAspectRatio=false, extent={{-180,-140},{320,440}})),
    Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-180,-140},{320,
            440}})),
    uses(Modelica(version="4.0.0"), Custom_Pump_V2(version="1")),
    version="1");
end mini_circular_water_network;

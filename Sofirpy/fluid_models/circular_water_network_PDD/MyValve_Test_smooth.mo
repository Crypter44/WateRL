within ;
model MyValve_Test_smooth
  Modelica.Fluid.Sources.FixedBoundary boundary(
    redeclare package Medium = Medium,
    p=0,
    nPorts=1) annotation (Placement(transformation(extent={{-84,2},{-64,22}})));
  replaceable package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater
    constrainedby Modelica.Media.Interfaces.PartialMedium annotation (
      __Dymola_choicesAllMatching=true);
  Modelica.Fluid.Machines.PrescribedPump pump(
    redeclare package Medium = Medium,
    redeclare function flowCharacteristic =
        Modelica.Fluid.Machines.BaseClasses.PumpCharacteristics.quadraticFlow (
          V_flow_nominal={0.0001,0.00038,0.00072}, head_nominal={5.85,5.6,4.7}),
    N_nominal=1,
    use_N_in=true)
    annotation (Placement(transformation(extent={{-28,24},{-8,4}})));

  PressureDrivenDemand_smooth
                        pressureDrivenDemand2_1(
    P0=10000,
    Pf=30000,
    Df=1,
    redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{26,56},{46,76}})));
  Modelica.Blocks.Math.MultiSum multiSum(nu=3)
    annotation (Placement(transformation(extent={{-46,-44},{-34,-32}})));
  Modelica.Blocks.Sources.Pulse pulse(
    amplitude=0.65,
    period=3,
    offset=0)
    annotation (Placement(transformation(extent={{-92,-36},{-72,-16}})));
  Modelica.Blocks.Sources.Pulse pulse1(amplitude=0.3, period=1)
    annotation (Placement(transformation(extent={{-90,-68},{-70,-48}})));
  Modelica.Blocks.Sources.Pulse pulse2(amplitude=0.05, period=0.5)
    annotation (Placement(transformation(extent={{-92,-98},{-72,-78}})));

equation
  connect(pump.port_a, boundary.ports[1]) annotation (Line(points={{-28,14},{-48,
          14},{-48,12},{-64,12}},     color={0,127,255}));
  connect(pump.port_b, pressureDrivenDemand2_1.port_a) annotation (Line(points={{-8,14},
          {10,14},{10,66},{26,66}},           color={0,127,255}));
  connect(pulse.y, multiSum.u[1]) annotation (Line(points={{-71,-26},{-60,-26},{
          -60,-39.4},{-46,-39.4}},  color={0,0,127}));
  connect(pulse1.y, multiSum.u[2]) annotation (Line(points={{-69,-58},{-58,-58},
          {-58,-38},{-46,-38}}, color={0,0,127}));
  connect(pulse2.y, multiSum.u[3]) annotation (Line(points={{-71,-88},{-58,-88},
          {-58,-36.6},{-46,-36.6}}, color={0,0,127}));
  connect(multiSum.y, pump.N_in) annotation (Line(points={{-32.98,-38},{-18,-38},
          {-18,4}},           color={0,0,127}));
  annotation (uses(Modelica(version="4.0.0"), PressureDrivenDemand2(version="1"),
      PressureDrivenDemand_smooth(version="1")),
    version="1",
    conversion(noneFromVersion=""));
end MyValve_Test_smooth;

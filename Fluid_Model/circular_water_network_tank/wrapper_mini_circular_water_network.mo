within ;
model wrapper_mini_circular_water_network
  Modelica.Blocks.Sources.Step valve_6(height=1)
    annotation (Placement(transformation(extent={{-16,-32},{4,-12}})));
  Modelica.Blocks.Sources.Step valve_2(height=1)
    annotation (Placement(transformation(extent={{-88,22},{-68,42}})));
  Modelica.Blocks.Sources.Step valve_3(height=1)
    annotation (Placement(transformation(extent={{-60,48},{-40,68}})));
  Modelica.Blocks.Sources.Step pump_4(height=1)
    annotation (Placement(transformation(extent={{96,74},{76,94}})));
  Modelica.Blocks.Sources.Step valve_5(height=1)
    annotation (Placement(transformation(extent={{96,36},{76,56}})));
  Modelica.Blocks.Sources.Step pump_1(height=1,   startTime=10)
    annotation (Placement(transformation(extent={{-94,-14},{-74,6}})));
  Modelica.Blocks.Math.Add3 add3_1
    annotation (Placement(transformation(extent={{-54,-18},{-34,2}})));
  Modelica.Blocks.Sources.Step pump_2(height=0,    startTime=20)
    annotation (Placement(transformation(extent={{-94,-44},{-74,-24}})));
  Modelica.Blocks.Sources.Step pump_3(height=0,    startTime=30)
    annotation (Placement(transformation(extent={{-94,-78},{-74,-58}})));
  mini_circular_water_network mini_circular_water_network1
    annotation (Placement(transformation(extent={{-6,6},{44,56}})));
  Modelica.Blocks.Sources.Step tank_7(height=1)
    annotation (Placement(transformation(extent={{96,-2},{76,18}})));
  Modelica.Blocks.Sources.Step tank_8(height=1)
    annotation (Placement(transformation(extent={{96,-36},{76,-16}})));
equation
  connect(pump_1.y, add3_1.u1) annotation (Line(points={{-73,-4},{-66,-4},{-66,
          0},{-56,0}}, color={0,0,127}));
  connect(pump_2.y, add3_1.u2) annotation (Line(points={{-73,-34},{-66,-34},{
          -66,-8},{-56,-8}}, color={0,0,127}));
  connect(pump_3.y, add3_1.u3) annotation (Line(points={{-73,-68},{-64,-68},{
          -64,-16},{-56,-16}}, color={0,0,127}));
  connect(add3_1.y, mini_circular_water_network1.w_p_1) annotation (Line(points
        ={{-33,-8},{-16,-8},{-16,15.1379},{-7,15.1379}}, color={0,0,127}));
  connect(valve_6.y, mini_circular_water_network1.w_v_6) annotation (Line(
        points={{5,-22},{28.8,-22},{28.8,5.31034}}, color={0,0,127}));
  connect(pump_4.y, mini_circular_water_network1.w_p_4) annotation (Line(points
        ={{75,84},{26.4,84},{26.4,56.1724}}, color={0,0,127}));
  connect(valve_3.y, mini_circular_water_network1.w_v_3) annotation (Line(
        points={{-39,58},{-14,58},{-14,51.1724},{-6.6,51.1724}}, color={0,0,127}));
  connect(valve_2.y, mini_circular_water_network1.w_v_2) annotation (Line(
        points={{-67,32},{-16,32},{-16,28.7586},{-7.8,28.7586}}, color={0,0,127}));
  connect(valve_5.y, mini_circular_water_network1.w_v_5) annotation (Line(
        points={{75,46},{54,46},{54,39.7931},{44.8,39.7931}}, color={0,0,127}));
  connect(tank_7.y, mini_circular_water_network1.w_v_7) annotation (Line(points
        ={{75,8},{54,8},{54,33.4138},{44.4,33.4138}}, color={0,0,127}));
  connect(tank_8.y, mini_circular_water_network1.w_v_8) annotation (Line(points
        ={{75,-26},{50,-26},{50,26},{48,26},{48,25.4828},{44.2,25.4828}}, color
        ={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    uses(Modelica(version="4.0.0"), MinimalBergischGladbachPI(version="1"),
      mini_circular_water_network(version="1")),
    version="1");
end wrapper_mini_circular_water_network;

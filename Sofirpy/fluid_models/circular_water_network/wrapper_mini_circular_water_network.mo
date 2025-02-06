within ;
model wrapper_mini_circular_water_network
  mini_circular_water_network mini_circular_water_network_test1
    annotation (Placement(transformation(extent={{-22,4},{32,54}})));
  Modelica.Blocks.Sources.Step valve_6(height=3)
    annotation (Placement(transformation(extent={{-16,-32},{4,-12}})));
  Modelica.Blocks.Sources.Step valve_2(height=3)
    annotation (Placement(transformation(extent={{-88,22},{-68,42}})));
  Modelica.Blocks.Sources.Step valve_3(height=1)
    annotation (Placement(transformation(extent={{-60,48},{-40,68}})));
  Modelica.Blocks.Sources.Step pump_4(height=0.8)
    annotation (Placement(transformation(extent={{90,58},{70,78}})));
  Modelica.Blocks.Sources.Step valve_5(height=3)
    annotation (Placement(transformation(extent={{94,20},{74,40}})));
  Modelica.Blocks.Sources.Step pump_1(height=0.8, startTime=10)
    annotation (Placement(transformation(extent={{-94,-14},{-74,6}})));
  Modelica.Blocks.Math.Add3 add3_1
    annotation (Placement(transformation(extent={{-54,-18},{-34,2}})));
  Modelica.Blocks.Sources.Step pump_2(height=-0.3, startTime=20)
    annotation (Placement(transformation(extent={{-94,-44},{-74,-24}})));
  Modelica.Blocks.Sources.Step pump_3(height=0,    startTime=30)
    annotation (Placement(transformation(extent={{-94,-78},{-74,-58}})));
equation
  connect(valve_6.y, mini_circular_water_network_test1.w_v_6) annotation (Line(
        points={{5,-22},{15.584,-22},{15.584,3.13043}}, color={0,0,127}));
  connect(valve_3.y, mini_circular_water_network_test1.w_v_3) annotation (Line(
        points={{-39,58},{-34,58},{-34,49},{-23.728,49}}, color={0,0,127}));
  connect(valve_2.y, mini_circular_water_network_test1.w_v_2) annotation (Line(
        points={{-67,32},{-32,32},{-32,32.6957},{-23.944,32.6957}}, color={0,0,
          127}));
  connect(valve_5.y, mini_circular_water_network_test1.w_v_5) annotation (Line(
        points={{73,30},{42,30},{42,33.7826},{32.432,33.7826}}, color={0,0,127}));
  connect(pump_4.y, mini_circular_water_network_test1.w_p_4) annotation (Line(
        points={{69,68},{12.56,68},{12.56,54.6522}}, color={0,0,127}));
  connect(add3_1.y, mini_circular_water_network_test1.w_p_1) annotation (Line(
        points={{-33,-8},{-28,-8},{-28,6},{-30,6},{-30,14.6522},{-22.864,
          14.6522}}, color={0,0,127}));
  connect(pump_1.y, add3_1.u1) annotation (Line(points={{-73,-4},{-66,-4},{-66,
          0},{-56,0}}, color={0,0,127}));
  connect(pump_2.y, add3_1.u2) annotation (Line(points={{-73,-34},{-66,-34},{
          -66,-8},{-56,-8}}, color={0,0,127}));
  connect(pump_3.y, add3_1.u3) annotation (Line(points={{-73,-68},{-64,-68},{
          -64,-16},{-56,-16}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    uses(Modelica(version="4.0.0"), MinimalBergischGladbachPI(version="1"),
      mini_circular_water_network(version="1")),
    version="1");
end wrapper_mini_circular_water_network;

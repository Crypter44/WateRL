within ;
model wrapper_mini_circular_water_network
  Modelica.Blocks.Sources.Step step_6(height=1)
    annotation (Placement(transformation(extent={{-16,-32},{4,-12}})));
  Modelica.Blocks.Sources.Step pump_1(height=1)
    annotation (Placement(transformation(extent={{-58,0},{-38,20}})));
  Modelica.Blocks.Sources.Step valve_2(height=1)
    annotation (Placement(transformation(extent={{-88,22},{-68,42}})));
  Modelica.Blocks.Sources.Step valve_3(height=1)
    annotation (Placement(transformation(extent={{-60,48},{-40,68}})));
  Modelica.Blocks.Sources.Step pump_4(height=1)
    annotation (Placement(transformation(extent={{90,58},{70,78}})));
  Modelica.Blocks.Sources.Step step5(height=1)
    annotation (Placement(transformation(extent={{94,20},{74,40}})));
  mini_circular_water_network_wo_PI mini_circular_water_network_wo_PI1
    annotation (Placement(transformation(extent={{-6,10},{44,56}})));
equation
  connect(step_6.y, mini_circular_water_network_wo_PI1.w_v_6)
    annotation (Line(points={{5,-22},{30.6,-22},{30.6,9.2}}, color={0,0,127}));
  connect(pump_1.y, mini_circular_water_network_wo_PI1.w_p_1) annotation (Line(
        points={{-37,10},{-16,10},{-16,19.8},{-6.8,19.8}}, color={0,0,127}));
  connect(valve_2.y, mini_circular_water_network_wo_PI1.w_v_2) annotation (Line(
        points={{-67,32},{-18,32},{-18,27.2},{-7,27.2}}, color={0,0,127}));
  connect(valve_3.y, mini_circular_water_network_wo_PI1.w_v_3) annotation (Line(
        points={{-39,58},{-16,58},{-16,39.6},{-7.4,39.6}}, color={0,0,127}));
  connect(pump_4.y, mini_circular_water_network_wo_PI1.w_p_4)
    annotation (Line(points={{69,68},{26,68},{26,56.6}}, color={0,0,127}));
  connect(step5.y, mini_circular_water_network_wo_PI1.w_v_5) annotation (Line(
        points={{73,30},{50,30},{50,37.4},{44.4,37.4}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    uses(Modelica(version="4.0.0"), MinimalBergischGladbachPI(version="1"),
      mini_circular_water_network(version="1"),
      mini_circular_water_network_wo_PI(version="1")),
    version="1");
end wrapper_mini_circular_water_network;

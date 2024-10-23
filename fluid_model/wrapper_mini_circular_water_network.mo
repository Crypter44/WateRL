within ;
model wrapper_mini_circular_water_network
  mini_circular_water_network mini_circular_water_network_test1
    annotation (Placement(transformation(extent={{-22,4},{32,54}})));
  Modelica.Blocks.Sources.Step step
    annotation (Placement(transformation(extent={{-16,-32},{4,-12}})));
  Modelica.Blocks.Sources.Step step1
    annotation (Placement(transformation(extent={{-58,0},{-38,20}})));
  Modelica.Blocks.Sources.Step step2
    annotation (Placement(transformation(extent={{-88,22},{-68,42}})));
  Modelica.Blocks.Sources.Step step3
    annotation (Placement(transformation(extent={{-60,48},{-40,68}})));
  Modelica.Blocks.Sources.Step step4
    annotation (Placement(transformation(extent={{90,58},{70,78}})));
  Modelica.Blocks.Sources.Step step5
    annotation (Placement(transformation(extent={{94,20},{74,40}})));
equation
  connect(step.y, mini_circular_water_network_test1.w_v_6) annotation (Line(
        points={{5,-22},{15.584,-22},{15.584,3.13043}}, color={0,0,127}));
  connect(step3.y, mini_circular_water_network_test1.w_v_3) annotation (Line(
        points={{-39,58},{-34,58},{-34,49},{-23.728,49}}, color={0,0,127}));
  connect(step2.y, mini_circular_water_network_test1.w_v_2) annotation (Line(
        points={{-67,32},{-32,32},{-32,32.6957},{-23.944,32.6957}}, color={0,0,
          127}));
  connect(step1.y, mini_circular_water_network_test1.w_p_1) annotation (Line(
        points={{-37,10},{-32,10},{-32,14.6522},{-22.864,14.6522}}, color={0,0,
          127}));
  connect(step5.y, mini_circular_water_network_test1.w_v_5) annotation (Line(
        points={{73,30},{42,30},{42,33.7826},{32.432,33.7826}}, color={0,0,127}));
  connect(step4.y, mini_circular_water_network_test1.w_p_4) annotation (Line(
        points={{69,68},{12.56,68},{12.56,54.6522}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    uses(Modelica(version="4.0.0"), MinimalBergischGladbachPI(version="1"),
      mini_circular_water_network(version="1")),
    version="1");
end wrapper_mini_circular_water_network;

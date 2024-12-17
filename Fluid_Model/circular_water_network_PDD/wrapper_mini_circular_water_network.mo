within ;
model wrapper_mini_circular_water_network
  Modelica.Blocks.Sources.Step pump_4(height=0.5, startTime=2)
    annotation (Placement(transformation(extent={{90,58},{70,78}})));
  Modelica.Blocks.Sources.Step pump_1(height=0.5, startTime=10)
    annotation (Placement(transformation(extent={{-88,-10},{-68,10}})));
  mini_circular_water_network mini_circular_water_network1
    annotation (Placement(transformation(extent={{-10,8},{40,54}})));
  Modelica.Blocks.Sources.Step valve_3(height=1)
    annotation (Placement(transformation(extent={{-90,64},{-70,84}})));
  Modelica.Blocks.Sources.Step valve_5(height=1)
    annotation (Placement(transformation(extent={{92,24},{72,44}})));
  Modelica.Blocks.Sources.Step valve_6(height=1)
    annotation (Placement(transformation(extent={{92,-16},{72,4}})));
  Modelica.Blocks.Sources.Step valve_2(height=1)
    annotation (Placement(transformation(extent={{-90,28},{-70,48}})));
equation
  connect(pump_4.y, mini_circular_water_network1.w_p_4)
    annotation (Line(points={{69,68},{22,68},{22,54.6}}, color={0,0,127}));
  connect(pump_1.y, mini_circular_water_network1.w_p_1) annotation (Line(points
        ={{-67,0},{-20,0},{-20,17.8},{-10.8,17.8}}, color={0,0,127}));
  connect(valve_3.y, mini_circular_water_network1.w_v_3) annotation (Line(
        points={{-69,74},{-22,74},{-22,37.2},{-11.2,37.2}}, color={0,0,127}));
  connect(valve_5.y, mini_circular_water_network1.w_v_4) annotation (Line(
        points={{71,34},{48,34},{48,34.2},{40.6,34.2}}, color={0,0,127}));
  connect(valve_6.y, mini_circular_water_network1.w_v_6) annotation (Line(
        points={{71,-6},{50,-6},{50,16.2},{40,16.2}}, color={0,0,127}));
  connect(valve_2.y, mini_circular_water_network1.w_v_2) annotation (Line(
        points={{-69,38},{-26,38},{-26,21.6},{-11,21.6}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    uses(Modelica(version="4.0.0"), MinimalBergischGladbachPI(version="1"),
      mini_circular_water_network(version="1")),
    version="1");
end wrapper_mini_circular_water_network;

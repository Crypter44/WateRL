within ;
model wrapper_circular_water_network_tank
  Modelica.Blocks.Sources.Step valve_6(height=0)
    annotation (Placement(transformation(extent={{-16,-32},{4,-12}})));
  Modelica.Blocks.Sources.Step valve_2(height=0)
    annotation (Placement(transformation(extent={{-88,22},{-68,42}})));
  Modelica.Blocks.Sources.Step valve_3(height=0)
    annotation (Placement(transformation(extent={{-60,48},{-40,68}})));
  Modelica.Blocks.Sources.Step pump_4(height=1)
    annotation (Placement(transformation(extent={{96,74},{76,94}})));
  Modelica.Blocks.Sources.Step valve_5(height=0)
    annotation (Placement(transformation(extent={{96,36},{76,56}})));
  Modelica.Blocks.Sources.Step pump_1(height=1,   startTime=10)
    annotation (Placement(transformation(extent={{-94,-14},{-74,6}})));
  Modelica.Blocks.Math.Add3 add3_1
    annotation (Placement(transformation(extent={{-54,-18},{-34,2}})));
  Modelica.Blocks.Sources.Step pump_2(height=0,    startTime=20)
    annotation (Placement(transformation(extent={{-94,-44},{-74,-24}})));
  Modelica.Blocks.Sources.Step pump_3(height=0,    startTime=30)
    annotation (Placement(transformation(extent={{-94,-78},{-74,-58}})));
  Modelica.Blocks.Sources.Step tank_7(height=2)
    annotation (Placement(transformation(extent={{96,-2},{76,18}})));
  Modelica.Blocks.Sources.Step tank_8(height=2)
    annotation (Placement(transformation(extent={{96,-38},{76,-18}})));
  circular_water_network_tank circular_water_network_tank1
    annotation (Placement(transformation(extent={{-6,6},{44,64}})));
equation
  connect(pump_1.y, add3_1.u1) annotation (Line(points={{-73,-4},{-66,-4},{-66,
          0},{-56,0}}, color={0,0,127}));
  connect(pump_2.y, add3_1.u2) annotation (Line(points={{-73,-34},{-66,-34},{
          -66,-8},{-56,-8}}, color={0,0,127}));
  connect(pump_3.y, add3_1.u3) annotation (Line(points={{-73,-68},{-64,-68},{
          -64,-16},{-56,-16}}, color={0,0,127}));
  connect(valve_6.y, circular_water_network_tank1.w_v_6)
    annotation (Line(points={{5,-22},{28.8,-22},{28.8,5.2}}, color={0,0,127}));
  connect(add3_1.y, circular_water_network_tank1.w_p_1) annotation (Line(points
        ={{-33,-8},{-16,-8},{-16,16.6},{-7,16.6}}, color={0,0,127}));
  connect(valve_2.y, circular_water_network_tank1.w_v_2) annotation (Line(
        points={{-67,32},{-66,32.4},{-7.8,32.4}}, color={0,0,127}));
  connect(pump_4.y, circular_water_network_tank1.w_p_4)
    annotation (Line(points={{75,84},{26.4,84},{26.4,64.2}}, color={0,0,127}));
  connect(valve_5.y, circular_water_network_tank1.w_v_5)
    annotation (Line(points={{75,46},{74,45.2},{44.8,45.2}}, color={0,0,127}));
  connect(tank_8.y, circular_water_network_tank1.w_v_8) annotation (Line(points
        ={{75,-28},{50,-28},{50,20},{52,20},{52,28.6},{44.2,28.6}}, color={0,0,
          127}));
  connect(valve_3.y, circular_water_network_tank1.w_v_3) annotation (Line(
        points={{-39,58},{-38,58.4},{-6.6,58.4}}, color={0,0,127}));
  connect(tank_7.y, circular_water_network_tank1.w_v_7) annotation (Line(points
        ={{75,8},{54,8},{54,37.8},{44.4,37.8}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    uses(Modelica(version="4.0.0"), MinimalBergischGladbachPI(version="1"),
      circular_water_network_tank(version="1")),
    version="1");
end wrapper_circular_water_network_tank;

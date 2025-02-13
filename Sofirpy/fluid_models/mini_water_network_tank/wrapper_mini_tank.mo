within ;
model wrapper_mini_tank
  Modelica.Blocks.Sources.Step pump(height=1)
    annotation (Placement(transformation(extent={{52,-14},{32,6}})));
  Modelica.Blocks.Sources.Step valve(height=1)
    annotation (Placement(transformation(extent={{80,60},{60,80}})));
  Modelica.Blocks.Sources.Step tank1(height=-0.25, startTime=2000)
    annotation (Placement(transformation(extent={{126,38},{106,58}})));
  mini_tank mini_tank1
    annotation (Placement(transformation(extent={{-44,-48},{6,22}})));
  Modelica.Blocks.Math.Add add2 annotation (Placement(transformation(
        extent={{10,10},{-10,-10}},
        rotation=0,
        origin={26,38})));
  Modelica.Blocks.Sources.Step tank2(height=0.25,  startTime=20000)
    annotation (Placement(transformation(extent={{126,0},{106,20}})));
  Modelica.Blocks.Math.Add add1 annotation (Placement(transformation(
        extent={{10,10},{-10,-10}},
        rotation=0,
        origin={68,28})));
  Modelica.Blocks.Sources.Step tank3(height=1, startTime=0)
    annotation (Placement(transformation(extent={{78,-42},{58,-22}})));
equation
  connect(add1.y, add2.u1)
    annotation (Line(points={{57,28},{46,28},{46,24},{38,24},{38,32}},
                                                 color={0,0,127}));
  connect(valve.y, add2.u2)
    annotation (Line(points={{59,70},{38,70},{38,44}}, color={0,0,127}));
  connect(tank2.y, add1.u1)
    annotation (Line(points={{105,10},{80,10},{80,22}},   color={0,0,127}));
  connect(tank1.y, add1.u2)
    annotation (Line(points={{105,48},{80,48},{80,34}},
                                                      color={0,0,127}));
  connect(tank3.y, mini_tank1.w_v_7) annotation (Line(points={{57,-32},{16,-32},
          {16,-31.0526},{8.14286,-31.0526}}, color={0,0,127}));
  connect(pump.y, mini_tank1.w_v_5) annotation (Line(points={{31,-4},{18,-4},{
          18,-10.4211},{7.19048,-10.4211}}, color={0,0,127}));
  connect(add2.y, mini_tank1.w_p_4) annotation (Line(points={{15,38},{-14.9524,
          38},{-14.9524,23.1053}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    uses(mini_tank(version="1"), Modelica(version="4.0.0"),
      mini_tank_abs(version="1"),
      mini_tank_abs2(version="1")));
end wrapper_mini_tank;

within ;
model wrapper_mini_tank
  Modelica.Blocks.Sources.Step pump(height=1)
    annotation (Placement(transformation(extent={{-84,74},{-64,94}})));
  Modelica.Blocks.Sources.Step valve(height=0)
    annotation (Placement(transformation(extent={{54,28},{34,48}})));
  Modelica.Blocks.Sources.Step tank1(height=0,    startTime=30)
    annotation (Placement(transformation(extent={{54,-30},{34,-10}})));
  Modelica.Blocks.Math.Add add1 annotation (Placement(transformation(
        extent={{10,10},{-10,-10}},
        rotation=90,
        origin={-22,56})));
  Modelica.Blocks.Sources.Step pump1(height=0,    startTime=30)
    annotation (Placement(transformation(extent={{-6,78},{14,98}})));
  mini_tank mini_tank1
    annotation (Placement(transformation(extent={{-44,-48},{6,22}})));
equation
  connect(add1.u2, pump.y)
    annotation (Line(points={{-28,68},{-28,84},{-63,84}}, color={0,0,127}));
  connect(pump1.y, add1.u1) annotation (Line(points={{15,88},{24,88},{24,72},{
          -6,72},{-6,68},{-16,68}}, color={0,0,127}));
  connect(add1.y, mini_tank1.w_p_4) annotation (Line(points={{-22,45},{-22,30},
          {-11.6,30},{-11.6,22.875}},
                                    color={0,0,127}));
  connect(valve.y, mini_tank1.w_v_5) annotation (Line(points={{33,38},{16,38},{
          16,-5.41667},{6.8,-5.41667}},
                              color={0,0,127}));
  connect(tank1.y, mini_tank1.w_v_7)
    annotation (Line(points={{33,-20},{7.8,-20}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    uses(mini_tank(version="1"), Modelica(version="4.0.0"),
      mini_tank_abs(version="1"),
      mini_tank_abs2(version="1")));
end wrapper_mini_tank;

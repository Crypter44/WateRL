within ;
model TestCase
  Modelica.Blocks.Logical.GreaterEqualThreshold greaterEqualThreshold(threshold
      =0.01) annotation (Placement(transformation(extent={{4,-4},{24,16}})));
  Modelica.Blocks.Interfaces.BooleanOutput y1
                                    "Connector of Boolean output signal"
    annotation (Placement(transformation(extent={{104,-4},{124,16}})));
  Modelica.Blocks.Nonlinear.FixedDelay fixedDelay(delayTime=0.01)
    annotation (Placement(transformation(extent={{-54,-4},{-34,16}})));
  Modelica.Blocks.Interfaces.RealInput u1
              "Connector of Real input signal"
    annotation (Placement(transformation(extent={{-140,-14},{-100,26}})));
equation
  connect(greaterEqualThreshold.y, y1) annotation (Line(points={{25,6},{114,6}},
                          color={255,0,255}));
  connect(fixedDelay.y, greaterEqualThreshold.u)
    annotation (Line(points={{-33,6},{2,6}}, color={0,0,127}));
  connect(fixedDelay.u, u1)
    annotation (Line(points={{-56,6},{-120,6}}, color={0,0,127}));
  annotation (uses(Modelica(version="4.0.0")));
end TestCase;

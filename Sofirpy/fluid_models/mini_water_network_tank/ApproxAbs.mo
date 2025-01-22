within ;
block ApproxAbs "Approximation for abs"
  Modelica.Blocks.Interfaces.RealInput u "Connector of Real input signal"
    annotation (Placement(transformation(extent={{-140,-20},{-100,20}}),
        iconTransformation(extent={{-140,-20},{-100,20}})));
  Modelica.Blocks.Math.Sqrt sqrt1
    annotation (Placement(transformation(extent={{66,-10},{86,10}})));
  Modelica.Blocks.Math.Product product1
    annotation (Placement(transformation(extent={{-34,-10},{-14,10}})));
  Modelica.Blocks.Math.Add add
    annotation (Placement(transformation(extent={{22,-10},{42,10}})));
  Modelica.Blocks.Math.Gain gain(k=4)
    annotation (Placement(transformation(extent={{-14,-60},{6,-40}})));
  Modelica.Blocks.Math.Product product2
    annotation (Placement(transformation(extent={{-48,-60},{-28,-40}})));
  Modelica.Blocks.Sources.RealExpression realExpression(y=0.1)
    annotation (Placement(transformation(extent={{-88,-62},{-68,-42}})));
  Modelica.Blocks.Interfaces.RealOutput y "Connector of Real output signal"
    annotation (Placement(transformation(extent={{100,-10},{120,10}}),
        iconTransformation(extent={{100,-10},{120,10}})));
equation
  connect(product1.u1, u) annotation (Line(points={{-36,6},{-80,6},{-80,0},{
          -120,0}}, color={0,0,127}));
  connect(product1.u2, u) annotation (Line(points={{-36,-6},{-80,-6},{-80,0},{
          -120,0}}, color={0,0,127}));
  connect(add.u1, product1.y)
    annotation (Line(points={{20,6},{4,6},{4,0},{-13,0}}, color={0,0,127}));
  connect(gain.y, add.u2) annotation (Line(points={{7,-50},{12,-50},{12,-6},{20,
          -6}}, color={0,0,127}));
  connect(product2.y, gain.u)
    annotation (Line(points={{-27,-50},{-16,-50}}, color={0,0,127}));
  connect(realExpression.y, product2.u1) annotation (Line(points={{-67,-52},{
          -60,-52},{-60,-44},{-50,-44}}, color={0,0,127}));
  connect(product2.u2, realExpression.y) annotation (Line(points={{-50,-56},{
          -60,-56},{-60,-52},{-67,-52}}, color={0,0,127}));
  connect(add.y, sqrt1.u)
    annotation (Line(points={{43,0},{64,0}}, color={0,0,127}));
  connect(sqrt1.y, y)
    annotation (Line(points={{87,0},{110,0}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false), graphics={Text(
          extent={{-64,40},{64,-34}},
          textColor={0,0,0},
          textString="|u|"), Rectangle(extent={{-100,100},{100,-100}},
            lineColor={28,108,200})}),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    uses(Modelica(version="4.0.0")));
end ApproxAbs;

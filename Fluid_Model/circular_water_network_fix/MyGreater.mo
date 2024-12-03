within ;
block MyGreater
  "Output y is true if input u1 is greater than input u2"
  Modelica.Blocks.Interfaces.RealInput u1
    annotation (Placement(transformation(extent={{-140,44},{-100,84}}),
        iconTransformation(extent={{-140,44},{-100,84}})));
  Modelica.Blocks.Interfaces.RealInput u2
    annotation (Placement(transformation(extent={{-140,-84},{-100,-44}}),
        iconTransformation(extent={{-140,-84},{-100,-44}})));
  Modelica.Blocks.Interfaces.BooleanOutput
                                        y
    annotation (Placement(transformation(extent={{100,-2},{120,18}}),
        iconTransformation(extent={{100,-2},{120,18}})));
equation
  if u1 > u2 then
    y = true;
  else
    y = false;
  end if;
   annotation (
    Icon(coordinateSystem(preserveAspectRatio=false), graphics={Text(
          extent={{-44,50},{52,-54}},
          textColor={0,0,0},
          textString=">"), Rectangle(extent={{-100,100},{100,-100}}, lineColor=
              {28,108,200})}),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    uses(Modelica(version="4.0.0")));
end MyGreater;

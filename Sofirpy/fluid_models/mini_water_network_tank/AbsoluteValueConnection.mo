within ;
model AbsoluteValueConnection
  Modelica.Blocks.Interfaces.RealOutput y
    annotation (Placement(transformation(extent={{-100,-10},{-120,10}})));
  Modelica.Blocks.Interfaces.RealInput u
    annotation (Placement(transformation(extent={{140,-20},{100,20}})));
  Real absValue;
equation
  absValue = if u >= 0 then u else -u;  // Calculate absolute value
  y = absValue;  // Assign the result to the output
  annotation (uses(Modelica(version="4.0.0")));
end AbsoluteValueConnection;

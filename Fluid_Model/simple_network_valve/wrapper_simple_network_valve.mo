within ;
model wrapper_simple_network_valve "wrapper"
  simple_network_valve simple_network_valve1
    annotation (Placement(transformation(extent={{0,-8},{20,14}})));
  Modelica.Blocks.Sources.Step step
    annotation (Placement(transformation(extent={{-52,4},{-32,24}})));
equation
  connect(step.y, simple_network_valve1.w_v_2) annotation (Line(points={{-31,14},
          {-10,14},{-10,2.4},{-0.8,2.4}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    uses(simple_network_valve(version="1"), Modelica(version="4.0.0")));
end wrapper_simple_network_valve;

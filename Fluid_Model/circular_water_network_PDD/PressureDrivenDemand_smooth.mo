within ;
model PressureDrivenDemand_smooth

  parameter Modelica.Units.SI.Pressure P0=1e4 "Minimum pressure for demand";
  parameter Modelica.Units.SI.Pressure Pf=5e4
    "Pressure at which maximum demand is reached";

  replaceable package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater
    constrainedby Modelica.Media.Interfaces.PartialMedium
      "Medium model within the source"
     annotation (choicesAllMatching=true);
  Modelica.Fluid.Interfaces.FluidPort_a port_a(redeclare package Medium = Medium)
    annotation (Placement(transformation(extent={{-110,-10},{-90,10}})));
  Modelica.Units.SI.Density rho "Debsity at the fluid port";

  Modelica.Units.SI.MassFlowRate Df
    "Demand, e.g. the amount of volume flow leaving the system when 
    the required pressure pf is reached";

  Modelica.Units.SI.MassFlowRate m_flow
    "Mass flow rate through the demand node";

  Modelica.Blocks.Interfaces.RealInput  Qf
    "Maximum demand (full flow rate)" annotation (Placement(transformation(
        extent={{-20,-20},{20,20}},
        rotation=90,
        origin={0,-108})));
protected
  Modelica.Units.SI.Pressure p "Pressure at the fluid port";

equation
  // Pressure at the port
  rho = Medium.density(Medium.setState_phX(port_a.p, port_a.h_outflow, port_a.Xi_outflow));
  Df = Qf * rho;
  p = port_a.p;

  if p <= Pf then
    m_flow = smooth(0, if p<=P0 then 0 else Df * sqrt((p - P0) / (Pf - P0)));
  else
    m_flow = Df;
  end if;

  // Flow conservation
  port_a.m_flow = m_flow;

  // Energy conservation (isothermal assumption)
  port_a.h_outflow = inStream(port_a.h_outflow);

  annotation (
    Icon(graphics={
      Rectangle(extent={{-100,100},{100,-100}},
                                            lineColor={28,108,200}),
        Polygon(
          points={{-80,-4},{0,96},{80,-4},{20,16},{20,-44},{-20,-44},{-20,16},{-80,
              -4}},
          lineColor={28,108,200},
          fillColor={170,213,255},
          fillPattern=FillPattern.Solid,
          lineThickness=0.5),
      Text(extent={{-66,-32},{74,-112}},                            lineColor={0,0,0},
          textString="PDD"),
        Text(
          extent={{-100,130},{100,110}},
          textColor={28,108,200},
          textString="%name")}),
    Documentation(info="
      <html>
        <p>This block represents a pressure-driven demand node. It calculates the mass flow rate
        based on the available pressure at the port and applies a capped demand behavior when
        pressure exceeds the threshold Pf.</p>
        <p>Flow rate is determined using the following piecewise function:</p>
        <code>d = 0, if p <= P0</code><br>
        <code>d = Df * sqrt((p - P0) / (Pf - P0)), if P0 < p <= Pf</code><br>
        <code>d = Df, if p > Pf</code>
      </html>"),
    uses(Modelica(version="4.0.0")),
    version="1",
    conversion(noneFromVersion=""));
end PressureDrivenDemand_smooth;

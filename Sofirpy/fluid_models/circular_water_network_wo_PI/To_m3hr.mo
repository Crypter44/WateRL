within ;
block To_m3hr "From m^3/s to m^3/hour"
 extends Modelica.Blocks.Interfaces.PartialConversionBlock(u(unit="m3/s"), y(
        unit="m3/h"));
equation
  y = u*60*60
  annotation (Icon(coordinateSystem(preserveAspectRatio=true, extent={{-100,
            -100},{100,100}}), graphics={Text(
              extent={{-20,100},{-100,20}},
              textString="m3/s"),Text(
              extent={{100,-20},{20,-100}},
              textString="m3/h")}), Documentation(info="<html>
<p>
This block converts the input signal from m^3/s to m^3/hour and returns
the result as output signal.
</p>
</html>"));
  annotation (Icon(graphics={
        Text(
          extent={{-72,72},{-78,72}},
          textColor={28,108,200},
          textString="m3/s"),
        Text(
          extent={{-96,120},{28,18}},
          textColor={0,0,0},
          textString="m3/s"),
        Text(
          extent={{-32,-20},{92,-122}},
          textColor={0,0,0},
          textString="m3/h")}));
end To_m3hr;

within ;
block To_m3s
   extends Modelica.Blocks.Interfaces.PartialConversionBlock(u(unit="m3/h"), y(
        unit="m3/s"));
equation
  y = u/60/60
  annotation (Icon(coordinateSystem(preserveAspectRatio=true, extent={{-100,
            -100},{100,100}}), graphics={Text(
              extent={{-20,100},{-100,20}},
              textString="m3/s"),Text(
              extent={{100,-20},{20,-100}},
              textString="m3/h")}), Documentation(info="<html>
<p>
This block converts the input signal from m^3/hour to m^3/s and returns
the result as output signal.
</p>
</html>"));
  annotation (Icon(graphics={
        Text(
          extent={{-72,72},{-78,72}},
          textColor={28,108,200},
          textString="m3/h"),
        Text(
          extent={{-96,120},{28,18}},
          textColor={0,0,0},
          textString="m3/h"),
        Text(
          extent={{-32,-20},{92,-122}},
          textColor={0,0,0},
          textString="m3/s")}));
end To_m3s;

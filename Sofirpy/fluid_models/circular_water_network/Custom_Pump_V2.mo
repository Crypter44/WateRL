within ;
package Custom_Pump_V2

  package BaseClasses_Custom
    "Base classes used in the Machines package (only of interest to build new component models)"
    extends Modelica.Icons.BasesPackage;

  model Pump_cs "Centrifugal pump with constant speed"
      import         Modelica.Units.NonSI;
      import Modelica.Constants;

    extends Custom_Pump_Partial(N(start=rpm_rel));

      parameter Real rpm_rel
        "Rotational speed (relative)"
      annotation(Dialog);

  equation
    N = rpm_rel;

    connect(heatTransfer.heatPorts[1], heatPort) annotation (Line(
        points={{40,-34},{40,-60}},
        color={127,0,0}));
    annotation (
      Icon(coordinateSystem(preserveAspectRatio=true,  extent={{-100,-100},{100,
                100}}), graphics={
            Rectangle(
              extent={{-100,46},{100,-46}},
              lineColor={0,0,0},
              fillColor={0,127,255},
              fillPattern=FillPattern.HorizontalCylinder),
            Polygon(
              points={{-48,-60},{-72,-100},{72,-100},{48,-60},{-48,-60}},
              lineColor={0,0,255},
              pattern=LinePattern.None,
              fillColor={0,0,0},
              fillPattern=FillPattern.VerticalCylinder),
            Ellipse(
              extent={{-80,80},{80,-80}},
              lineColor={0,0,0},
              fillPattern=FillPattern.Sphere,
              fillColor={0,100,199}),
            Polygon(
              points={{-28,30},{-28,-30},{50,-2},{-28,30}},
              lineColor={0,0,0},
              pattern=LinePattern.None,
              fillPattern=FillPattern.HorizontalCylinder,
              fillColor={255,255,255})}),
      Documentation(info="<html>
<p>This is the base model for pumps.
<p>The model describes a centrifugal pump, or a group of <code>nParallel</code> identical pumps. The pump model is based on the theory of kinematic similarity: the pump characteristics are given for nominal operating conditions (rotational speed and fluid density), and then adapted to actual operating condition, according to the similarity equations.

<p><b>Pump characteristics</b></p>
<p> The nominal hydraulic characteristic (head vs. volume flow rate) is given by the the replaceable function <code>flowCharacteristic</code>.
<p> The pump energy balance can be specified in two alternative ways:
<ul>
<li><code>use_powerCharacteristic = false</code> (default option): the replaceable function <code>efficiencyCharacteristic</code> (efficiency vs. volume flow rate in nominal conditions) is used to determine the efficiency, and then the power consumption.
    The default is a constant efficiency of 0.8.</li>
<li><code>use_powerCharacteristic = true</code>: the replaceable function <code>powerCharacteristic</code> (power consumption vs. volume flow rate in nominal conditions) is used to determine the power consumption, and then the efficiency.
    Use <code>powerCharacteristic</code> to specify a non-zero power consumption for zero flow rate.
</ul>
<p>
Several functions are provided in the package <code>PumpCharacteristics</code> to specify the characteristics as a function of some operating points at nominal conditions.
<p>Depending on the value of the <code>checkValve</code> parameter, the model either supports reverse flow conditions, or includes a built-in check valve to avoid flow reversal.
</p>
<p>It is possible to take into account the mass and energy storage of the fluid inside the pump by specifying its volume <code>V</code>, and by selecting appropriate dynamic mass and energy balance assumptions (see below);
this is recommended to avoid singularities in the computation of the outlet enthalpy in case of zero flow rate.
If zero flow rate conditions are always avoided, this dynamic effect can be neglected by leaving the default value <code>V = 0</code>, thus avoiding fast state variables in the model.
</p>

<p><b>Dynamics options</b></p>
<p>
Steady-state mass and energy balances are assumed per default, neglecting the holdup of fluid in the pump; this configuration works well if the flow rate is always positive.
Dynamic mass and energy balance can be used by setting the corresponding dynamic parameters. This is recommended to avoid singularities at zero or reversing mass flow rate. If the initial conditions imply non-zero mass flow rate, it is possible to use the <code>SteadyStateInitial</code> condition, otherwise it is recommended to use <code>FixedInitial</code> in order to avoid undetermined initial conditions.
</p>

<p><b>Heat transfer</b></p>
<p>
The Boolean parameter <code>use_HeatTransfer</code> can be set to true if heat exchanged with the environment
should be taken into account or to model a housing. This might be desirable if a pump with realistic
<code>powerCharacteristic</code> for zero flow operates while a valve prevents fluid flow.
</p>

<p><b>Diagnostics of Cavitation</b></p>
<p>The replaceable Monitoring submodel can be configured to PumpMonitoringNPSH,
in order to compute the Net Positive Suction Head available and check for cavitation,
provided a two-phase medium model is used (see Advanced tab).
</p>
</html>",
        revisions="<html>
<ul>
<li><i>8 Jan 2013</i>
    by R&uuml;diger Franke:<br>
    moved NPSH diagnostics from PartialPump to replaceable sub-model PumpMonitoring.PumpMonitoringNPSH (see ticket #646)</li>
<li><i>Dec 2008</i>
    by R&uuml;diger Franke:<br>
    <ul>
    <li>Replaced simplified mass and energy balances with rigorous formulation (base class PartialLumpedVolume)</li>
    <li>Introduced optional HeatTransfer model defining Qb_flow</li>
    <li>Enabled events when the checkValve is operating to support the opening of a discrete valve before port_a</li>
    </ul></li>
<li><i>31 Oct 2005</i>
    by <a href=\"mailto:francesco.casella@polimi.it\">Francesco Casella</a>:<br>
       Model added to the Fluid library</li>
</ul>
</html>"));
  end Pump_cs;

  package PumpCharacteristics "Functions for pump characteristics"
    extends Modelica.Icons.Package;
    import         Modelica.Units.NonSI;

    partial function baseFlow "Base class for pump flow characteristics"
      extends Modelica.Icons.Function;
        input Modelica.Units.SI.VolumeFlowRate V_flow "Volumetric flow rate";
      input Real rpm_rel "Rotational speed (relative n/n_max)";
        output Modelica.Units.SI.Height head "Pump head";
    end baseFlow;

    partial function basePower
        "Base class for pump power consumption characteristics"
      extends Modelica.Icons.Function;
        input Modelica.Units.SI.VolumeFlowRate V_flow "Volumetric flow rate";
      input Real rpm_rel "Rotational speed (relative n/n_max)";
        output Modelica.Units.SI.Power consumption "Power consumption";
    end basePower;

    function quadraticFlow
        "Quadratic flow characteristic for both relative rpm and flow as independent variables"
      extends baseFlow;
      input Real c[3]
        "Coefficients of quadratic head curve"
        annotation (Dialog);

    algorithm

        // Flow equation
        head := c[1]*V_flow^2 + c[2]*V_flow*rpm_rel + c[3]*rpm_rel^2;

      annotation(Documentation(revisions="<html>
<ul>
<li><i>Apr 2019</i>
by Tobias Meck:<br>
Vorlage angepasst auf multiple Regression</li>
</ul>
</html>"));
    end quadraticFlow;

    function cubicPower "Cubic power consumption characteristic for both relative rpm and flow as independent variables"
      extends basePower;
      input Real c[5]
        "Coefficients of cubic power curve"
        annotation (Dialog);
    algorithm
      consumption := c[1]*V_flow^3 + c[2]*V_flow^2*rpm_rel + c[3]*V_flow*rpm_rel^2 + c[4]*rpm_rel^3 + c[5];
    end cubicPower;

    function Flow_Standard
      "Quadratic flow characteristic, including linear extrapolation"
      extends baseFlow2;
      input Real c[3] "Koeffizienten für Nominaldrehzahl"          annotation(Dialog);

    algorithm

        head := c[3] + V_flow*(c[2] + V_flow*c[1]);

      annotation(Documentation(revisions="<html>
<ul>
<li><i>Jan 2013</i>
    by R&uuml;diger Franke:<br>
    Extended with linear extrapolation outside specified points</li>
</ul>
</html>"));
    end Flow_Standard;

    partial function baseFlow2 "Base class for pump flow characteristics"
      extends Modelica.Icons.Function;
        input Modelica.Units.SI.VolumeFlowRate V_flow "Volumetric flow rate";
        output Modelica.Units.SI.Height head "Pump head";
    end baseFlow2;

    partial function basePower2
      "Base class for pump power consumption characteristics"
      extends Modelica.Icons.Function;
        input Modelica.Units.SI.VolumeFlowRate V_flow "Volumetric flow rate";
        output Modelica.Units.SI.Power consumption "Power consumption";
    end basePower2;

    function Power_Standard "Quadratic power consumption characteristic"
      extends basePower2;
      input Real c[4]
          "Regressionskoeffizienten"                                annotation(Dialog);

    algorithm
      consumption := c[4] + V_flow*c[3] + V_flow^2*c[2] + V_flow^3*c[1];
    end Power_Standard;
  end PumpCharacteristics;

    package PumpMonitoring "Monitoring of pump operation"
      extends Modelica.Icons.Package;
      model PumpMonitoringBase "Interface for pump monitoring"
        outer Modelica.Fluid.System system "System wide properties";
        //
        // Internal interface
        // (not exposed to GUI; needs to be hard coded when using this model
        //
        replaceable package Medium =
          Modelica.Media.Interfaces.PartialMedium "Medium in the component"
            annotation(Dialog(tab="Internal Interface",enable=false));

        // Inputs
        input Medium.ThermodynamicState state_in
          "Thermodynamic state of inflow";
        input Medium.ThermodynamicState state "Thermodynamic state in the pump";

      end PumpMonitoringBase;

      model PumpMonitoringNPSH "Monitor Net Positive Suction Head (NPSH)"
        extends PumpMonitoringBase(redeclare replaceable package Medium =
          Modelica.Media.Interfaces.PartialTwoPhaseMedium);
        Medium.Density rho_in = Medium.density(state_in)
          "Liquid density at the inlet port_a";
        Modelica.Units.SI.Length NPSHa=NPSPa/(rho_in*system.g)
          "Net Positive Suction Head available";
        Modelica.Units.SI.Pressure NPSPa=assertPositiveDifference(
            Medium.pressure(state_in), Medium.saturationPressure(
            Medium.temperature(state_in)),
            "Cavitation occurs at the pump inlet")
          "Net Positive Suction Pressure available";
        Modelica.Units.SI.Pressure NPDPa=assertPositiveDifference(
            Medium.pressure(state), Medium.saturationPressure(
            Medium.temperature(state)), "Cavitation occurs in the pump")
          "Net Positive Discharge Pressure available";
      end PumpMonitoringNPSH;

      function assertPositiveDifference
        extends Modelica.Icons.Function;
        input Modelica.Units.SI.Pressure p;
        input Modelica.Units.SI.Pressure p_sat;
        input String message;
        output Modelica.Units.SI.Pressure dp;
      algorithm
        dp := p - p_sat;
        assert(p >= p_sat, message);
      end assertPositiveDifference;
    end PumpMonitoring;

  model Pump_vs "Centrifugal pump with variable speed"
    extends Custom_Pump_Partial;

    parameter Boolean use_N_in = false
     "Get the relative rotational speed from the input connector";

     parameter Real rpm_rel=1
     "Rotational speed (relative)"
     annotation(Dialog(enable = not use_N_in));

  Modelica.Blocks.Interfaces.RealInput N_in if use_N_in
      "Prescribed rotational speed"
      annotation (Placement(transformation(
          extent={{-20,-20},{20,20}},
          rotation=-90,
          origin={0,100}), iconTransformation(
          extent={{-20,-20},{20,20}},
          rotation=-90,
          origin={0,100})));

    protected
    Modelica.Blocks.Interfaces.RealInput N_in_internal
      "Needed to connect to conditional connector";
  equation
    // Connect statement active only if use_p_in = true
    connect(N_in, N_in_internal);
    // Internal connector value when use_p_in = false
    if not use_N_in then
      N_in_internal = rpm_rel;
    end if;
    // Set N with a lower limit to avoid singularities at zero speed
    N = max(N_in_internal,1e-3) "Rotational speed";

  end Pump_vs;

  partial model Custom_Pump_Partial "Partial Model"
      import         Modelica.Units.NonSI;
      import Modelica.Constants;

    extends Modelica.Fluid.Interfaces.PartialTwoPort(
      port_b_exposesState = energyDynamics<>Modelica.Fluid.Types.Dynamics.SteadyState
                                                                       or massDynamics<>Modelica.Fluid.Types.Dynamics.SteadyState,
      port_a(
        p(start=p_a_start),
        m_flow(start = m_flow_start,
               min = if allowFlowReversal and not checkValve then -Constants.inf else 0)),
      port_b(
        p(start=p_b_start),
        m_flow(start = -m_flow_start,
               max = if allowFlowReversal and not checkValve then +Constants.inf else 0)));

    // Initialization
    parameter Medium.AbsolutePressure p_a_start=system.p_start
        "Guess value for inlet pressure"
      annotation(Dialog(tab="Initialization"));
    parameter Medium.AbsolutePressure p_b_start=p_a_start
        "Guess value for outlet pressure"
      annotation(Dialog(tab="Initialization"));
    parameter Medium.MassFlowRate m_flow_start = system.m_flow_start
        "Guess value of m_flow = port_a.m_flow"
      annotation(Dialog(tab = "Initialization"));

      final parameter Modelica.Units.SI.VolumeFlowRate V_flow_single_init=
          m_flow_start/rho_nominal "Used for simplified initialization model";
      final parameter Modelica.Units.SI.Height delta_head_init=
          flowCharacteristic(V_flow_single_init, 1) - flowCharacteristic(0, 1)
        "Used for simplified initialization model";

    // Characteristic curves

    replaceable function flowCharacteristic =
        Custom_Pump_V2.BaseClasses_Custom.PumpCharacteristics.quadraticFlow                        "Head vs. V_flow characteristic"
      annotation(Dialog(group="Characteristics"), choicesAllMatching=true);

    replaceable function powerCharacteristic =
          Custom_Pump_V2.BaseClasses_Custom.PumpCharacteristics.cubicPower
        "Power consumption vs. V_flow at nominal speed and density"
        annotation(Dialog(group="Characteristics"), choicesAllMatching=true);

   final parameter Medium.Density rho_nominal = Medium.density_pTX(Medium.p_default, Medium.T_default, Medium.X_default);
     //   "Nominal fluid density for characteristic"
   //   annotation(Dialog(group="Characteristics"));

    // Assumptions
    parameter Boolean checkValve=false "= true to prevent reverse flow"
      annotation(Dialog(tab="Assumptions"), Evaluate=true);

      parameter Modelica.Units.SI.Volume V=0 "Volume inside the pump"
        annotation (Dialog(tab="Assumptions"), Evaluate=true);

    // Energy and mass balance
    extends Modelica.Fluid.Interfaces.PartialLumpedVolume(
        final fluidVolume = V,
        energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
        massDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
        final p_start = p_b_start);                                // change by Katharina Henn, 13.09.24

    // Heat transfer through boundary, e.g., to add a housing
    parameter Boolean use_HeatTransfer = false
        "= true to use a HeatTransfer model, e.g., for a housing"
        annotation (Dialog(tab="Assumptions",group="Heat transfer"));
    replaceable model HeatTransfer =
        Modelica.Fluid.Vessels.BaseClasses.HeatTransfer.IdealHeatTransfer
      constrainedby
        Modelica.Fluid.Vessels.BaseClasses.HeatTransfer.PartialVesselHeatTransfer
        "Wall heat transfer"
        annotation (Dialog(tab="Assumptions",group="Heat transfer",enable=use_HeatTransfer),choicesAllMatching=true);
    HeatTransfer heatTransfer(
      redeclare final package Medium = Medium,
      final n=1,
      surfaceAreas={4*Modelica.Constants.pi*(3/4*V/Modelica.Constants.pi)^(2/3)},
      final states = {medium.state},
      final use_k = use_HeatTransfer)
        annotation (Placement(transformation(
          extent={{-10,-10},{30,30}},
          rotation=180,
          origin={50,-10})));
    Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_a heatPort if use_HeatTransfer
      annotation (Placement(transformation(extent={{30,-70},{50,-50}})));

    // Variables
      final parameter Modelica.Units.SI.Acceleration g=system.g;
    Medium.Density rho = medium.d;
      Modelica.Units.SI.Pressure dp_pump=port_b.p - port_a.p
        "Pressure increase";
      Modelica.Units.SI.Height head=dp_pump/(rho*g) "Pump head";
      Modelica.Units.SI.MassFlowRate m_flow=port_a.m_flow
        "Mass flow rate (total)";
      Modelica.Units.SI.MassFlowRate m_flow_single=m_flow
        "Mass flow rate (single pump)";
      Modelica.Units.SI.VolumeFlowRate V_flow "Volume flow rate (total)";
      Modelica.Units.SI.VolumeFlowRate V_flow_single(start=m_flow_start/
            rho_nominal) "Volume flow rate (single pump)";
    Real N "Shaft rotational speed";
      Modelica.Units.SI.Power W_single "Power Consumption (single pump)";
      Modelica.Units.SI.Power W_total=W_single "Power Consumption (total)";
    Real eta "Global Efficiency";
    final constant Medium.MassFlowRate unit_m_flow=1 annotation (HideResult=true);
    Real s(start = m_flow_start/unit_m_flow)
        "Curvilinear abscissa for the flow curve in parametric form (either mass flow rate or head)";

    // Diagnostics
    replaceable model Monitoring =
      Custom_Pump_V2.BaseClasses_Custom.PumpMonitoring.PumpMonitoringBase
        constrainedby
        Custom_Pump_V2.BaseClasses_Custom.PumpMonitoring.PumpMonitoringBase
        "Optional pump monitoring"
        annotation(Dialog(tab="Advanced", group="Diagnostics"), choicesAllMatching=true);
    final parameter Boolean show_NPSHa = false
        "obsolete -- remove modifier and specify Monitoring for NPSH instead"
      annotation(Dialog(tab="Advanced", group="Obsolete"));
    Monitoring monitoring(
            redeclare final package Medium = Medium,
            final state_in = Medium.setState_phX(port_a.p, inStream(port_a.h_outflow), inStream(port_a.Xi_outflow)),
            final state = medium.state) "Monitoring model"
       annotation (Placement(transformation(extent={{-64,-42},{-20,0}})));
    protected
      constant Modelica.Units.SI.Height unitHead=1;
      constant Modelica.Units.SI.MassFlowRate unitMassFlowRate=1;

  equation

    // Flow equations
     V_flow = homotopy(m_flow/rho,
                       m_flow/rho_nominal);
     V_flow_single = V_flow;

      // Regular flow characteristics without check valve
      head = flowCharacteristic(60*60*V_flow_single, N);
      s = 0;

    // Power consumption

      W_single = powerCharacteristic(60*60*V_flow_single,N);

      eta = dp_pump*V_flow_single/W_single;

    // Energy balance
    Wb_flow = W_total;
    Qb_flow = heatTransfer.Q_flows[1];
    Hb_flow = port_a.m_flow*actualStream(port_a.h_outflow) +
              port_b.m_flow*actualStream(port_b.h_outflow);

    // Ports
    port_a.h_outflow = medium.h;
    port_b.h_outflow = medium.h;
    port_b.p = medium.p
        "outlet pressure is equal to medium pressure, which includes Wb_flow";

    // Mass balance
    mb_flow = port_a.m_flow + port_b.m_flow;

    mbXi_flow = port_a.m_flow*actualStream(port_a.Xi_outflow) +
                port_b.m_flow*actualStream(port_b.Xi_outflow);
    port_a.Xi_outflow = medium.Xi;
    port_b.Xi_outflow = medium.Xi;

    mbC_flow = port_a.m_flow*actualStream(port_a.C_outflow) +
               port_b.m_flow*actualStream(port_b.C_outflow);
    port_a.C_outflow = C;
    port_b.C_outflow = C;

    connect(heatTransfer.heatPorts[1], heatPort) annotation (Line(
        points={{40,-34},{40,-60}},
        color={127,0,0}));
    annotation (
      Icon(coordinateSystem(preserveAspectRatio=true,  extent={{-100,-100},{100,
                100}}), graphics={
            Rectangle(
              extent={{-100,46},{100,-46}},
              lineColor={0,0,0},
              fillColor={0,127,255},
              fillPattern=FillPattern.HorizontalCylinder),
            Polygon(
              points={{-48,-60},{-72,-100},{72,-100},{48,-60},{-48,-60}},
              lineColor={0,0,255},
              pattern=LinePattern.None,
              fillColor={0,0,0},
              fillPattern=FillPattern.VerticalCylinder),
            Ellipse(
              extent={{-80,80},{80,-80}},
              lineColor={0,0,0},
              fillPattern=FillPattern.Sphere,
              fillColor={0,100,199}),
            Polygon(
              points={{-28,30},{-28,-30},{50,-2},{-28,30}},
              lineColor={0,0,0},
              pattern=LinePattern.None,
              fillPattern=FillPattern.HorizontalCylinder,
              fillColor={255,255,255})}),
      Documentation(info="<html>
<p>This is the base model for pumps.
<p>The model describes a centrifugal pump, or a group of <code>nParallel</code> identical pumps. The pump model is based on the theory of kinematic similarity: the pump characteristics are given for nominal operating conditions (rotational speed and fluid density), and then adapted to actual operating condition, according to the similarity equations.

<p><b>Pump characteristics</b></p>
<p> The nominal hydraulic characteristic (head vs. volume flow rate) is given by the the replaceable function <code>flowCharacteristic</code>.
<p> The pump energy balance can be specified in two alternative ways:
<ul>
<li><code>use_powerCharacteristic = false</code> (default option): the replaceable function <code>efficiencyCharacteristic</code> (efficiency vs. volume flow rate in nominal conditions) is used to determine the efficiency, and then the power consumption.
    The default is a constant efficiency of 0.8.</li>
<li><code>use_powerCharacteristic = true</code>: the replaceable function <code>powerCharacteristic</code> (power consumption vs. volume flow rate in nominal conditions) is used to determine the power consumption, and then the efficiency.
    Use <code>powerCharacteristic</code> to specify a non-zero power consumption for zero flow rate.
</ul>
<p>
Several functions are provided in the package <code>PumpCharacteristics</code> to specify the characteristics as a function of some operating points at nominal conditions.
<p>Depending on the value of the <code>checkValve</code> parameter, the model either supports reverse flow conditions, or includes a built-in check valve to avoid flow reversal.
</p>
<p>It is possible to take into account the mass and energy storage of the fluid inside the pump by specifying its volume <code>V</code>, and by selecting appropriate dynamic mass and energy balance assumptions (see below);
this is recommended to avoid singularities in the computation of the outlet enthalpy in case of zero flow rate.
If zero flow rate conditions are always avoided, this dynamic effect can be neglected by leaving the default value <code>V = 0</code>, thus avoiding fast state variables in the model.
</p>

<p><b>Dynamics options</b></p>
<p>
Steady-state mass and energy balances are assumed per default, neglecting the holdup of fluid in the pump; this configuration works well if the flow rate is always positive.
Dynamic mass and energy balance can be used by setting the corresponding dynamic parameters. This is recommended to avoid singularities at zero or reversing mass flow rate. If the initial conditions imply non-zero mass flow rate, it is possible to use the <code>SteadyStateInitial</code> condition, otherwise it is recommended to use <code>FixedInitial</code> in order to avoid undetermined initial conditions.
</p>

<p><b>Heat transfer</b></p>
<p>
The Boolean parameter <code>use_HeatTransfer</code> can be set to true if heat exchanged with the environment
should be taken into account or to model a housing. This might be desirable if a pump with realistic
<code>powerCharacteristic</code> for zero flow operates while a valve prevents fluid flow.
</p>

<p><b>Diagnostics of Cavitation</b></p>
<p>The replaceable Monitoring submodel can be configured to PumpMonitoringNPSH,
in order to compute the Net Positive Suction Head available and check for cavitation,
provided a two-phase medium model is used (see Advanced tab).
</p>
</html>",
        revisions="<html>
<ul>
<li><i>8 Jan 2013</i>
    by R&uuml;diger Franke:<br>
    moved NPSH diagnostics from PartialPump to replaceable sub-model PumpMonitoring.PumpMonitoringNPSH (see ticket #646)</li>
<li><i>Dec 2008</i>
    by R&uuml;diger Franke:<br>
    <ul>
    <li>Replaced simplified mass and energy balances with rigorous formulation (base class PartialLumpedVolume)</li>
    <li>Introduced optional HeatTransfer model defining Qb_flow</li>
    <li>Enabled events when the checkValve is operating to support the opening of a discrete valve before port_a</li>
    </ul></li>
<li><i>31 Oct 2005</i>
    by <a href=\"mailto:francesco.casella@polimi.it\">Francesco Casella</a>:<br>
       Model added to the Fluid library</li>
</ul>
</html>"));
  end Custom_Pump_Partial;

  model Pump_cs_Backup "Centrifugal pump with constant speed"
      import         Modelica.Units.NonSI;
      import Modelica.Constants;

    extends Modelica.Fluid.Interfaces.PartialTwoPort(
      port_b_exposesState = energyDynamics<>Modelica.Fluid.Types.Dynamics.SteadyState
                                                                       or massDynamics<>Modelica.Fluid.Types.Dynamics.SteadyState,
      port_a(
        p(start=p_a_start),
        m_flow(start = m_flow_start,
               min = if allowFlowReversal and not checkValve then -Constants.inf else 0)),
      port_b(
        p(start=p_b_start),
        m_flow(start = -m_flow_start,
               max = if allowFlowReversal and not checkValve then +Constants.inf else 0)));

    // Initialization
    parameter Medium.AbsolutePressure p_a_start=system.p_start
        "Guess value for inlet pressure"
      annotation(Dialog(tab="Initialization"));
    parameter Medium.AbsolutePressure p_b_start=p_a_start
        "Guess value for outlet pressure"
      annotation(Dialog(tab="Initialization"));
    parameter Medium.MassFlowRate m_flow_start = system.m_flow_start
        "Guess value of m_flow = port_a.m_flow"
      annotation(Dialog(tab = "Initialization"));

      parameter Real rpm_rel
        "Rotational speed (relative)"
      annotation(Dialog);

      final parameter Modelica.Units.SI.VolumeFlowRate V_flow_single_init=
          m_flow_start/rho_nominal "Used for simplified initialization model";
      final parameter Modelica.Units.SI.Height delta_head_init=
          flowCharacteristic(V_flow_single_init, 1) - flowCharacteristic(0, 1)
        "Used for simplified initialization model";

    // Characteristic curves

    replaceable function flowCharacteristic =
        Custom_Pump_V2.BaseClasses_Custom.PumpCharacteristics.quadraticFlow                        "Head vs. V_flow characteristic"
      annotation(Dialog(group="Characteristics"), choicesAllMatching=true);

    replaceable function powerCharacteristic =
          Custom_Pump_V2.BaseClasses_Custom.PumpCharacteristics.cubicPower
        "Power consumption vs. V_flow at nominal speed and density"
        annotation(Dialog(group="Characteristics"), choicesAllMatching=true);

   final parameter Medium.Density rho_nominal = Medium.density_pTX(Medium.p_default, Medium.T_default, Medium.X_default);
     //   "Nominal fluid density for characteristic"
   //   annotation(Dialog(group="Characteristics"));

    // Assumptions
    parameter Boolean checkValve=false "= true to prevent reverse flow"
      annotation(Dialog(tab="Assumptions"), Evaluate=true);

      parameter Modelica.Units.SI.Volume V=0 "Volume inside the pump"
        annotation (Dialog(tab="Assumptions"), Evaluate=true);

    // Energy and mass balance
    extends Modelica.Fluid.Interfaces.PartialLumpedVolume(
        final fluidVolume = V,
        energyDynamics=Modelica.Fluid.Types.Dynamics.SteadyState,
        massDynamics=Modelica.Fluid.Types.Dynamics.SteadyState,
        final p_start = p_b_start);

    // Heat transfer through boundary, e.g., to add a housing
    parameter Boolean use_HeatTransfer = false
        "= true to use a HeatTransfer model, e.g., for a housing"
        annotation (Dialog(tab="Assumptions",group="Heat transfer"));
    replaceable model HeatTransfer =
        Modelica.Fluid.Vessels.BaseClasses.HeatTransfer.IdealHeatTransfer
      constrainedby
        Modelica.Fluid.Vessels.BaseClasses.HeatTransfer.PartialVesselHeatTransfer
        "Wall heat transfer"
        annotation (Dialog(tab="Assumptions",group="Heat transfer",enable=use_HeatTransfer),choicesAllMatching=true);
    HeatTransfer heatTransfer(
      redeclare final package Medium = Medium,
      final n=1,
      surfaceAreas={4*Modelica.Constants.pi*(3/4*V/Modelica.Constants.pi)^(2/3)},
      final states = {medium.state},
      final use_k = use_HeatTransfer)
        annotation (Placement(transformation(
          extent={{-10,-10},{30,30}},
          rotation=180,
          origin={50,-10})));
    Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_a heatPort if use_HeatTransfer
      annotation (Placement(transformation(extent={{30,-70},{50,-50}})));

    // Variables
      final parameter Modelica.Units.SI.Acceleration g=system.g;
    Medium.Density rho = medium.d;
      Modelica.Units.SI.Pressure dp_pump=port_b.p - port_a.p
        "Pressure increase";
      Modelica.Units.SI.Height head=dp_pump/(rho*g) "Pump head";
      Modelica.Units.SI.MassFlowRate m_flow=port_a.m_flow
        "Mass flow rate (total)";
      Modelica.Units.SI.MassFlowRate m_flow_single=m_flow
        "Mass flow rate (single pump)";
      Modelica.Units.SI.VolumeFlowRate V_flow "Volume flow rate (total)";
      Modelica.Units.SI.VolumeFlowRate V_flow_single(start=m_flow_start/
            rho_nominal) "Volume flow rate (single pump)";
    Real N(start = rpm_rel) "Shaft rotational speed";
      Modelica.Units.SI.Power W_single "Power Consumption (single pump)";
      Modelica.Units.SI.Power W_total=W_single "Power Consumption (total)";
    Real eta "Global Efficiency";
    final constant Medium.MassFlowRate unit_m_flow=1 annotation (HideResult=true);
    Real s(start = m_flow_start/unit_m_flow)
        "Curvilinear abscissa for the flow curve in parametric form (either mass flow rate or head)";

    // Diagnostics
    replaceable model Monitoring =
      Custom_Pump_V2.BaseClasses_Custom.PumpMonitoring.PumpMonitoringBase
        constrainedby
        Custom_Pump_V2.BaseClasses_Custom.PumpMonitoring.PumpMonitoringBase
        "Optional pump monitoring"
        annotation(Dialog(tab="Advanced", group="Diagnostics"), choicesAllMatching=true);
    final parameter Boolean show_NPSHa = false
        "obsolete -- remove modifier and specify Monitoring for NPSH instead"
      annotation(Dialog(tab="Advanced", group="Obsolete"));
    Monitoring monitoring(
            redeclare final package Medium = Medium,
            final state_in = Medium.setState_phX(port_a.p, inStream(port_a.h_outflow), inStream(port_a.Xi_outflow)),
            final state = medium.state) "Monitoring model"
       annotation (Placement(transformation(extent={{-64,-42},{-20,0}})));
    protected
      constant Modelica.Units.SI.Height unitHead=1;
      constant Modelica.Units.SI.MassFlowRate unitMassFlowRate=1;

  equation
    N = rpm_rel;

    // Flow equations
     V_flow = homotopy(m_flow/rho,
                       m_flow/rho_nominal);
     V_flow_single = V_flow;

      // Regular flow characteristics without check valve
      head = flowCharacteristic(60*60*V_flow_single, N);
      s = 0;

    // Power consumption

      W_single = powerCharacteristic(60*60*V_flow_single,N);

      eta = dp_pump*V_flow_single/W_single;

    // Energy balance
    Wb_flow = W_total;
    Qb_flow = heatTransfer.Q_flows[1];
    Hb_flow = port_a.m_flow*actualStream(port_a.h_outflow) +
              port_b.m_flow*actualStream(port_b.h_outflow);

    // Ports
    port_a.h_outflow = medium.h;
    port_b.h_outflow = medium.h;
    port_b.p = medium.p
        "outlet pressure is equal to medium pressure, which includes Wb_flow";

    // Mass balance
    mb_flow = port_a.m_flow + port_b.m_flow;

    mbXi_flow = port_a.m_flow*actualStream(port_a.Xi_outflow) +
                port_b.m_flow*actualStream(port_b.Xi_outflow);
    port_a.Xi_outflow = medium.Xi;
    port_b.Xi_outflow = medium.Xi;

    mbC_flow = port_a.m_flow*actualStream(port_a.C_outflow) +
               port_b.m_flow*actualStream(port_b.C_outflow);
    port_a.C_outflow = C;
    port_b.C_outflow = C;

    connect(heatTransfer.heatPorts[1], heatPort) annotation (Line(
        points={{40,-34},{40,-60}},
        color={127,0,0}));
    annotation (
      Icon(coordinateSystem(preserveAspectRatio=true,  extent={{-100,-100},{100,
                100}}), graphics={
            Rectangle(
              extent={{-100,46},{100,-46}},
              lineColor={0,0,0},
              fillColor={0,127,255},
              fillPattern=FillPattern.HorizontalCylinder),
            Polygon(
              points={{-48,-60},{-72,-100},{72,-100},{48,-60},{-48,-60}},
              lineColor={0,0,255},
              pattern=LinePattern.None,
              fillColor={0,0,0},
              fillPattern=FillPattern.VerticalCylinder),
            Ellipse(
              extent={{-80,80},{80,-80}},
              lineColor={0,0,0},
              fillPattern=FillPattern.Sphere,
              fillColor={0,100,199}),
            Polygon(
              points={{-28,30},{-28,-30},{50,-2},{-28,30}},
              lineColor={0,0,0},
              pattern=LinePattern.None,
              fillPattern=FillPattern.HorizontalCylinder,
              fillColor={255,255,255})}),
      Documentation(info="<html>
<p>This is the base model for pumps.
<p>The model describes a centrifugal pump, or a group of <code>nParallel</code> identical pumps. The pump model is based on the theory of kinematic similarity: the pump characteristics are given for nominal operating conditions (rotational speed and fluid density), and then adapted to actual operating condition, according to the similarity equations.

<p><b>Pump characteristics</b></p>
<p> The nominal hydraulic characteristic (head vs. volume flow rate) is given by the the replaceable function <code>flowCharacteristic</code>.
<p> The pump energy balance can be specified in two alternative ways:
<ul>
<li><code>use_powerCharacteristic = false</code> (default option): the replaceable function <code>efficiencyCharacteristic</code> (efficiency vs. volume flow rate in nominal conditions) is used to determine the efficiency, and then the power consumption.
    The default is a constant efficiency of 0.8.</li>
<li><code>use_powerCharacteristic = true</code>: the replaceable function <code>powerCharacteristic</code> (power consumption vs. volume flow rate in nominal conditions) is used to determine the power consumption, and then the efficiency.
    Use <code>powerCharacteristic</code> to specify a non-zero power consumption for zero flow rate.
</ul>
<p>
Several functions are provided in the package <code>PumpCharacteristics</code> to specify the characteristics as a function of some operating points at nominal conditions.
<p>Depending on the value of the <code>checkValve</code> parameter, the model either supports reverse flow conditions, or includes a built-in check valve to avoid flow reversal.
</p>
<p>It is possible to take into account the mass and energy storage of the fluid inside the pump by specifying its volume <code>V</code>, and by selecting appropriate dynamic mass and energy balance assumptions (see below);
this is recommended to avoid singularities in the computation of the outlet enthalpy in case of zero flow rate.
If zero flow rate conditions are always avoided, this dynamic effect can be neglected by leaving the default value <code>V = 0</code>, thus avoiding fast state variables in the model.
</p>

<p><b>Dynamics options</b></p>
<p>
Steady-state mass and energy balances are assumed per default, neglecting the holdup of fluid in the pump; this configuration works well if the flow rate is always positive.
Dynamic mass and energy balance can be used by setting the corresponding dynamic parameters. This is recommended to avoid singularities at zero or reversing mass flow rate. If the initial conditions imply non-zero mass flow rate, it is possible to use the <code>SteadyStateInitial</code> condition, otherwise it is recommended to use <code>FixedInitial</code> in order to avoid undetermined initial conditions.
</p>

<p><b>Heat transfer</b></p>
<p>
The Boolean parameter <code>use_HeatTransfer</code> can be set to true if heat exchanged with the environment
should be taken into account or to model a housing. This might be desirable if a pump with realistic
<code>powerCharacteristic</code> for zero flow operates while a valve prevents fluid flow.
</p>

<p><b>Diagnostics of Cavitation</b></p>
<p>The replaceable Monitoring submodel can be configured to PumpMonitoringNPSH,
in order to compute the Net Positive Suction Head available and check for cavitation,
provided a two-phase medium model is used (see Advanced tab).
</p>
</html>",
        revisions="<html>
<ul>
<li><i>8 Jan 2013</i>
    by R&uuml;diger Franke:<br>
    moved NPSH diagnostics from PartialPump to replaceable sub-model PumpMonitoring.PumpMonitoringNPSH (see ticket #646)</li>
<li><i>Dec 2008</i>
    by R&uuml;diger Franke:<br>
    <ul>
    <li>Replaced simplified mass and energy balances with rigorous formulation (base class PartialLumpedVolume)</li>
    <li>Introduced optional HeatTransfer model defining Qb_flow</li>
    <li>Enabled events when the checkValve is operating to support the opening of a discrete valve before port_a</li>
    </ul></li>
<li><i>31 Oct 2005</i>
    by <a href=\"mailto:francesco.casella@polimi.it\">Francesco Casella</a>:<br>
       Model added to the Fluid library</li>
</ul>
</html>"));
  end Pump_cs_Backup;
  end BaseClasses_Custom;
  annotation (uses(Modelica(version="4.0.0")),
    version="1",
    conversion(noneFromVersion=""));
end Custom_Pump_V2;

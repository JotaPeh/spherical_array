# ----------------------------------------------
# Script Recorded by Ansys Electronics Desktop Version 2024.2.0
# 22:55:29  Sep 26, 2025
# ----------------------------------------------
import ScriptEnv
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()
oProject = oDesktop.SetActiveProject("Spherical_Array")
oDesign = oProject.SetActiveDesign("CP")
oModule = oDesign.GetModule("ReportSetup")
oModule.CreateReport("Antenna Params Table 3", "Antenna Parameters", "Data Table", "GPS_Band : LastAdaptive", 
	[
		"Context:="		, "3D"
	], 
	[
		"Freq:="		, ["All"],
		"a:="			, ["Nominal"],
		"numseg:="		, ["Nominal"],
		"h:="			, ["Nominal"],
		"Dthetaa:="		, ["Nominal"],
		"Dphia:="		, ["Nominal"],
		"phip1:="		, ["Nominal"],
		"thetap1:="		, ["Nominal"],
		"Rteflon:="		, ["Nominal"],
		"rprobe:="		, ["Nominal"],
		"Hprobe:="		, ["Nominal"],
		"Alpha1:="		, ["Nominal"],
		"Beta1:="		, ["Nominal"],
		"Gamma1:="		, ["Nominal"]
	], 
	[
		"X Component:="		, "Freq",
		"Y Component:="		, ["dB(RadiationEfficiency)"]
	])
oModule.CreateReport("Gain 3D (dB)", "Far Fields", "3D Polar Plot", "GPS_Band : GPS_Discrete", 
	[
		"Context:="		, "3D"
	], 
	[
		"Phi:="			, ["All"],
		"Theta:="		, ["All"],
		"a:="			, ["100mm"],
		"numseg:="		, ["19"],
		"h:="			, ["1.524mm"],
		"Rteflon:="		, ["2.05mm"],
		"rprobe:="		, ["0.65mm"],
		"Hprobe:="		, ["15.07mm"],
		"Alpha1:="		, ["0deg"],
		"Beta1:="		, ["0deg"],
		"Gamma1:="		, ["0deg"],
		"Dthetaa:="		, ["33.5657deg"],
		"Dphia:="		, ["33.3212deg"],
		"phip1:="		, ["94.5327deg"],
		"thetap1:="		, ["95.5768deg"],
		"Freq:="		, ["1.57542GHz"]
	], 
	[
		"Phi Component:="	, "Phi",
		"Theta Component:="	, "Theta",
		"Mag Component:="	, ["dB(GainLHCP)","dB(GainRHCP)"]
	])
oModule.ExportToFile("Gain 3D (dB)", "C:/Users/ITA/Documents/Alunos/JoaoPedroFalcao/Mestrado/Codes/Gain 3D (dB).csv", False)
oModule.CreateReport("Efficiency5", "Antenna Parameters", "Data Table", "GPS_Band : LastAdaptive", 
	[
		"Domain:="		, "Sweep"
	], 
	[
		"Freq:="		, ["All"],
		"a:="			, ["Nominal"],
		"numseg:="		, ["Nominal"],
		"h:="			, ["Nominal"],
		"Rteflon:="		, ["Nominal"],
		"rprobe:="		, ["Nominal"],
		"Hprobe:="		, ["Nominal"],
		"Alpha1:="		, ["Nominal"],
		"Beta1:="		, ["Nominal"],
		"Gamma1:="		, ["Nominal"],
		"Dthetaa:="		, ["Nominal"],
		"Dphia:="		, ["Nominal"],
		"phip1:="		, ["Nominal"],
		"thetap1:="		, ["Nominal"]
	], 
	[
		"X Component:="		, "Freq",
		"Y Component:="		, ["RadiationEfficiency"]
	])

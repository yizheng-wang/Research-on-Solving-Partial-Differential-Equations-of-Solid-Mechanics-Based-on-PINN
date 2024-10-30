# trace generated using paraview version 5.11.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
plate_hole_PINNsvtu = XMLUnstructuredGridReader(registrationName='Plate_hole_PINNs.vtu', FileName=['C:/Users/admin/OneDrive/KINN/src_KINN/Plate_hole/results/PINNs_MLP_penalty/Plate_hole_PINNs.vtu'])
plate_hole_PINNsvtu.PointArrayStatus = ['displacementX', 'displacementY', 'displacementZ', 'S-VonMises', 'U-mag', 'U_mag_error', 'MISES_error', 'S11', 'S12', 'S13', 'S22', 'S23', 'S33', 'E11', 'E12', 'E13', 'E22', 'E23', 'E33']

# Properties modified on plate_hole_PINNsvtu
plate_hole_PINNsvtu.TimeArray = 'None'

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
plate_hole_PINNsvtuDisplay = Show(plate_hole_PINNsvtu, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'displacementX'
displacementXLUT = GetColorTransferFunction('displacementX')

# get opacity transfer function/opacity map for 'displacementX'
displacementXPWF = GetOpacityTransferFunction('displacementX')

# trace defaults for the display properties.
plate_hole_PINNsvtuDisplay.Representation = 'Surface'
plate_hole_PINNsvtuDisplay.ColorArrayName = ['POINTS', 'displacementX']
plate_hole_PINNsvtuDisplay.LookupTable = displacementXLUT
plate_hole_PINNsvtuDisplay.SelectTCoordArray = 'None'
plate_hole_PINNsvtuDisplay.SelectNormalArray = 'None'
plate_hole_PINNsvtuDisplay.SelectTangentArray = 'None'
plate_hole_PINNsvtuDisplay.OSPRayScaleArray = 'displacementX'
plate_hole_PINNsvtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
plate_hole_PINNsvtuDisplay.SelectOrientationVectors = 'None'
plate_hole_PINNsvtuDisplay.ScaleFactor = 1.99329984895885
plate_hole_PINNsvtuDisplay.SelectScaleArray = 'displacementX'
plate_hole_PINNsvtuDisplay.GlyphType = 'Arrow'
plate_hole_PINNsvtuDisplay.GlyphTableIndexArray = 'displacementX'
plate_hole_PINNsvtuDisplay.GaussianRadius = 0.0996649924479425
plate_hole_PINNsvtuDisplay.SetScaleArray = ['POINTS', 'displacementX']
plate_hole_PINNsvtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
plate_hole_PINNsvtuDisplay.OpacityArray = ['POINTS', 'displacementX']
plate_hole_PINNsvtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'
plate_hole_PINNsvtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
plate_hole_PINNsvtuDisplay.PolarAxes = 'PolarAxesRepresentation'
plate_hole_PINNsvtuDisplay.ScalarOpacityFunction = displacementXPWF
plate_hole_PINNsvtuDisplay.ScalarOpacityUnitDistance = 0.6675643736558438
plate_hole_PINNsvtuDisplay.OpacityArrayName = ['POINTS', 'displacementX']
plate_hole_PINNsvtuDisplay.SelectInputVectors = [None, '']
plate_hole_PINNsvtuDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
plate_hole_PINNsvtuDisplay.ScaleTransferFunction.Points = [-0.28545910120010376, 0.0, 0.5, 0.0, 2.547891139984131, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
plate_hole_PINNsvtuDisplay.OpacityTransferFunction.Points = [-0.28545910120010376, 0.0, 0.5, 0.0, 2.547891139984131, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera(False)

#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [10.000000083819032, 10.000000083819032, 10000.0]
renderView1.CameraFocalPoint = [10.000000083819032, 10.000000083819032, 0.0]

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
plate_hole_PINNsvtuDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get 2D transfer function for 'displacementX'
displacementXTF2D = GetTransferFunction2D('displacementX')

# set scalar coloring
ColorBy(plate_hole_PINNsvtuDisplay, ('POINTS', 'U-mag'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(displacementXLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
plate_hole_PINNsvtuDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
plate_hole_PINNsvtuDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Umag'
umagLUT = GetColorTransferFunction('Umag')

# get opacity transfer function/opacity map for 'Umag'
umagPWF = GetOpacityTransferFunction('Umag')

# get 2D transfer function for 'Umag'
umagTF2D = GetTransferFunction2D('Umag')

# get color legend/bar for umagLUT in view renderView1
umagLUTColorBar = GetScalarBar(umagLUT, renderView1)

# change scalar bar placement
umagLUTColorBar.WindowLocation = 'Any Location'
umagLUTColorBar.Position = [0.15324858757062146, 0.521978021978022]
umagLUTColorBar.ScalarBarLength = 0.32999999999999985

# set scalar coloring using an separate color/opacity maps
ColorBy(plate_hole_PINNsvtuDisplay, ('POINTS', 'U-mag'), True)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(umagLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
plate_hole_PINNsvtuDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
plate_hole_PINNsvtuDisplay.SetScalarBarVisibility(renderView1, True)

# get separate color transfer function/color map for 'Umag'
separate_plate_hole_PINNsvtuDisplay_UmagLUT = GetColorTransferFunction('Umag', plate_hole_PINNsvtuDisplay, separate=True)

# get separate opacity transfer function/opacity map for 'Umag'
separate_plate_hole_PINNsvtuDisplay_UmagPWF = GetOpacityTransferFunction('Umag', plate_hole_PINNsvtuDisplay, separate=True)

# get separate 2D transfer function for 'Umag'
separate_plate_hole_PINNsvtuDisplay_UmagTF2D = GetTransferFunction2D('Umag', plate_hole_PINNsvtuDisplay, separate=True)

# set scalar coloring
ColorBy(plate_hole_PINNsvtuDisplay, ('POINTS', 'U-mag'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(separate_plate_hole_PINNsvtuDisplay_UmagLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
plate_hole_PINNsvtuDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
plate_hole_PINNsvtuDisplay.SetScalarBarVisibility(renderView1, True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
umagLUT.ApplyPreset('Rainbow Uniform', True)

# Properties modified on umagLUT
umagLUT.NumberOfTableValues = 12

# change scalar bar placement
umagLUTColorBar.Position = [0.06403290129611167, 0.5230769230769231]

# change scalar bar placement
umagLUTColorBar.Position = [0.06403290129611167, 0.521978021978022]
umagLUTColorBar.ScalarBarLength = 0.33109890109890094

# Properties modified on umagLUTColorBar
umagLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
umagLUTColorBar.LabelColor = [0.0, 0.0, 0.0]

# Properties modified on renderView1
renderView1.UseColorPaletteForBackground = 0

# Properties modified on renderView1
renderView1.Background = [1.0, 1.0, 1.0]

# change scalar bar placement
umagLUTColorBar.Position = [0.2575357261548687, 0.4120879120879122]

# export view
ExportView('E:/paraview/bin/Plate_hole_PINNs_U_mag.pdf', view=renderView1)

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1416, 910)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [10.000000083819032, 10.000000083819032, 10000.0]
renderView1.CameraFocalPoint = [10.000000083819032, 10.000000083819032, 0.0]
renderView1.CameraParallelScale = 20.636135775444696

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
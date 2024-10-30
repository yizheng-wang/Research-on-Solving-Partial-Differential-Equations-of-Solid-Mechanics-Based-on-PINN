# trace generated using paraview version 5.11.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
neoHook_KAN_Simp_disy_errorvtu = XMLUnstructuredGridReader(registrationName='NeoHook_KAN_Simp_disy_error.vtu', FileName=['C:/Users/admin/OneDrive/KINN/src_KINN/dem_hyperelasticity/Beam2D/output/dem/NeoHook_KAN_Simp_disy_error.vtu'])
neoHook_KAN_Simp_disy_errorvtu.PointArrayStatus = ['dis_Y_error']

# Properties modified on neoHook_KAN_Simp_disy_errorvtu
neoHook_KAN_Simp_disy_errorvtu.TimeArray = 'None'

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
neoHook_KAN_Simp_disy_errorvtuDisplay = Show(neoHook_KAN_Simp_disy_errorvtu, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'dis_Y_error'
dis_Y_errorLUT = GetColorTransferFunction('dis_Y_error')

# get opacity transfer function/opacity map for 'dis_Y_error'
dis_Y_errorPWF = GetOpacityTransferFunction('dis_Y_error')

# trace defaults for the display properties.
neoHook_KAN_Simp_disy_errorvtuDisplay.Representation = 'Surface'
neoHook_KAN_Simp_disy_errorvtuDisplay.ColorArrayName = ['POINTS', 'dis_Y_error']
neoHook_KAN_Simp_disy_errorvtuDisplay.LookupTable = dis_Y_errorLUT
neoHook_KAN_Simp_disy_errorvtuDisplay.SelectTCoordArray = 'None'
neoHook_KAN_Simp_disy_errorvtuDisplay.SelectNormalArray = 'None'
neoHook_KAN_Simp_disy_errorvtuDisplay.SelectTangentArray = 'None'
neoHook_KAN_Simp_disy_errorvtuDisplay.OSPRayScaleArray = 'dis_Y_error'
neoHook_KAN_Simp_disy_errorvtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
neoHook_KAN_Simp_disy_errorvtuDisplay.SelectOrientationVectors = 'None'
neoHook_KAN_Simp_disy_errorvtuDisplay.ScaleFactor = 0.4
neoHook_KAN_Simp_disy_errorvtuDisplay.SelectScaleArray = 'dis_Y_error'
neoHook_KAN_Simp_disy_errorvtuDisplay.GlyphType = 'Arrow'
neoHook_KAN_Simp_disy_errorvtuDisplay.GlyphTableIndexArray = 'dis_Y_error'
neoHook_KAN_Simp_disy_errorvtuDisplay.GaussianRadius = 0.02
neoHook_KAN_Simp_disy_errorvtuDisplay.SetScaleArray = ['POINTS', 'dis_Y_error']
neoHook_KAN_Simp_disy_errorvtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
neoHook_KAN_Simp_disy_errorvtuDisplay.OpacityArray = ['POINTS', 'dis_Y_error']
neoHook_KAN_Simp_disy_errorvtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'
neoHook_KAN_Simp_disy_errorvtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
neoHook_KAN_Simp_disy_errorvtuDisplay.PolarAxes = 'PolarAxesRepresentation'
neoHook_KAN_Simp_disy_errorvtuDisplay.ScalarOpacityFunction = dis_Y_errorPWF
neoHook_KAN_Simp_disy_errorvtuDisplay.ScalarOpacityUnitDistance = 0.15126634134114414
neoHook_KAN_Simp_disy_errorvtuDisplay.OpacityArrayName = ['POINTS', 'dis_Y_error']
neoHook_KAN_Simp_disy_errorvtuDisplay.SelectInputVectors = [None, '']
neoHook_KAN_Simp_disy_errorvtuDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
neoHook_KAN_Simp_disy_errorvtuDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.0540975923090925, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
neoHook_KAN_Simp_disy_errorvtuDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.0540975923090925, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera(False)

#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [2.0, 0.5, 10000.0]
renderView1.CameraFocalPoint = [2.0, 0.5, 0.0]

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
neoHook_KAN_Simp_disy_errorvtuDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get 2D transfer function for 'dis_Y_error'
dis_Y_errorTF2D = GetTransferFunction2D('dis_Y_error')

# get color legend/bar for dis_Y_errorLUT in view renderView1
dis_Y_errorLUTColorBar = GetScalarBar(dis_Y_errorLUT, renderView1)

# change scalar bar placement
dis_Y_errorLUTColorBar.WindowLocation = 'Any Location'
dis_Y_errorLUTColorBar.Position = [0.07928118393234684, 0.25250278086763067]
dis_Y_errorLUTColorBar.ScalarBarLength = 0.32999999999999974

# change scalar bar placement
dis_Y_errorLUTColorBar.Position = [0.07928118393234684, 0.41601779755283647]
dis_Y_errorLUTColorBar.ScalarBarLength = 0.16648498331479394

# change scalar bar placement
dis_Y_errorLUTColorBar.Position = [0.05073995771670202, 0.41601779755283647]

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
dis_Y_errorLUT.ApplyPreset('Turbo', True)

# Properties modified on dis_Y_errorLUTColorBar
dis_Y_errorLUTColorBar.LabelColor = [0.0, 0.0, 0.16000610360875867]

# Properties modified on renderView1
renderView1.UseColorPaletteForBackground = 0

# Properties modified on renderView1
renderView1.Background = [1.0, 1.0, 1.0]

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(946, 899)

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [2.0, 0.5, 10000.0]
renderView1.CameraFocalPoint = [2.0, 0.5, 0.0]
renderView1.CameraParallelScale = 3.018319473233409

# save screenshot
SaveScreenshot('C:/Users/admin/OneDrive/KINN/src_KINN/dem_hyperelasticity/Beam2D/output/Beam_KAN_Simp_disy_error.png', renderView1, ImageResolution=[946, 899])

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(946, 899)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [2.0, 0.5, 10000.0]
renderView1.CameraFocalPoint = [2.0, 0.5, 0.0]
renderView1.CameraParallelScale = 3.018319473233409

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
# trace generated using paraview version 5.11.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
error_pinn_mlpvtu = XMLUnstructuredGridReader(registrationName='error_pinn_mlp.vtu', FileName=['C:/Users/admin/OneDrive/KINN/src_KINN/complex_geo/poisson/output_ntk/error_pinn_mlp.vtu'])
error_pinn_mlpvtu.PointArrayStatus = ['error_pinn']

# Properties modified on error_pinn_mlpvtu
error_pinn_mlpvtu.TimeArray = 'None'

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
error_pinn_mlpvtuDisplay = Show(error_pinn_mlpvtu, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'error_pinn'
error_pinnLUT = GetColorTransferFunction('error_pinn')

# get opacity transfer function/opacity map for 'error_pinn'
error_pinnPWF = GetOpacityTransferFunction('error_pinn')

# trace defaults for the display properties.
error_pinn_mlpvtuDisplay.Representation = 'Surface'
error_pinn_mlpvtuDisplay.ColorArrayName = ['POINTS', 'error_pinn']
error_pinn_mlpvtuDisplay.LookupTable = error_pinnLUT
error_pinn_mlpvtuDisplay.SelectTCoordArray = 'None'
error_pinn_mlpvtuDisplay.SelectNormalArray = 'None'
error_pinn_mlpvtuDisplay.SelectTangentArray = 'None'
error_pinn_mlpvtuDisplay.OSPRayScaleArray = 'error_pinn'
error_pinn_mlpvtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
error_pinn_mlpvtuDisplay.SelectOrientationVectors = 'None'
error_pinn_mlpvtuDisplay.ScaleFactor = 0.34409303665161134
error_pinn_mlpvtuDisplay.SelectScaleArray = 'error_pinn'
error_pinn_mlpvtuDisplay.GlyphType = 'Arrow'
error_pinn_mlpvtuDisplay.GlyphTableIndexArray = 'error_pinn'
error_pinn_mlpvtuDisplay.GaussianRadius = 0.01720465183258057
error_pinn_mlpvtuDisplay.SetScaleArray = ['POINTS', 'error_pinn']
error_pinn_mlpvtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
error_pinn_mlpvtuDisplay.OpacityArray = ['POINTS', 'error_pinn']
error_pinn_mlpvtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'
error_pinn_mlpvtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
error_pinn_mlpvtuDisplay.PolarAxes = 'PolarAxesRepresentation'
error_pinn_mlpvtuDisplay.ScalarOpacityFunction = error_pinnPWF
error_pinn_mlpvtuDisplay.ScalarOpacityUnitDistance = 0.12377974811132524
error_pinn_mlpvtuDisplay.OpacityArrayName = ['POINTS', 'error_pinn']
error_pinn_mlpvtuDisplay.SelectInputVectors = [None, '']
error_pinn_mlpvtuDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
error_pinn_mlpvtuDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.06842303276062012, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
error_pinn_mlpvtuDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.06842303276062012, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera(False)

#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.0, 0.0, 10000.0]

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
error_pinn_mlpvtuDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get 2D transfer function for 'error_pinn'
error_pinnTF2D = GetTransferFunction2D('error_pinn')

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
error_pinnLUT.ApplyPreset('Turbo', True)

# get color legend/bar for error_pinnLUT in view renderView1
error_pinnLUTColorBar = GetScalarBar(error_pinnLUT, renderView1)

# Properties modified on error_pinnLUTColorBar
error_pinnLUTColorBar.LabelColor = [0.0, 0.0, 0.16000610360875867]

# change scalar bar placement
error_pinnLUTColorBar.WindowLocation = 'Any Location'
error_pinnLUTColorBar.Position = [0.09799789251844052, 0.5212765957446808]
error_pinnLUTColorBar.ScalarBarLength = 0.33000000000000007

# change scalar bar placement
error_pinnLUTColorBar.Position = [0.15173867228661753, 0.553191489361702]

# change scalar bar placement
error_pinnLUTColorBar.Position = [0.15173867228661753, 0.15653495440729473]
error_pinnLUTColorBar.ScalarBarLength = 0.7266565349544074

# change scalar bar placement
error_pinnLUTColorBar.Position = [0.20864067439409906, 0.1641337386018237]

# Properties modified on renderView1
renderView1.OrientationAxesVisibility = 0

# Properties modified on renderView1
renderView1.UseColorPaletteForBackground = 0

# Properties modified on renderView1
renderView1.Background = [1.0, 1.0, 1.0]

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(949, 658)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [-0.7540419488951751, -0.03458908022454931, 10000.0]
renderView1.CameraFocalPoint = [-0.7540419488951751, -0.03458908022454931, 0.0]
renderView1.CameraParallelScale = 2.275961478775345

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
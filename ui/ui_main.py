import sys
import vtk
from custom_types import *
from ui import ui_utils, interactors, ui_controllers
import options


def init_camera(*renders: vtk.vtkRenderer):
    for render in renders:
        camera = render.GetActiveCamera()
        camera.SetPosition(3, 1, -3)


def add_shadows(renderer):
    colors = vtk.vtkNamedColors()
    colors.SetColor('HighNoonSun', [255, 255, 251, 255])  # Color temp. 5400°K
    colors.SetColor('100W Tungsten', [255, 214, 170, 255])  # Color temp. 2850°K
    light1 = vtk.vtkLight()
    light1.SetFocalPoint(0, 0, 0)
    light1.SetPosition(0, 1, 0.2)
    light1.SetColor(colors.GetColor3d('HighNoonSun'))
    light1.SetIntensity(0.3)
    light2 = vtk.vtkLight()
    light2.SetFocalPoint(0, 0, 0)
    light2.SetPosition(1.0, 1.0, 1.0)
    light2.SetColor(colors.GetColor3d('100W Tungsten'))
    light2.SetIntensity(0.8)
    renderer.AddLight(light1)
    renderer.AddLight(light2)

    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()
    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)
    #
    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)
    #
    glrenderer = renderer
    glrenderer.SetPass(cameraP)
    # renderer.GetActiveCamera().SetPosition(-0.2, 0.2, 1)
    # renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
    # renderer.GetActiveCamera().SetViewUp(0, 1, 0)
    # renderer.ResetCamera()
    # renderer.GetActiveCamera().Dolly(2.25)
    # renderer.ResetCameraClippingRange()


def main_single(opt: options.Options, with_model: bool, shape_num: int):
    render_window = vtk.vtkRenderWindow()
    background_renderer_a, background_renderer_b = vtk.vtkRenderer(), vtk.vtkRenderer()
    ren_left = ui_utils.CanvasRender((0.0, 0.0, 1, 1), render_window,
                                     ui_utils.rgb_to_float(ui_utils.bg_source_color))

    render_window.SetMultiSamples(0)
    render_window.SetNumberOfLayers(2)
    background_renderer_a.SetViewport(0.0, 0.0, 1., 1)
    background_renderer_b.SetViewport(0.0, 0.05, 0.4, .15)
    background_renderer_a.InteractiveOff()
    background_renderer_b.InteractiveOff()
    background_renderer_a.SetLayer(0)
    background_renderer_b.SetLayer(0)
    ren_left.SetLayer(1)
    background_renderer_a.SetBackground(*ui_utils.rgb_to_float(ui_utils.bg_source_color))
    background_renderer_b.SetBackground(*ui_utils.rgb_to_float(ui_utils.bg_menu_color))
    render_window.AddRenderer(background_renderer_a)
    render_window.AddRenderer(background_renderer_b)
    init_camera(ren_left)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(render_window)
    view_style = [ui_utils.ViewStyle((200, 200, 200), (200, 200, 200), ui_utils.palette[0], 1)]
    status = ui_controllers.MeshGmmStatuses(opt, [shape_num], [ren_left], view_style, with_model)

    style = interactors.SingleInteractorStyle(opt, status, iren)

    # add_shadows(ren_left)
    iren.SetInteractorStyle(style)
    render_window.Render()
    iren.Initialize()
    render_window.Render()
    iren.Start()
    del iren
    del render_window
    status.exit()


def main_mix(opt: options.Options, with_model: bool, *shape_num: int):
    # colors = vtk.vtkNamedColors()
    num_shapes = len(shape_num)
    # background_renderer_b = vtk.vtkRenderer()
    background_renderer_a, background_renderer_b = vtk.vtkRenderer(), vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.SetNumberOfLayers(2)
    right_height = 1. / num_shapes
    offset_left = 1 - right_height * 1080. / 1920

    ren_left = ui_utils.CanvasRender((0.0, 0.0, offset_left, 1), render_window,
                                     ui_utils.rgb_to_float(ui_utils.bg_source_color),
                                     stroke_color=list(ui_utils.bg_target_color) + [200])
    background_renderer_a.SetViewport(0.0, 0.0, offset_left, 1)
    background_renderer_b.SetViewport(0.0, 0.05, offset_left / 2, .15)
    background_renderer_a.SetBackground(*ui_utils.rgb_to_float(ui_utils.bg_source_color))
    background_renderer_b.SetBackground(*ui_utils.rgb_to_float(ui_utils.bg_menu_color))
    background_renderer_a.InteractiveOff()
    background_renderer_b.InteractiveOff()
    background_renderer_a.SetLayer(0)
    background_renderer_b.SetLayer(0)
    ren_left.SetLayer(1)
    render_window.AddRenderer(background_renderer_a)
    render_window.AddRenderer(background_renderer_b)
    rens_right = [ui_utils.CanvasRender((offset_left, i * right_height, 1., (i + 1) * right_height), render_window,
                                        ui_utils.rgb_to_float(ui_utils.bg_target_color),
                                        stroke_color=list(ui_utils.bg_source_color) + [200]) for i in range(num_shapes)]
    gmm_united = ui_controllers.MeshGmmUnited(opt, list(shape_num), rens_right,
                                              ui_utils.get_view_styles(num_shapes + 1), ren_left, with_model)
    init_camera(ren_left, *rens_right)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(render_window)
    iren.Initialize()
    style = interactors.MixInteractorStyle(opt, gmm_united, iren)
    iren.SetInteractorStyle(style)
    render_window.Render()
    iren.Start()
    del iren
    del render_window
    gmm_united.exit()


def main():
    args = sys.argv
    select = 0
    opt = (options.Options(device=CUDA(0), tag='chairs_sym_hard', model_name="occ_gmm"),
           options.Options(device=CUDA(0), tag='airplanes_sym_hard', model_name="occ_gmm"),
           options.Options(device=CUDA(0), tag='guitars_tb', model_name="occ_gmm"),
           options.Options(device=CUDA(0), tag='coseg_vases', model_name="occ_gmm"))[select].load()
    shape_num = [0, 0, 3631, 3718][3]
    if len(args) > 1:
        for i in range(1, len(args), 2):
            arg = args[i].split('-')[-1]
            if hasattr(opt, arg):
                setattr(opt, arg, args[i + 1])
            elif arg == "shape_num":
                shape_num = int(args[i + 1])
    # main_single(opt, True, shape_num)
    # main_mix(opt, True, 3658, 3091, 252)
    # main_mix(opt, True, 1637, 3631, 6567)
    # main_single(opt, True, 127)
    main_mix(opt, True, 188, 4814, 2954, 3631, 3327, 4410, 5551)
    # main_mix(opt, False, 3631, 5710)
    # main_mix(opt, False, 164, 260, 27)
    # main_mix(opt, True, 0, 1, 4)
    # main_single(opt, False, 3631)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    main()
    # interactors.make_slider(None)
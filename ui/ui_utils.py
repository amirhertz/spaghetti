import torch
import vtk
from utils import files_utils
from custom_types import *
import functools
import matplotlib.pyplot as plt


bg_source_color = (152, 181, 234)
bg_target_color = (250, 200, 152)
button_color = (255, 0, 255)
bg_menu_color = (214, 139, 202)
bg_stage_color = (255, 180, 110)
default_colors = [(82, 108, 255), (160, 82, 255), (255, 43, 43), (255, 246, 79),
                  (153, 227, 107), (58, 186, 92), (8, 243, 255), (240, 136, 0)]


class SmoothingMethod(Enum):
    Laplace = "laplace"
    Taubin = "taubin"


class EditType(enum.Enum):
    Pondering = 'pondering'
    Translating = 'translating'
    Rotating = 'rotating'
    Scaling = 'scaling'
    Marking = 'marking'


class EditDirection(enum.Enum):
    X_Axis = 'axis_x'
    Y_Axis = 'axis_y'
    Z_Axis = 'axis_z'


palette = (
    (63, 72, 204),
    (51, 213, 73),
    (213, 51, 159),
    (153, 227, 107),
    (246, 162, 81)
)
# palette = [(.6196, 0.0039, 0.2588),
#            (.6873, 0.0790, 0.2748),
#            (.7549, 0.1540, 0.2908),
#            (.8226, 0.2291, 0.3068),
#            (.8710, 0.2973, 0.2960),
#            (.9092, 0.3552, 0.2812),
#            (.9473, 0.4130, 0.2664),
#            (.9652, 0.4874, 0.2904),
#            (.9776, 0.5774, 0.3319),
#            (.9887, 0.6574, 0.3689),
#            (.9930, 0.7246, 0.4159),
#            (.9942, 0.7862, 0.4676),
#            (.9956, 0.8554, 0.5257),
#            (.9968, 0.9023, 0.5851),
#            (.9981, 0.9404, 0.6491),
#            (.9993, 0.9785, 0.7130),
#            (.9827, 0.9931, 0.7220),
#            (.9519, 0.9808, 0.6740),
#            (.9212, 0.9685, 0.6261),
#            (.8747, 0.9497, 0.6016),
#            (.7931, 0.9165, 0.6182),
#            (.7205, 0.8870, 0.6330),
#            (.6441, 0.8563, 0.6435),
#            (.5592, 0.8231, 0.6448),
#            (.4637, 0.7857, 0.6461),
#            (.3840, 0.7429, 0.6544),
#            (.3200, 0.6716, 0.6840),
#            (.2561, 0.6002, 0.7135),
#            (.2062, 0.5202, 0.7349),
#            (.2604, 0.4501, 0.7017),
#            (.3145, 0.3799, 0.6685),
#            (.3686, 0.3098, 0.6353)]

RGB_COLOR = Union[Tuple[int, int, int], List[int]]
RGB_FLOAT_COLOR = Union[Tuple[float, float, float], List[float]]
RGBA_COLOR = Union[Tuple[int, int, int, int], List[int]]
RGBA_FLOAT_COLOR = Union[Tuple[float, float, float, float], List[float]]


def channel_to_float(*channel: int):
    if type(channel[0]) is float and 0 <= channel[0] <= 1:
        return channel
    return [c / 255. for c in channel]


def rgb_to_float(*colors: RGB_COLOR) -> Union[RGB_FLOAT_COLOR, List[RGB_FLOAT_COLOR]]:
    float_colors = [channel_to_float(*c) for c in colors]
    if len(float_colors) == 1:
        return float_colors[0]
    return float_colors


def rgb_to_rgba_float(color: RGB_COLOR, alpha: float) -> RGBA_FLOAT_COLOR:
    color = list(rgb_to_float(color)) + [alpha]
    return color


class Buttons(enum.Enum):
    translate = 'T'
    rotate = 'R'
    stretch = 'S'
    reset = 'reset'
    update = 'hq'
    symmetric = 'symmetric'
    empty = -1


class ViewStyle:

    def __init__(self, base_color: RGB_COLOR, included_color: RGB_COLOR, selected_color: RGB_COLOR,
                 opacity: float):
        self.base_color = rgb_to_float(base_color)
        self.included_color = rgb_to_float(included_color)
        self.stroke_color = list(selected_color) + [200]
        self.selected_color = rgb_to_float(selected_color)
        self.opacity = opacity


class Transition:

    def __init__(self, transition_origin: ARRAY, transition_type: EditType):
        self.transition_origin: ARRAY = transition_origin
        self.transition_type: ARRAY = transition_type
        self.translation: ARRAY = np.zeros(3)
        self.rotation: ARRAY = np.eye(3)


@functools.lru_cache(10)
def get_rotation_matrix(theta: float, axis: float) -> ARRAY:
    rotate_mat = np.eye(3)
    rotate_mat[axis, axis] = 1
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotate_mat[(axis + 1) % 3, (axis + 1) % 3] = cos_theta
    rotate_mat[(axis + 2) % 3, (axis + 2) % 3] = cos_theta
    rotate_mat[(axis + 1) % 3, (axis + 2) % 3] = sin_theta
    rotate_mat[(axis + 2) % 3, (axis + 1) % 3] = -sin_theta
    return rotate_mat


def load_vtk(path: str, vtk_reader):
    vtk_reader.SetFileName(path)
    vtk_reader.Update()
    source = vtk_reader.GetOutput()
    return source


def save_vtk(data, path: str, vtk_writer):
    vtk_writer.SetFileName(path)
    vtk_writer.SetInputData(data)
    vtk_writer.Update()
    vtk_writer.Write()


def load_vtk_obj(path: str):
    path = files_utils.add_suffix(path, ".obj")
    return load_vtk(path, vtk.vtkOBJReader())


def save_vtk_image(data, path: str):
    path = files_utils.add_suffix(path, ".vtk")
    files_utils.init_folders(path)
    save_vtk(data, path, vtk.vtkXMLImageDataWriter())


def load_vtk_image(path: str) -> vtk.vtkImageData:
    path = files_utils.add_suffix(path, ".vtk")
    return load_vtk(path, vtk.vtkXMLImageDataReader())


def set_default_properties(actor: vtk.vtkActor, color: Tuple[float, float, float]):
    properties = actor.GetProperty()
    properties.SetPointSize(10)
    properties.SetDiffuseColor(.6, .6, .6)
    properties.SetAmbient(.2)
    properties.SetDiffuse(.8)
    properties.SetSpecular(.5)
    properties.SetSpecularColor(.2, .2, .2)
    properties.SetSpecularPower(30.0)
    properties.SetColor(*color)
    return actor


def wrap_mesh(source, color):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(source)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor = set_default_properties(actor, color)
    return actor, mapper


def create_vtk_image(path: str) -> vtk.vtkImageData:
    root, name, _ = files_utils.split_path(path)
    cache_image_path = f"{root}/cache/{name}.vtk"
    if not files_utils.is_file(cache_image_path):
        np_image = files_utils.load_image(path, 'RGBA')
        image = vtk.vtkImageData()
        image.SetDimensions(np_image.shape[1], np_image.shape[0], 1)
        image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, np_image.shape[2])
        dims = image.GetDimensions()
        for y in range(dims[1]):
            for x in range(dims[0]):
                pixel = np_image[dims[1] - 1 - y, x]
                for i in range(np_image.shape[2]):
                    image.SetScalarComponentFromDouble(x, y, 0, i, pixel[i])
        # points = image.GetPointData().GetArray(0)
        save_vtk_image(image, cache_image_path)
    else:
        image = load_vtk_image(cache_image_path)
    return image


class ImageButton(vtk.vtkButtonWidget):


    def process_state_change_event(self, obj, event):
        print(f"end event {self.button_representation.GetState()}")

    def set_size(self, window_size: Tuple[float, float]):
        w, h = window_size[0] * self.full_size[0], window_size[1] * self.full_size[1]
        pos_left, pos_top = int(w * self.position[0]), int(h * self.position[1])
        position_coords = [pos_left,
                           pos_left + int(w * self.size[0]),
                           pos_top - int(h *self.size[1]),
                           pos_top,
                           0, 0]
        self.button_representation.PlaceWidget(position_coords)

    def resize_event(self, obj, event):
        self.set_size(obj.GetSize())

    def __init__(self, images_paths: List[str], interactor, render, size: Union[float, Tuple[float, float]],
                 position: Tuple[float, float], on_click: Optional[Callable[[Any, Any], None]] = None,
                 full_size: Tuple[float, float] = (1., 1.)):
        super(ImageButton, self).__init__()
        self.SetCurrentRenderer(render)
        if type(size) is float:
            size = (size, size)
        self.full_size = full_size
        render_window: vtk.vtkRenderWindow = interactor.GetRenderWindow()
        images = map(lambda x: create_vtk_image(x), images_paths)
        self.button_representation = vtk.vtkTexturedButtonRepresentation2D()
        self.button_representation.SetNumberOfStates(len(images_paths))
        self.button_representation.GetProperty().SetColor(1, 1, 1)
        for i, image in enumerate(images):
            self.button_representation.SetButtonTexture(i, image)
        self.SetInteractor(interactor)
        self.SetRepresentation(self.button_representation)
        self.size = size
        self.position = position
        self.button_representation.SetPlaceFactor(1)
        self.set_size(render_window.GetSize())
        render_window.AddObserver(vtk.vtkCommand.WindowResizeEvent, self.resize_event)
        if on_click is not None:
            self.AddObserver(vtk.vtkCommand.StateChangedEvent, on_click)
        self.On()
        selection_prop = self.button_representation.GetSelectingProperty()
        selection_prop.SetLineWidth(0.)
        selection_prop.SetColor(1., 1., 1.)


def make_slider(iren, observer):
    to_show = False
    if to_show:
        ren_left = vtk.vtkRenderer()
        ren_left.SetBackground(*rgb_to_float((250, 255, 255)))
        ren_window = vtk.vtkRenderWindow()
        ren_window.AddRenderer(ren_left)
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(ren_window)
        ren_window.Render()
    slider_repres = vtk.vtkSliderRepresentation2D()
    slider_repres.SetMinimumValue(0)
    slider_repres.SetMaximumValue(100.)
    # slider_repres.SetTitleText('Mesh\nOpacity')
    slider_repres.SetValue(30.)
    slider_repres.GetSliderProperty().SetColor(*rgb_to_float(bg_target_color))
    slider_repres.ShowSliderLabelOff()
    # slider_repres.GetLabelProperty().SetColor(1., 0., 0.)
    slider_repres.GetCapProperty().SetColor(*rgb_to_float(bg_menu_color))
    slider_repres.GetSelectedProperty().SetColor(1., 0., 0)
    slider_repres.GetTubeProperty().SetColor(*rgb_to_float(bg_source_color))
    slider_repres.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_repres.GetPoint1Coordinate().SetValue(0.01, 0.1)
    slider_repres.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_repres.GetPoint2Coordinate().SetValue(0.23, 0.1)
    slider_repres.SetSliderLength(0.01)
    slider_repres.SetSliderWidth(0.01)
    slider_repres.SetEndCapLength(0.01)
    slider_repres.SetEndCapWidth(0.01)
    slider_repres.SetTubeWidth(0.01)
    slider_repres.SetLabelFormat('%f')
    slider_widget = vtk.vtkSliderWidget()
    slider_widget.SetInteractor(iren)
    slider_widget.SetRepresentation(slider_repres)
    slider_widget.KeyPressActivationOff()
    slider_widget.SetAnimationModeToAnimate()
    slider_widget.SetEnabled(True)
    slider_widget.AddObserver('InteractionEvent', observer)
    slider_widget.EnabledOn()

    if to_show:
        iren.Initialize()
        ren_window.Render()
        iren.Start()
        del iren
        del ren_window
    return slider_widget, slider_repres


class CanvasRender(vtk.vtkRenderer):

    @property
    def origin_x(self):
        return self.viewport_ren[0]

    @property
    def origin_y(self):
        return self.viewport_ren[1]

    @property
    def width(self):
        return self.viewport_ren[2] - self.viewport_ren[0]

    @property
    def height(self):
        return self.viewport_ren[3] - self.viewport_ren[1]

    def translate_point(self, pt: Tuple[int, int]) -> Tuple[int, int]:
        return pt[0] - self.origin_x, pt[1] - self.origin_y

    def get_mid_points(self, pt: Tuple[int, int]) -> List[List[int]]:
        if self.last_point is None:
            return []
        pt_a, pt_b = torch.tensor(pt, dtype=torch.float32), torch.tensor(self.last_point, dtype=torch.float32)
        delta = pt_b - pt_a
        num_mids = max(int(delta.norm(2, 0).item() / 10), 2)
        # num_mids = 4
        mid_points = pt_a[None, :] + torch.linspace(0, 1, num_mids)[:, None] * delta[None, :]
        mid_points[:, 0] += self.origin_x
        mid_points[:, 1] += self.origin_y
        return mid_points[:-1].long().tolist()

    def draw(self, pt: Tuple[int, int], stroke_width: float = 5.) -> List[List[int]]:
        pt = self.translate_point(pt)
        if self.last_point is not None:
            self.canvas.FillTube(*self.last_point, *pt, stroke_width)
            self.canvas.Update()
        mid_points = self.get_mid_points(pt)
        self.last_point = pt
        return mid_points

    def clear(self):
        self.last_point = None
        self.canvas.SetDrawColor(0, 0, 0, 0)
        self.canvas.FillBox(0, self.width, 0, self.height)
        self.canvas.SetDrawColor(*self.stroke_color)
        self.canvas.Update()

    def resize_event_(self, obj):
        self.viewport_ren = self.set_int_viewport(obj.GetSize())
        self.canvas.SetExtent(0, self.width, 0, self.height, 0, 0)
        self.canvas.Update()
        self.clear()
        self.set_camera()

    def resize_event(self, obj, event):
        self.resize_event_(obj)

    def set_camera(self):
        origin = self.image_data.GetOrigin()
        spacing = self.image_data.GetSpacing()
        extent = self.image_data.GetExtent()
        camera = self.canvas_render.GetActiveCamera()
        camera.ParallelProjectionOn()
        xc = origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0]
        yc = origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1]
        # xd = (extent[1] - extent[0] + 1) * spacing[0]
        yd = (extent[3] - extent[2] + 1) * spacing[1]
        d = camera.GetDistance()
        camera.SetParallelScale(0.5 * yd)
        camera.SetFocalPoint(xc, yc, 0.0)
        camera.SetPosition(xc, yc, d)

    def set_int_viewport(self, win_size) -> Tuple[int, int, int, int]:
        w, h = win_size
        return int(self.viewport[0] * w), int(self.viewport[1] * h), int(self.viewport[2] * w), int(self.viewport[3] * h)

    def init_canvas(self):
        self.canvas.SetExtent(0, self.width, 0, self.height, 0, 0)
        self.canvas.PropagateUpdateExtent()
        self.canvas.UpdateExtent((0, self.width, 0, self.height, 0, 0))
        self.canvas.SetScalarTypeToUnsignedChar()
        self.canvas.SetNumberOfScalarComponents(4)
        self.set_brush(True)
        image_data = self.canvas.GetOutput()
        image_actor = vtk.vtkImageActor()
        image_actor.SetInputData(image_data)
        self.canvas_render.AddActor(image_actor)
        return image_data

    def set_brush(self, is_draw: bool):
        self.is_draw = is_draw
        self.stroke_color = self.base_stroke_color if is_draw else (255, 255, 255, 200)
        # (*bg_menu_color, 150)
        self.canvas.SetDrawColor(*self.stroke_color)
        self.canvas.Update()

    def change_brush(self, stroke_color):
        self.base_stroke_color = stroke_color
        self.set_brush(self.is_draw)

    def __init__(self, viewport: Tuple[float, float, float, float], render_window: vtk.vtkRenderWindow,
                 bg_color: RGB_COLOR, stroke_color: Optional[RGBA_COLOR] = None):
        super(CanvasRender, self).__init__()
        self.SetViewport(*viewport)
        self.viewport = viewport
        self.canvas_render = vtk.vtkRenderer()
        self.canvas_render.SetViewport(*viewport)
        if stroke_color is None:
            stroke_color = vtk.vtkNamedColors().GetColor4ub('LightCoral')
            stroke_color = stroke_color.GetRed(), stroke_color.GetGreen(), stroke_color.GetBlue(), 200
        self.base_stroke_color = self.stroke_color = stroke_color
        # self.SetBackground(*bg_color)
        self.is_draw = True
        self.canvas_render.InteractiveOff()
        self.viewport_ren = self.set_int_viewport(render_window.GetSize())
        self.canvas = vtk.vtkImageCanvasSource2D()
        self.image_data = self.init_canvas()
        render_window.AddObserver(vtk.vtkCommand.WindowResizeEvent, self.resize_event)
        self.last_point: Optional[Tuple[int, int]] = None
        self.SetLayer(0)
        self.canvas_render.SetLayer(1)
        self.SetBackground(*rgb_to_float(bg_color))
        render_window.AddRenderer(self)
        render_window.AddRenderer(self.canvas_render)
        self.set_camera()






def init_palettes(cmap='Spectral'):
    colors = {}
    color_map = plt.cm.get_cmap(cmap)

    def get_palette(num_colors: int) -> T:
        nonlocal colors, color_map
        if num_colors == 1:
            colors[num_colors] = torch.tensor([.45])
        if num_colors not in colors:
            colors[num_colors] = torch.tensor([color_map(float(idx) / (num_colors - 1)) for idx in range(num_colors)])
        return colors[num_colors]

    return get_palette


def get_view_styles(num_styles: int, is_main: bool) -> List[ViewStyle]:
    global palette
    base_color = (255, 255, 255)
    opacity = 1

    colors = init_palettes()(max(num_styles, 100))
    colors = colors[torch.rand(100).argsort()][:num_styles].tolist()
    colors = map(lambda x: list(map(lambda c: int(255 * c), x[:3])), colors)
    # if len(palette_) < num_styles:
    #     palette_ = palette_ + [tuple(item) for item in torch.randint(255, size=(num_styles - len(palette_), 3)).tolist()]
    view_styles = []
    for i, color in enumerate(colors):
        if is_main:
            view_styles.append(ViewStyle(base_color, base_color, color, opacity))
        else:
            view_styles.append(ViewStyle(base_color, color, color, opacity))
    return view_styles

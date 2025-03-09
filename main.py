import functools
import os
import threading
import tkinter as tk
from collections.abc import Callable
from typing import Literal, cast

from PIL import Image, ImageColor, ImageTk

from rwlock import ReadWriteLock


def process_image(
    img: Image.Image, threshold: int = 128, color_as_transp: str = "#0000ff"
):
    r, g, b, a = img.split()
    a_bin = a.point(lambda x: 255 if x >= threshold else 0)
    img = Image.merge("RGBA", (r, g, b, a_bin))
    color = ImageColor.getrgb(color_as_transp)[:3]
    img.putdata([(*color, 255) if item[3] == 0 else item for item in img.getdata()])
    return img


def load_profile(
    conf_path: str, sprites_path: str | None = None, color_as_transp: str = "#0000ff"
):
    if sprites_path is None:
        sprites_path = os.path.splitext(conf_path)[0] + ".png"
    image = Image.open(sprites_path).convert("RGBA")
    data: dict[str, list[Image.Image]] = {}
    with open(conf_path, "r", encoding="utf-8") as fp:
        actions = list(filter(bool, map(str.strip, fp.read().splitlines())))
    h = image.size[1] // len(actions)
    w = image.size[0] // 8
    for y, a in zip(range(0, image.size[1] + 1, h), actions):
        row: list[ImageTk.PhotoImage] = []
        for x in range(0, image.size[0] + 1, w):
            clip = process_image(
                image.crop((x, y, x + w, y + h)), color_as_transp=color_as_transp
            )
            row.append(clip)
        data[a] = row
    return data


def load_whole_dir(dir: str = "artworks"):
    sprites: dict[str, dict[str, list[ImageTk.PhotoImage]]] = {}
    for conf_filename in os.listdir(dir):
        conf_name, ext = os.path.splitext(conf_filename)
        if ext.lower() != ".txt":
            continue
        conf_path = os.path.join(dir, conf_filename)
        sprites_path = os.path.join(dir, conf_name + ".png")
        if not os.path.exists(sprites_path):
            print(f"跳过 <{conf_name}> 因为找不到对应的图像文件")
            continue
        sprites[os.path.splitext(conf_name)[0]] = load_profile(conf_path, sprites_path)
    return sprites


class ImageLabel(tk.Label):
    def update_pic(self, new_pic: ImageTk.PhotoImage):
        self.configure(image=new_pic)
        self.image = new_pic

    @property
    def width(self) -> int:
        image = getattr(self, "image", None)
        return 0 if image is None else cast(ImageTk.PhotoImage, image).width()

    @property
    def height(self) -> int:
        image = getattr(self, "image", None)
        return 0 if image is None else cast(ImageTk.PhotoImage, image).height()


class FLChan:
    class NotActive(Exception):
        """FL Chan is not active yet"""

    class AlreadyActive(Exception):
        """FL Chan is active already"""

    def __init__(
        self,
        conf_path: str,
        sprites_path: str | None = None,
        drag_action: str | None = None,
    ):
        self._is_active = threading.Event()
        self._rwlock = ReadWriteLock()
        self._root: tk.Tk | None = None
        self._label: ImageLabel | None = None
        self._menu: tk.Menu | None = None

        self._color_as_transp = "#0000ff"
        self._sequence: list[ImageTk.PhotoImage] = []
        self._resources = load_profile(
            conf_path, sprites_path, color_as_transp=self._color_as_transp
        )
        self._action = [_ for _ in self._resources.keys()][0]
        self._size = [_ for _ in self._resources.values()][0][0].size
        self._drag_action = drag_action if drag_action else self._action

        self._frame_idx = 0
        self._interval = 100  # ms
        self._drag_data = {
            "x": 0,
            "y": 0,
            "dragging": False,
            "action_store": self._action,
        }
        self._loop_task = None

    @staticmethod
    def __with_lock(rw: Literal["read", "write"]):
        def deco[**P, T](func: Callable[P, T]):
            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs):
                with cast(FLChan, args[0])._rwlock.get_lock(rw):
                    return func(*args, **kwargs)
                # return func(*args, **kwargs)

            return wrapper

        return deco

    @staticmethod
    def __validate_active(require_state: bool):
        def deco[**P, T](func: Callable[P, T]):
            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs):
                if cast(FLChan, args[0])._is_active.is_set() != require_state:
                    raise FLChan.NotActive if require_state else FLChan.AlreadyActive
                return func(*args, **kwargs)

            return wrapper

        return deco

    @property
    def is_active(self):
        return self._is_active.is_set()

    @property
    def action(self):
        return self._action

    @property
    def actions(self):
        return list(self._resources.keys())

    def __load_seq(self):
        self._sequence = [ImageTk.PhotoImage(p) for p in self._resources[self._action]]

    @__validate_active(True)
    @__with_lock("write")
    def set_action(self, val: str):
        if val in self._resources.keys():
            self._action = val
            self.__load_seq()
        else:
            raise KeyError("action not found")

    @property
    def interval(self):
        return self._interval

    @__with_lock("read")
    def set_interval(self, val: int):
        self._interval = int(val)

    @__validate_active(True)
    @__with_lock("write")
    def close(self):
        self._root.destroy()
        self._root = self._label = self._menu = None
        self._sequence = []
        self._is_active.clear()

    @__validate_active(False)
    @__with_lock("write")
    def show(self):
        self._root = r = tk.Tk()
        self._label = ImageLabel(self._root, bg=self._color_as_transp)
        self._label.pack()
        r.geometry("{}x{}".format(*self._size))
        r.wm_attributes("-topmost", True)
        r.wm_attributes("-transparentcolor", self._color_as_transp)
        r.overrideredirect(True)
        r.lift()

        r.bind("<ButtonPress-1>", self.__start_drag)
        r.bind("<ButtonRelease-1>", self.__stop_drag)
        r.bind("<B1-Motion>", self.__on_drag)
        r.bind("<Button-3>", self.__right_click)

        self.__load_seq()
        self._root.after(1, self._switch_frame)
        self._is_active.set()

    def __right_click(self, event: tk.Event):
        menu = tk.Menu(self._root, tearoff=False)
        for action in self.actions:
            menu.add_command(label=action, command=lambda a=action: self.set_action(a))
        menu.add_separator()
        menu.add_command(label="Close", command=self.close)
        menu.post(event.x_root, event.y_root)

    @__validate_active(True)
    def block(self):
        self._root.wait_window(self._root)

    @__with_lock("read")
    def _switch_frame(self):
        if not self._is_active.is_set():
            return
        self._frame_idx = (self._frame_idx + 1) % (len(self._sequence) - 1)
        self._label.update_pic(self._sequence[self._frame_idx])
        self._loop_task = self._root.after(self._interval, self._switch_frame)

    def __start_drag(self, event: tk.Event):
        self._drag_data["x"] = event.x_root
        self._drag_data["y"] = event.y_root
        self._drag_data["dragging"] = True
        if self.action != self._drag_action:
            self._drag_data["action_store"] = self.action
            self.set_action(self._drag_action)

    def __stop_drag(self, _):
        self._drag_data["dragging"] = False
        if self.action != self._drag_data["action_store"]:
            self.set_action(self._drag_data["action_store"])

    def __on_drag(self, event: tk.Event):
        if self._drag_data["dragging"]:
            delta_x = event.x_root - self._drag_data["x"]
            delta_y = event.y_root - self._drag_data["y"]

            x = self._root.winfo_x() + delta_x
            y = self._root.winfo_y() + delta_y
            self._root.geometry(f"+{x}+{y}")

            self._drag_data["x"] = event.x_root
            self._drag_data["y"] = event.y_root


def main():
    flchan = FLChan("artworks/Dance.txt", drag_action="Held")
    flchan.show()
    flchan.block()


if __name__ == "__main__":
    main()

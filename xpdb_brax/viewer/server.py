
# import http.server

# class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
#     def end_headers(self):
#         self.send_my_headers()
#         http.server.SimpleHTTPRequestHandler.end_headers(self)

#     def send_my_headers(self):
#         self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
#         self.send_header("Pragma", "no-cache")
#         self.send_header("Expires", "0")
#         self.send_header("Access-Control-Allow-Origin", "*")


# if __name__ == '__main__':
#     http.server.test(HandlerClass=MyHTTPRequestHandler)

# class Server:
#     pass

from panda3d_viewer import Viewer, ViewerConfig
import imageio

from xpdb_brax.physics.config import Box, Plane, Sphere

import numpy as np

class MyViewer:
    def __init__ (self, make_gif=True):
        self.make_gif = make_gif

    def init (self, config):

        viewer_config = ViewerConfig()
        viewer_config.set_window_size(900, 600)
        viewer_config.enable_antialiasing(True, multisamples=4)

        if self.make_gif:
            viewer_config.enable_shadow(True)
            # viewer_config.show_axes(False)
            # viewer_config.show_grid(False)
            viewer_config.show_floor(False)

            self.writer = imageio.get_writer('anim.gif', mode='I')

        self.viewer = Viewer(window_type='onscreen', window_title='example', config=viewer_config)

        self.viewer.append_group('root')

        for i, body in enumerate(config.bodies):
            if type(body) == Box:
                self.viewer.append_box('root', str(i), size=body.size)
            if type(body) == Plane:
                self.viewer.append_plane('root', str(i), size=(100, 100))
            if type(body) == Sphere:
                # self.viewer.append_sphere('root', str(i), radius=body.radius)
                self.viewer.append_box('root', str(i), size=[body.radius for i in range(3)])


        for i, body in enumerate(config.bodies):
            # self.viewer.set_material('root', str(i), color_rgba=(0.7, 0.1, 0.1, 1))
            self.viewer.set_material('root', str(i), color_rgba=(0.7, 0.7, 0.7, 1))

        self.viewer.reset_camera(pos=(4, 4, 2), look_at=(0, 0, 1))
        # viewer.save_screenshot(filename='box_and_sphere.png')

    def update (self, pos, rot):
        # self.viewer.move_nodes('root', {str(i): (pos[i], np.concatenate((rot[i,3:4], rot[i,0:3]))) for i in range(len(pos)) })
        self.viewer.move_nodes('root', {str(i): (pos[i], rot[i]) for i in range(len(pos)) })

        if self.make_gif:
            image_rgb = self.viewer.get_screenshot(requested_format='RGB')
            self.writer.append_data(image_rgb)



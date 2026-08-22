"""Headless GLSL rendering with moderngl (no node / headless-gl needed).

Renders WebGL 1.0-style fragment shaders (the dialect Lluminate's LLM prompts ask for) by a
small textual transpile to GLSL 330 core, which is what a standalone moderngl context gives
on macOS (CGL) and Linux (EGL / mesa, so it also runs on Modal). The vertex shader maps the
full-screen triangle to `uv = offset + zoom * [0,1]^2`, so a CROP of the shader's canvas can
be rendered without the fragment shader's cooperation -- that is how one static shader yields
many swatches (offset / zoom / seed), which is what makes it a *style* rather than an image.
"""

import re
import threading

import numpy as np

_local = threading.local()

VERT = """#version 330
in vec2 position;
out vec2 uv;
uniform vec2 offset;
uniform float zoom;
void main() {
    uv = offset + zoom * 0.5 * (position + 1.0);
    gl_Position = vec4(position, 0.0, 1.0);
}
"""


def transpile(frag):
    """WebGL 1.0 (GLSL ES 1.00) fragment shader -> GLSL 330 core."""
    s = re.sub(r"^\s*#version[^\n]*\n", "", frag, flags=re.M)
    s = re.sub(r"^\s*precision\s+\w+\s+\w+\s*;\s*$", "", s, flags=re.M)
    s = re.sub(r"\b(lowp|mediump|highp)\s+", "", s)
    s = re.sub(r"\bvarying\b", "in", s)
    s = re.sub(r"\btexture2D\b", "texture", s)
    s = s.replace("gl_FragColor", "fragColor")
    if "in vec2 uv;" not in s:
        s = "in vec2 uv;\n" + s
    return "#version 330\nout vec4 fragColor;\n" + s


def _ctx():
    import moderngl
    if not hasattr(_local, "ctx"):
        _local.ctx = moderngl.create_standalone_context()
    return _local.ctx


def render(frag, width=256, height=256, uniforms=None, offset=(0.0, 0.0), zoom=1.0):
    """Render to an (H, W, 3) uint8 array. Raises on compile/link errors."""
    import moderngl
    ctx = _ctx()
    prog = ctx.program(vertex_shader=VERT, fragment_shader=transpile(frag))
    vbo = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, "position")
    fbo = ctx.simple_framebuffer((width, height))
    try:
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        prog["offset"].value = tuple(float(v) for v in offset)
        prog["zoom"].value = float(zoom)
        for k, v in (uniforms or {}).items():
            if k in prog:
                prog[k].value = v
        vao.render(moderngl.TRIANGLES)
        img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(height, width, 3)[::-1]
        return np.ascontiguousarray(img)
    finally:
        fbo.release(); vao.release(); vbo.release(); prog.release()


def render_to_file(frag, path, width=256, height=256, **kw):
    from PIL import Image
    img = render(frag, width, height, **kw)
    Image.fromarray(img).save(path)
    return path

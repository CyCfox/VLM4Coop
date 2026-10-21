"""Environment-only MPE RGB with fixed, in-circle velocity direction arrows."""
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RENDERER_VERSION = 'pil-mpe-square1440-agent-over-target-speed2dp-v7'
SIZE = 1440
ARROW_RADIUS_FRACTION = .5
AGENT_COLORS = ('#d84b42', '#e58b2a', '#4775c9')  # A0 red, A1 orange, A2 blue.
GOAL_COLOR = '#4c9955'


def render_geometry(world):
    """Preserve world positions/radii; arrow length encodes direction only."""
    lower = np.min([np.asarray(e.state.p_pos)-e.size for e in world.entities], axis=0)
    upper = np.max([np.asarray(e.state.p_pos)+e.size for e in world.entities], axis=0)
    extent = max(float(np.abs(lower).max()), float(np.abs(upper).max()))
    # MPE has no walls. Fit the entire entity envelope into a square camera;
    # retain world aspect ratio and all physical radii, without clipping.
    center = (upper+lower)/2
    side = float((upper-lower).max())
    lower, upper = center-side/2, center+side/2
    scale = (SIZE-2)/side
    width = height = SIZE
    def pixel(position):
        x, y = position
        return (1+(x-lower[0])*scale, 1+(upper[1]-y)*scale)
    agents = []
    for i, agent in enumerate(world.agents):
        start = np.asarray(pixel(agent.state.p_pos))
        radius = float(agent.size*scale)
        velocity = np.asarray(agent.state.p_vel, dtype=float)
        speed = float(np.linalg.norm(velocity))
        tip = start.copy()
        polygon = []
        if speed > 1e-12:
            direction = np.array([velocity[0], -velocity[1]])/speed
            perpendicular = np.array([-direction[1], direction[0]])
            tip = start+direction*ARROW_RADIUS_FRACTION*radius
            # Thick shaft plus broad head, entirely inside the half-radius disk.
            for along, across in ((0,-.07),(.22,-.07),(.22,-.18),(.5,0),
                                  (.22,.18),(.22,.07),(0,.07)):
                polygon.append((start+radius*(along*direction+across*perpendicular)).tolist())
        agents.append(dict(id=i, start=start.tolist(), tip=tip.tolist(), speed=speed,
            name=f'A{i}', label=f'V={speed:.2f}', radius=radius, arrow_polygon=polygon,
            name_center=(start+np.array([0,.60*radius])).tolist(),
            speed_center=(start+np.array([0,.82*radius])).tolist()))
    return dict(extent=extent, scale=scale, image_size=[width,height],
        viewport_min=lower.tolist(), viewport_max=upper.tolist(), agents=agents,
        goals=[dict(id=i, center=list(pixel(g.state.p_pos)), radius=float(g.size*scale))
               for i, g in enumerate(world.landmarks)])


@lru_cache(maxsize=128)
def _font(size):
    return ImageFont.load_default(size=size)


def _centered_text(draw, center, label, size, max_width, max_height, fill='white'):
    size = max(1, int(size))
    while True:
        font = _font(size)
        box = draw.textbbox((0,0), label, font=font)
        width, height = box[2]-box[0], box[3]-box[1]
        if size <= 1 or (width <= max_width and height <= max_height):
            break
        size -= 1
    x, y = center
    draw.text((x-width/2-box[0], y-height/2-box[1]), label, font=font, fill=fill)


def global_rgb(world):
    geometry = render_geometry(world)
    image = Image.new('RGB', tuple(geometry['image_size']), 'white')
    draw = ImageDraw.Draw(image)
    # Four ascending layers; each list is in increasing numeric ID order.
    # 1. Target circles. 2. Target names.
    for goal in geometry['goals']:
        x, y = goal['center']; r = goal['radius']
        draw.ellipse((x-r,y-r,x+r,y+r), fill=GOAL_COLOR, outline='black', width=2)
    for goal in geometry['goals']:
        r = goal['radius']
        _centered_text(draw, goal['center'], f"G{goal['id']}", 1.2*r, 1.65*r, .95*r)
    # 3. Agent circles, covering both target fills and target names.
    for agent in geometry['agents']:
        x, y = agent['start']; r = agent['radius']
        draw.ellipse((x-r,y-r,x+r,y+r), fill=AGENT_COLORS[agent['id']], outline='black', width=2)
    # 4. Agent arrows, names and speed labels.
    for agent in geometry['agents']:
        x, y = agent['start']; r = agent['radius']
        if agent['arrow_polygon']:
            draw.polygon([tuple(p) for p in agent['arrow_polygon']], fill='white')
        else:
            # Zero velocity has no direction: a small point instead of a heading.
            dot = max(1, .04*r)
            draw.ellipse((x-dot,y-dot,x+dot,y+dot), fill='white')
        _centered_text(draw, agent['name_center'], agent['name'], .32*r, 1.2*r, .18*r)
        _centered_text(draw, agent['speed_center'], agent['label'], .28*r, .80*r, .18*r)
    return np.asarray(image)
